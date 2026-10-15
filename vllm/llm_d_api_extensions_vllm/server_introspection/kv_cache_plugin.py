# SPDX-License-Identifier: Apache-2.0
"""`vllm.endpoint_plugins` entry point: `GET /plugins/llm-d-server-introspection/kv-cache`.

Post profiling KV cache capacity plus attention group structure. Capacity
fields (`kv_cache_size_tokens`, `max_concurrency`, `num_gpu_blocks`,
`num_cpu_blocks`) are read directly off `vllm_config.cache_config` &
`model_config` which are already post profiling values by the time
`init_state` runs (`EngineCore` mutates the shared config in place during
`_initialize_kv_caches`).

`groups` is sourced separately from `engine_client.get_kv_cache_group_metadata()`
once at startup and cached for the server's lifetime (immutable once
profiling has run). That method is not yet part of a released `EngineClient`,
it is proposed upstream in vllm-project/vllm#48121 which this plugin's
group schema (`kind` values are the `KVCacheSpecKind` strings that PR
serializes groups under) and response shape are ported from. Feature
detected via `hasattr` so this package still installs against a vLLM build
that predates that method which means `groups` are empty for this.

vLLM's group dicts are not a stable contract. A failing call, a group of a
`kind` unknown here or a group missing fields degrades (empty `groups`,
`UnknownGroupSpec`, `null` fields) instead of aborting server startup.

`required_tasks` excludes the `render` frontend task. This plugin needs an
engine (there is nothing to introspect on the CPU only render server).
"""

import inspect
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from starlette.datastructures import State

from .common import (
    ENGINE_TASKS,
    build_tolerant,
    cached_response,
    get_logger,
    guarded_init_state,
    no_engine_detail,
    read_attr,
    report_drift,
)
from .schemas import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    EncoderOnlyAttentionGroupSpec,
    FullAttentionGroupSpec,
    KVCacheGroupSpec,
    KVCacheResponse,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    SlidingWindowMLAGroupSpec,
    UnknownGroupSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient

logger = get_logger(__name__)

GET_KV_CACHE_GROUP_METADATA = "get_kv_cache_group_metadata"

_KIND_TO_MODEL: dict[str, type] = {
    "full_attention": FullAttentionGroupSpec,
    "mla_attention": MLAAttentionGroupSpec,
    "sliding_window": SlidingWindowGroupSpec,
    "sliding_window_mla": SlidingWindowMLAGroupSpec,
    "chunked_local_attention": ChunkedLocalAttentionGroupSpec,
    "mamba": MambaGroupSpec,
    "cross_attention": CrossAttentionGroupSpec,
    "encoder_only_attention": EncoderOnlyAttentionGroupSpec,
    "sink_full_attention": SinkFullAttentionGroupSpec,
    "unknown": UnknownGroupSpec,
}


def _build_group_spec(group: object) -> KVCacheGroupSpec | None:
    """Build the spec for one vLLM group dict, or `None` when it is unusable.

    A `kind` with no schema here (vLLM added a new `KVCacheSpecKind`) becomes
    an `UnknownGroupSpec` carrying the original string in `vllm_kind`.
    """
    if not isinstance(group, Mapping):
        report_drift(f"KV cache group metadata entry is a {type(group).__name__}, not a dict")
        return None
    kind = group.get("kind")
    model_cls = _KIND_TO_MODEL.get(kind) if isinstance(kind, str) else None
    if model_cls is None:
        report_drift(f"KV cache group kind {kind!r} is not known")
        fields = {**group, "kind": "unknown", "vllm_kind": _str_or_none(kind)}
        return build_tolerant(UnknownGroupSpec, **fields)
    return build_tolerant(model_cls, **group)


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _build_response(kv_cache_data: dict | None) -> KVCacheResponse:
    if kv_cache_data is None:
        return KVCacheResponse()
    raw_groups = kv_cache_data.get("groups") or []
    groups = [spec for g in raw_groups if (spec := _build_group_spec(g)) is not None]
    return build_tolerant(
        KVCacheResponse,
        kv_cache_size_tokens=kv_cache_data.get("kv_cache_size_tokens"),
        max_concurrency=kv_cache_data.get("max_concurrency"),
        num_gpu_blocks=kv_cache_data.get("num_gpu_blocks"),
        num_cpu_blocks=kv_cache_data.get("num_cpu_blocks"),
        groups=groups,
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _capacity_from_vllm_config(vllm_config: "VllmConfig | None") -> dict:
    cache_cfg = read_attr(vllm_config, "cache_config")
    num_gpu_blocks = read_attr(cache_cfg, "num_gpu_blocks")
    block_size = read_attr(cache_cfg, "block_size")
    kv_cache_size_tokens = (
        num_gpu_blocks * block_size
        if _is_number(num_gpu_blocks) and _is_number(block_size)
        else None
    )
    max_model_len = read_attr(read_attr(vllm_config, "model_config"), "max_model_len")
    max_concurrency = (
        kv_cache_size_tokens / max_model_len
        if kv_cache_size_tokens is not None and _is_number(max_model_len) and max_model_len
        else None
    )
    return {
        "kv_cache_size_tokens": kv_cache_size_tokens,
        "max_concurrency": max_concurrency,
        "num_gpu_blocks": num_gpu_blocks,
        "num_cpu_blocks": read_attr(cache_cfg, "num_cpu_blocks"),
        "groups": [],
    }


async def _fetch_groups(engine_client: "EngineClient") -> list:
    """Group metadata from the engine, or `[]` when it can't be obtained."""
    # Plain `getattr`: the method is absent on vLLM builds that predate it.
    method = getattr(engine_client, GET_KV_CACHE_GROUP_METADATA, None)
    if method is None:
        return []
    try:
        groups = method()
        if inspect.isawaitable(groups):
            groups = await groups
    except Exception:
        logger.warning(
            "kv_cache plugin: %s() failed, reporting capacity only",
            GET_KV_CACHE_GROUP_METADATA,
            exc_info=True,
        )
        return []
    if not isinstance(groups, list | tuple):
        report_drift(
            f"{GET_KV_CACHE_GROUP_METADATA}() returned a {type(groups).__name__}, not a list"
        )
        return []
    return list(groups)


class ServerKVCachePlugin:
    name = "llm_d_server_introspection_kv_cache"
    required_tasks: tuple[str, ...] | None = ENGINE_TASKS

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get(
            "/plugins/llm-d-server-introspection/kv-cache",
            response_model=KVCacheResponse,
        )
        async def get_kv_cache(raw_request: Request) -> KVCacheResponse:
            return cached_response(
                raw_request, "server_kv_cache_response", no_engine_detail("kv-cache")
            )

        app.include_router(router)

    @guarded_init_state("server_kv_cache_response", "kv-cache")
    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        if engine_client is None:
            state.server_kv_cache_response = None
            return

        kv_cache_data = _capacity_from_vllm_config(read_attr(state, "vllm_config"))
        kv_cache_data["groups"] = await _fetch_groups(engine_client)

        response = _build_response(kv_cache_data)
        state.server_kv_cache_response = response
        logger.info(
            "kv_cache plugin initialized: num_gpu_blocks=%s groups=%d",
            response.num_gpu_blocks,
            len(response.groups),
        )
