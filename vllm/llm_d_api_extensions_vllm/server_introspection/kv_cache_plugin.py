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

`required_tasks` excludes the `render` frontend task. This plugin needs an
engine (there is nothing to introspect on the CPU only render server).
"""

from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from starlette.datastructures import State

from .common import ENGINE_TASKS, cached_response, get_logger, no_engine_detail
from .schemas import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    EncoderOnlyAttentionGroupSpec,
    FullAttentionGroupSpec,
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


def _build_group_spec(group: dict):
    kind = group["kind"]
    model_cls = _KIND_TO_MODEL.get(kind)
    if model_cls is None:
        raise ValueError(f"Unhandled KVCacheSpec kind: {kind!r}")
    return model_cls(**group)


def _build_response(kv_cache_data: dict | None) -> KVCacheResponse:
    if kv_cache_data is None:
        return KVCacheResponse()
    raw_groups = kv_cache_data.get("groups", [])
    return KVCacheResponse(
        kv_cache_size_tokens=kv_cache_data.get("kv_cache_size_tokens"),
        max_concurrency=kv_cache_data.get("max_concurrency"),
        num_gpu_blocks=kv_cache_data.get("num_gpu_blocks"),
        num_cpu_blocks=kv_cache_data.get("num_cpu_blocks"),
        groups=[_build_group_spec(g) for g in raw_groups],
    )


def _capacity_from_vllm_config(vllm_config: "VllmConfig") -> dict:
    cache_cfg = vllm_config.cache_config
    num_gpu_blocks = cache_cfg.num_gpu_blocks
    block_size = cache_cfg.block_size
    kv_cache_size_tokens = (
        num_gpu_blocks * block_size
        if num_gpu_blocks is not None and block_size is not None
        else None
    )
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = (
        kv_cache_size_tokens / max_model_len
        if kv_cache_size_tokens is not None and max_model_len
        else None
    )
    return {
        "kv_cache_size_tokens": kv_cache_size_tokens,
        "max_concurrency": max_concurrency,
        "num_gpu_blocks": num_gpu_blocks,
        "num_cpu_blocks": cache_cfg.num_cpu_blocks,
        "groups": [],
    }


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

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        if engine_client is None:
            state.server_kv_cache_response = None
            return

        kv_cache_data = _capacity_from_vllm_config(state.vllm_config)
        if hasattr(engine_client, "get_kv_cache_group_metadata"):
            kv_cache_data["groups"] = await engine_client.get_kv_cache_group_metadata()

        response = _build_response(kv_cache_data)
        state.server_kv_cache_response = response
        logger.info(
            "kv_cache plugin initialized: num_gpu_blocks=%s groups=%d",
            response.num_gpu_blocks,
            len(response.groups),
        )
