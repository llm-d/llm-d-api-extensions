# SPDX-License-Identifier: Apache-2.0
"""`vllm.endpoint_plugins` entry point: `GET /plugins/llm-d-server-introspection/config`.

Operator supplied, config time values of how the server was launched.
Nothing profiled or derived from model internals.

`required_tasks` is `None`. This plugin needs no engine, so it is also
eligible on the CPU only render server (`init_state` receives
`engine_client=None` there which is fine since it is never touched).

Every value is read through `read_attr`/`build_tolerant` (see `common.py`),
so a config attribute vLLM renames or removes is reported as `null` instead
of aborting server startup.
"""

import os
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from starlette.datastructures import State

from .common import (
    MISSING,
    build_tolerant,
    cached_response,
    get_logger,
    guarded_init_state,
    read_attr,
    report_drift,
)
from .schemas import (
    FeaturesInfo,
    KVCacheInfo,
    KVTransferInfo,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient

logger = get_logger(__name__)


# `kv_connector_extra_config` is free form and operator controlled, so it may
# hold credentials. Values are redacted (keys kept) unless this env var is "1".
EXPOSE_EXTRA_CONFIG_ENV = "LLM_D_INTROSPECTION_EXPOSE_KV_EXTRA_CONFIG"
REDACTED = "<redacted>"

_UNAVAILABLE_DETAIL = "config is unavailable on this server"


def _redact_extra_config(extra_config: object) -> dict:
    if not isinstance(extra_config, Mapping):
        return {}
    if os.environ.get(EXPOSE_EXTRA_CONFIG_ENV) == "1":
        return dict(extra_config)
    return {key: REDACTED for key in extra_config}


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _dtype_str(dtype: object) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _resolve_kv_cache_dtype(cache_dtype: object, model_dtype_str: str | None) -> str | None:
    # "auto" must be resolved to the concrete model dtype, never left as "auto".
    return model_dtype_str if cache_dtype == "auto" else _str_or_none(cache_dtype)


def _is_configured(vllm_config: "VllmConfig | None", name: str) -> bool | None:
    # `None` (unknown) rather than `False` when vLLM no longer has the attribute.
    value = read_attr(vllm_config, name, MISSING)
    return None if value is MISSING else value is not None


def _as_name_list(names: object) -> list[str]:
    if not names:
        return []
    if isinstance(names, str):
        return [names]
    return [str(name) for name in names]


def _served_names(vllm_config: "VllmConfig | None", args: Namespace) -> list[str]:
    served_names = _as_name_list(getattr(args, "served_model_name", None))
    if served_names:
        return served_names
    return _as_name_list(read_attr(read_attr(vllm_config, "model_config"), "served_model_name"))


def _build_kv_transfer_info(vllm_config: "VllmConfig | None") -> KVTransferInfo | None:
    kv_transfer_cfg = read_attr(vllm_config, "kv_transfer_config")
    if kv_transfer_cfg is None:
        return None

    kv_connector = read_attr(kv_transfer_cfg, "kv_connector")
    nixl_host = None
    nixl_port = None
    if kv_connector == "NixlConnector":
        try:
            from vllm import envs
        except Exception:
            report_drift("vllm.envs is not importable")
            envs = None

        nixl_host = read_attr(envs, "VLLM_NIXL_SIDE_CHANNEL_HOST")
        nixl_port = read_attr(envs, "VLLM_NIXL_SIDE_CHANNEL_PORT")

    return build_tolerant(
        KVTransferInfo,
        kv_connector=kv_connector,
        kv_role=read_attr(kv_transfer_cfg, "kv_role"),
        kv_connector_module_path=read_attr(kv_transfer_cfg, "kv_connector_module_path"),
        kv_buffer_device=read_attr(kv_transfer_cfg, "kv_buffer_device"),
        kv_port=read_attr(kv_transfer_cfg, "kv_port"),
        engine_id=read_attr(kv_transfer_cfg, "engine_id"),
        extra_config=_redact_extra_config(read_attr(kv_transfer_cfg, "kv_connector_extra_config")),
        nixl_side_channel_host=nixl_host,
        nixl_side_channel_port=nixl_port,
    )


def _build_response(
    vllm_config: "VllmConfig | None", served_names: list[str]
) -> ServerConfigResponse:
    model_cfg = read_attr(vllm_config, "model_config")
    cache_cfg = read_attr(vllm_config, "cache_config")
    scheduler_cfg = read_attr(vllm_config, "scheduler_config")
    parallel_cfg = read_attr(vllm_config, "parallel_config")

    model_dtype_str = _dtype_str(read_attr(model_cfg, "dtype"))

    return ServerConfigResponse(
        model=build_tolerant(
            ModelInfo,
            name=served_names[0] if served_names else None,
            served_names=served_names,
            dtype=model_dtype_str,
            quantization=_str_or_none(read_attr(model_cfg, "quantization")),
            max_model_len=read_attr(model_cfg, "max_model_len"),
        ),
        kv_cache=build_tolerant(
            KVCacheInfo,
            gpu_memory_utilization=read_attr(cache_cfg, "gpu_memory_utilization"),
            dtype=_resolve_kv_cache_dtype(read_attr(cache_cfg, "cache_dtype"), model_dtype_str),
            enable_prefix_caching=read_attr(cache_cfg, "enable_prefix_caching"),
        ),
        scheduler=build_tolerant(
            SchedulerInfo,
            max_num_seqs=read_attr(scheduler_cfg, "max_num_seqs"),
            max_num_batched_tokens=read_attr(scheduler_cfg, "max_num_batched_tokens"),
            enable_chunked_prefill=read_attr(scheduler_cfg, "enable_chunked_prefill"),
            policy=_str_or_none(read_attr(scheduler_cfg, "policy")),
        ),
        parallelism=build_tolerant(
            ParallelismInfo,
            tensor_parallel_size=read_attr(parallel_cfg, "tensor_parallel_size"),
            pipeline_parallel_size=read_attr(parallel_cfg, "pipeline_parallel_size"),
            data_parallel_size=read_attr(parallel_cfg, "data_parallel_size"),
            data_parallel_rank=read_attr(parallel_cfg, "data_parallel_rank"),
        ),
        features=build_tolerant(
            FeaturesInfo,
            speculative_decoding=_is_configured(vllm_config, "speculative_config"),
            lora=_is_configured(vllm_config, "lora_config"),
            # Plain `getattr`: older vLLM has no such field and HMA is then on.
            hma=(
                not bool(getattr(scheduler_cfg, "disable_hybrid_kv_cache_manager", False))
                if scheduler_cfg is not None
                else None
            ),
        ),
        kv_transfer=_build_kv_transfer_info(vllm_config),
    )


class ServerConfigPlugin:
    name = "llm_d_server_introspection_config"
    required_tasks: tuple[str, ...] | None = None

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get(
            "/plugins/llm-d-server-introspection/config",
            response_model=ServerConfigResponse,
        )
        async def get_server_config(raw_request: Request) -> ServerConfigResponse:
            return cached_response(raw_request, "server_config_response", _UNAVAILABLE_DETAIL)

        app.include_router(router)

    @guarded_init_state("server_config_response", "config")
    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        vllm_config: VllmConfig | None = read_attr(state, "vllm_config")
        served_names = _served_names(vllm_config, args)
        response = _build_response(vllm_config, served_names)
        state.server_config_response = response
        logger.info(
            "config plugin initialized: served_names=%s kv_transfer=%s",
            served_names,
            "configured" if response.kv_transfer is not None else "none",
        )
