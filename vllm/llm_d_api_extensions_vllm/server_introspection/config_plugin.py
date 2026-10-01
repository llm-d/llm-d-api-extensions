# SPDX-License-Identifier: Apache-2.0
"""`vllm.endpoint_plugins` entry point: `GET /plugins/llm-d-server-introspection/config`.

Operator supplied, config time values of how the server was launched.
Nothing profiled or derived from model internals.

`required_tasks` is `None`. This plugin needs no engine, so it is also
eligible on the CPU only render server (`init_state` receives
`engine_client=None` there which is fine since it is never touched).
"""

import os
from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import State
from vllm.logger import init_logger

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

# "vllm." prefix required. vLLM's default logging config only attaches a
# handler to the "vllm" logger tree (propagate=False), so a bare __name__
# logger has no handler anywhere and silently drops messages.
logger = init_logger(f"vllm.{__name__}")


# `kv_connector_extra_config` is free form and operator controlled, so it may
# hold credentials. Values are redacted (keys kept) unless this env var is "1".
EXPOSE_EXTRA_CONFIG_ENV = "LLM_D_INTROSPECTION_EXPOSE_KV_EXTRA_CONFIG"
REDACTED = "<redacted>"


def _redact_extra_config(extra_config: dict | None) -> dict:
    extra_config = extra_config or {}
    if os.environ.get(EXPOSE_EXTRA_CONFIG_ENV) == "1":
        return dict(extra_config)
    return {key: REDACTED for key in extra_config}


def _dtype_str(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_kv_cache_dtype(cache_dtype: str, model_dtype_str: str) -> str:
    # "auto" must be resolved to the concrete model dtype, never left as "auto".
    return model_dtype_str if cache_dtype == "auto" else cache_dtype


def _build_kv_transfer_info(vllm_config: "VllmConfig") -> KVTransferInfo | None:
    kv_transfer_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_cfg is None:
        return None

    nixl_host = None
    nixl_port = None
    if kv_transfer_cfg.kv_connector == "NixlConnector":
        from vllm import envs

        nixl_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        nixl_port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT

    return KVTransferInfo(
        kv_connector=kv_transfer_cfg.kv_connector,
        kv_role=kv_transfer_cfg.kv_role,
        kv_connector_module_path=kv_transfer_cfg.kv_connector_module_path,
        kv_buffer_device=kv_transfer_cfg.kv_buffer_device,
        kv_buffer_size=kv_transfer_cfg.kv_buffer_size,
        kv_ip=kv_transfer_cfg.kv_ip,
        kv_port=kv_transfer_cfg.kv_port,
        kv_parallel_size=kv_transfer_cfg.kv_parallel_size,
        kv_rank=kv_transfer_cfg.kv_rank,
        engine_id=kv_transfer_cfg.engine_id,
        extra_config=_redact_extra_config(kv_transfer_cfg.kv_connector_extra_config),
        nixl_side_channel_host=nixl_host,
        nixl_side_channel_port=nixl_port,
    )


def _build_response(vllm_config: "VllmConfig", served_names: list[str]) -> ServerConfigResponse:
    model_cfg = vllm_config.model_config
    cache_cfg = vllm_config.cache_config
    scheduler_cfg = vllm_config.scheduler_config
    parallel_cfg = vllm_config.parallel_config
    spec_cfg = vllm_config.speculative_config
    lora_cfg = vllm_config.lora_config

    model_dtype_str = _dtype_str(model_cfg.dtype)

    return ServerConfigResponse(
        model=ModelInfo(
            name=served_names[0],
            served_names=served_names,
            dtype=model_dtype_str,
            quantization=(
                str(model_cfg.quantization) if model_cfg.quantization is not None else None
            ),
            max_model_len=model_cfg.max_model_len,
        ),
        kv_cache=KVCacheInfo(
            gpu_memory_utilization=cache_cfg.gpu_memory_utilization,
            dtype=_resolve_kv_cache_dtype(cache_cfg.cache_dtype, model_dtype_str),
            enable_prefix_caching=cache_cfg.enable_prefix_caching,
        ),
        scheduler=SchedulerInfo(
            max_num_seqs=scheduler_cfg.max_num_seqs,
            max_num_batched_tokens=scheduler_cfg.max_num_batched_tokens,
            enable_chunked_prefill=scheduler_cfg.enable_chunked_prefill,
            policy=str(scheduler_cfg.policy),
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=parallel_cfg.tensor_parallel_size,
            pipeline_parallel_size=parallel_cfg.pipeline_parallel_size,
            data_parallel_size=parallel_cfg.data_parallel_size,
            data_parallel_rank=parallel_cfg.data_parallel_rank,
        ),
        features=FeaturesInfo(
            speculative_decoding=spec_cfg is not None,
            lora=lora_cfg is not None,
            hma=not bool(getattr(scheduler_cfg, "disable_hybrid_kv_cache_manager", False)),
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
            response = getattr(raw_request.app.state, "server_config_response", None)
            if response is None:
                raise HTTPException(
                    status_code=500,
                    detail="server_config plugin state was never initialized",
                )
            return response

        app.include_router(router)

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        vllm_config: VllmConfig = state.vllm_config
        if args.served_model_name:
            served_names = list(args.served_model_name)
        else:
            served_names = [vllm_config.model_config.served_model_name]
        response = _build_response(vllm_config, served_names)
        state.server_config_response = response
        logger.info(
            "config plugin initialized: served_names=%s kv_transfer=%s",
            served_names,
            "configured" if response.kv_transfer is not None else "none",
        )
