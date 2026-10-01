# SPDX-License-Identifier: Apache-2.0
"""`vllm.endpoint_plugins` entry point: `GET /plugins/llm-d-server-introspection/devices`.

Per rank hardware properties (name, memory, compute capability, compute
units) gathered once at startup via `collective_rpc` and cached for the
server's lifetime (device properties are static).

Requires the worker side `get_device_properties` method installed by
`device_worker_ext.DeviceInfoWorkerExtension` (see that module's docstring
for the `--worker-extension-cls` flag). Without it, `collective_rpc` raises.
The plugin catches that at `init_state` time, logs a warning and serves `503`
with the reason rather than crashing the server.

`required_tasks` excludes the `render` frontend task. This plugin needs an
engine (there is nothing to introspect on the CPU only render server).
"""

from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from starlette.datastructures import State

from .common import ENGINE_TASKS, cached_response, get_logger, no_engine_detail
from .device_worker_ext import GET_DEVICE_PROPERTIES
from .schemas import ComputeCapability, DeviceInfo, DevicesResponse

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = get_logger(__name__)

_NO_ENGINE_DETAIL = no_engine_detail("devices")
_NO_WORKER_EXT_DETAIL = (
    "devices requires the DeviceInfoWorkerExtension worker extension "
    "(--worker-extension-cls), which is not installed on this server"
)


def _build_response(raw_devices: list[dict]) -> DevicesResponse:
    devices = []
    for d in raw_devices:
        cap = d.get("compute_capability")
        devices.append(
            DeviceInfo(
                rank=d["rank"],
                name=d["name"],
                total_memory_bytes=d["total_memory_bytes"],
                compute_capability=(
                    ComputeCapability(major=cap["major"], minor=cap["minor"])
                    if cap is not None
                    else None
                ),
                num_compute_units=d["num_compute_units"],
            )
        )
    return DevicesResponse(devices=devices)


class ServerDevicesPlugin:
    name = "llm_d_server_introspection_devices"
    required_tasks: tuple[str, ...] | None = ENGINE_TASKS

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get(
            "/plugins/llm-d-server-introspection/devices",
            response_model=DevicesResponse,
        )
        async def get_devices(raw_request: Request) -> DevicesResponse:
            unavailable_detail = getattr(
                raw_request.app.state, "server_devices_unavailable_detail", _NO_ENGINE_DETAIL
            )
            return cached_response(raw_request, "server_devices_response", unavailable_detail)

        app.include_router(router)

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        if engine_client is None:
            state.server_devices_response = None
            state.server_devices_unavailable_detail = _NO_ENGINE_DETAIL
            return
        # Uses DeviceInfoWorkerExtension::get_device_properties() extension to call vLLM engine
        # worker methods
        try:
            raw_devices = await engine_client.collective_rpc(GET_DEVICE_PROPERTIES)
        except Exception:
            logger.warning(
                "devices plugin: collective_rpc(%r) failed "
                "(is --worker-extension-cls set to DeviceInfoWorkerExtension?)",
                GET_DEVICE_PROPERTIES,
                exc_info=True,
            )
            state.server_devices_response = None
            state.server_devices_unavailable_detail = _NO_WORKER_EXT_DETAIL
            return
        response = _build_response(raw_devices)
        state.server_devices_response = response
        logger.info("devices plugin initialized: devices=%d", len(response.devices))
