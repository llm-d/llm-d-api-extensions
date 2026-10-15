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
    report_drift,
)
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


def _build_device(raw_device: object) -> DeviceInfo | None:
    if not isinstance(raw_device, Mapping):
        report_drift(f"device properties entry is a {type(raw_device).__name__}, not a dict")
        return None
    cap = raw_device.get("compute_capability")
    return build_tolerant(
        DeviceInfo,
        rank=raw_device.get("rank"),
        name=raw_device.get("name"),
        total_memory_bytes=raw_device.get("total_memory_bytes"),
        compute_capability=(
            build_tolerant(ComputeCapability, major=cap.get("major"), minor=cap.get("minor"))
            if isinstance(cap, Mapping)
            else None
        ),
        num_compute_units=raw_device.get("num_compute_units"),
    )


def _build_response(raw_devices: object) -> DevicesResponse:
    if not isinstance(raw_devices, list | tuple):
        report_drift(
            f"collective_rpc({GET_DEVICE_PROPERTIES!r}) returned a "
            f"{type(raw_devices).__name__}, not a list"
        )
        raw_devices = []
    devices = [device for d in raw_devices if (device := _build_device(d)) is not None]
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
            return cached_response(raw_request, "server_devices_response", _NO_ENGINE_DETAIL)

        app.include_router(router)

    @guarded_init_state("server_devices_response", "devices")
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
