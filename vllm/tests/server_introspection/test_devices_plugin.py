# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `GET /plugins/llm-d-server-introspection/devices`.

Fast with no engine and no real vLLM/model. `EngineClient` is a fake engine
exercising only `collective_rpc`, matching the pattern vLLM's own
`tests/plugins_tests/test_endpoint_plugins.py` uses for its `_FakeEngineClient`.
"""

import asyncio
from argparse import Namespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_d_api_extensions_vllm.server_introspection.device_worker_ext import (
    GET_DEVICE_PROPERTIES,
    DeviceInfoWorkerExtension,
)
from llm_d_api_extensions_vllm.server_introspection.devices_plugin import (
    ServerDevicesPlugin,
    _build_response,
)
from llm_d_api_extensions_vllm.server_introspection.schemas import (
    ComputeCapability,
    DeviceInfo,
    DevicesResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_A100_ENTRY = {
    "rank": 0,
    "name": "A100-PCIE-40GB",
    "total_memory_bytes": 42_949_672_960,
    "compute_capability": {"major": 8, "minor": 0},
    "num_compute_units": 108,
}

_H100_ENTRY = {
    "rank": 1,
    "name": "H100-SXM5-80GB",
    "total_memory_bytes": 85_899_345_920,
    "compute_capability": {"major": 9, "minor": 0},
    "num_compute_units": 132,
}

_NO_CAPABILITY_ENTRY = {
    "rank": 0,
    "name": "SomeNonCudaDevice",
    "total_memory_bytes": 8_589_934_592,
    "compute_capability": None,
    "num_compute_units": None,
}


class _FakeEngineClient:
    """Minimal stand in exercising `collective_rpc`. Not a real engine."""

    def __init__(self, rpc_result: Any = None, rpc_error: Exception | None = None):
        self.rpc_result = rpc_result
        self.rpc_error = rpc_error
        self.rpc_calls: list[tuple[str, tuple, dict]] = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args, kwargs or {}))
        if self.rpc_error is not None:
            raise self.rpc_error
        return self.rpc_result


def _make_test_app(engine_client: "_FakeEngineClient | None") -> FastAPI:
    """FastAPI app with the plugin's route + state wired through both phases."""
    app = FastAPI()
    plugin = ServerDevicesPlugin()
    plugin.attach_router(app)
    asyncio.run(plugin.init_state(engine_client, app.state, Namespace()))
    return app


# ---------------------------------------------------------------------------
# _build_response unit tests
# ---------------------------------------------------------------------------


class TestBuildResponse:
    def test_single_device(self):
        result = _build_response([_A100_ENTRY])
        assert isinstance(result, DevicesResponse)
        assert len(result.devices) == 1
        d = result.devices[0]
        assert d.rank == 0
        assert d.name == "A100-PCIE-40GB"
        assert d.total_memory_bytes == 42_949_672_960
        assert d.compute_capability == ComputeCapability(major=8, minor=0)
        assert d.num_compute_units == 108

    def test_multi_rank(self):
        result = _build_response([_A100_ENTRY, _H100_ENTRY])
        assert len(result.devices) == 2
        assert result.devices[0].rank == 0
        assert result.devices[1].rank == 1
        assert result.devices[1].compute_capability == ComputeCapability(major=9, minor=0)

    def test_null_compute_capability_and_num_compute_units(self):
        result = _build_response([_NO_CAPABILITY_ENTRY])
        d = result.devices[0]
        assert d.compute_capability is None
        assert d.num_compute_units is None

    def test_empty_list(self):
        result = _build_response([])
        assert result.devices == []

    def test_preserves_input_order(self):
        entries = [
            {**_A100_ENTRY, "rank": 3},
            {**_A100_ENTRY, "rank": 1},
            {**_A100_ENTRY, "rank": 0},
        ]
        result = _build_response(entries)
        assert [d.rank for d in result.devices] == [3, 1, 0]

    def test_returns_device_info_instances(self):
        result = _build_response([_A100_ENTRY])
        assert isinstance(result.devices[0], DeviceInfo)


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetDevicesEndpoint:
    def test_returns_200_with_devices_from_engine_client(self):
        fake_client = _FakeEngineClient(rpc_result=[_A100_ENTRY, _H100_ENTRY])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["devices"]) == 2
        assert data["devices"][0]["name"] == "A100-PCIE-40GB"

    def test_calls_collective_rpc_with_get_device_properties(self):
        fake_client = _FakeEngineClient(rpc_result=[_A100_ENTRY])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            client.get("/plugins/llm-d-server-introspection/devices")
        assert fake_client.rpc_calls == [("get_device_properties", (), {})]

    def test_response_cached_not_rebuilt_per_request(self):
        fake_client = _FakeEngineClient(rpc_result=[_A100_ENTRY])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            client.get("/plugins/llm-d-server-introspection/devices")
            client.get("/plugins/llm-d-server-introspection/devices")
        # collective_rpc only happens once, during init_state.
        assert len(fake_client.rpc_calls) == 1

    def test_engine_client_none_returns_503(self):
        app = _make_test_app(None)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 503
        assert "requires an engine" in resp.json()["detail"]

    def test_missing_worker_extension_does_not_crash_init_and_returns_503(self):
        # vLLM raises from collective_rpc when no worker has the method.
        fake_client = _FakeEngineClient(
            rpc_error=AttributeError("'Worker' object has no attribute 'get_device_properties'")
        )
        app = _make_test_app(fake_client)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 503
        assert "--worker-extension-cls" in resp.json()["detail"]

    def test_missing_state_returns_500(self):
        # attach_router without ever calling init_state.
        # The handler must not crash on a missing attribute.
        app = FastAPI()
        ServerDevicesPlugin().attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 500

    def test_empty_devices_list_returns_200(self):
        fake_client = _FakeEngineClient(rpc_result=[])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 200
        assert resp.json() == {"devices": []}

    def test_content_type_is_json(self):
        fake_client = _FakeEngineClient(rpc_result=[_A100_ENTRY])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        assert "application/json" in resp.headers["content-type"]

    def test_wrong_method_returns_405(self):
        app = _make_test_app(_FakeEngineClient(rpc_result=[]))
        with TestClient(app) as client:
            resp = client.post("/plugins/llm-d-server-introspection/devices")
        assert resp.status_code == 405

    def test_null_capability_device_serializes_correctly(self):
        fake_client = _FakeEngineClient(rpc_result=[_NO_CAPABILITY_ENTRY])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/devices")
        data = resp.json()
        assert data["devices"][0]["compute_capability"] is None
        assert data["devices"][0]["num_compute_units"] is None


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


class TestRpcMethodName:
    def test_worker_extension_defines_rpc_method(self):
        assert callable(getattr(DeviceInfoWorkerExtension, GET_DEVICE_PROPERTIES, None))

    def test_plugin_calls_shared_rpc_method_name(self):
        fake_client = _FakeEngineClient(rpc_result=[])
        _make_test_app(fake_client)
        assert fake_client.rpc_calls[0][0] == GET_DEVICE_PROPERTIES


class TestPluginMetadata:
    def test_name(self):
        assert ServerDevicesPlugin().name == "llm_d_server_introspection_devices"

    def test_required_tasks_excludes_render(self):
        assert "render" not in ServerDevicesPlugin().required_tasks

    def test_required_tasks_includes_generate(self):
        assert "generate" in ServerDevicesPlugin().required_tasks
