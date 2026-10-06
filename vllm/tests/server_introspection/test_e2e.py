# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for `GET /plugins/llm-d-server-introspection/{config,devices,kv-cache}`
against a real `vllm serve`.

Mirrors `tests/plugins_tests/test_endpoint_plugins.py` in vllm-project/vllm:
`pip install -e` this package, launch a tiny model with `VLLM_PLUGINS` set,
assert a real HTTP 200 + schema.

Unlike the unit tests in `test_config_plugin.py`/`test_devices_plugin.py`/
`test_kv_cache_plugin.py`, this needs `vllm` importable and downloads model
weights over the network, so it is opt-in:

    pip install -e .[test]
    RUN_VLLM_E2E=1 pytest tests/server_introspection/test_e2e.py -v

It is skipped by default (no `vllm` install, no `RUN_VLLM_E2E`) so `pytest`
without extra setup only runs the fast unit tests.
"""

import contextlib
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytest.importorskip("vllm")

if not os.environ.get("RUN_VLLM_E2E"):
    pytest.skip(
        "set RUN_VLLM_E2E=1 to run the real server e2e test (downloads model "
        "weights, launches a subprocess server)",
        allow_module_level=True,
    )

MODEL = "facebook/opt-125m"
STARTUP_TIMEOUT_S = 300


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _running_server(*, plugins: str | None, worker_extension: bool = False):
    port = _free_port()
    env = dict(os.environ)
    if plugins:
        env["VLLM_PLUGINS"] = plugins
    else:
        env.pop("VLLM_PLUGINS", None)
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(port),
    ]
    if worker_extension:
        args += [
            "--worker-extension-cls",
            "llm_d_api_extensions_vllm.server_introspection.device_worker_ext.DeviceInfoWorkerExtension",
        ]
    proc = subprocess.Popen(args, env=env)
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server process exited early with {proc.returncode}")
            try:
                resp = httpx.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("server did not become healthy in time")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_server_config_endpoint_returns_200_with_valid_schema():
    with _running_server(plugins="llm_d_server_introspection_config") as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/config", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "model",
        "kv_cache",
        "kv_transfer",
        "scheduler",
        "parallelism",
        "features",
    }
    assert data["model"]["name"]
    assert data["parallelism"]["tensor_parallel_size"] == 1


def test_server_config_not_attached_without_allowlist():
    """No VLLM_PLUGINS set -> route must not exist (strict allowlist)."""
    with _running_server(plugins=None) as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/config", timeout=10)

    assert resp.status_code == 404


def test_server_devices_endpoint_returns_200_with_valid_schema():
    with _running_server(
        plugins="llm_d_server_introspection_devices", worker_extension=True
    ) as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/devices", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"devices"}
    assert len(data["devices"]) == 1
    device = data["devices"][0]
    assert device["rank"] == 0
    assert device["name"]
    assert device["total_memory_bytes"] > 0


def test_server_devices_not_attached_without_allowlist():
    """No VLLM_PLUGINS set -> route must not exist (strict allowlist)."""
    with _running_server(plugins=None) as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/devices", timeout=10)

    assert resp.status_code == 404


def test_server_kv_cache_endpoint_returns_200_with_valid_schema():
    with _running_server(plugins="llm_d_server_introspection_kv_cache") as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/kv-cache", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "kv_cache_size_tokens",
        "max_concurrency",
        "num_gpu_blocks",
        "num_cpu_blocks",
        "groups",
    }
    # `EngineClient.get_kv_cache_group_metadata` (vllm-project/vllm#48121)
    # isn't in a released vLLM yet, so this real server exercises the
    # capacity only fallback path which means capacity fields populated, no group
    # structure.
    assert data["num_gpu_blocks"] > 0
    assert data["kv_cache_size_tokens"] > 0
    assert data["groups"] == []


def test_server_kv_cache_not_attached_without_allowlist():
    """No VLLM_PLUGINS set -> route must not exist (strict allowlist)."""
    with _running_server(plugins=None) as base_url:
        resp = httpx.get(f"{base_url}/plugins/llm-d-server-introspection/kv-cache", timeout=10)

    assert resp.status_code == 404
