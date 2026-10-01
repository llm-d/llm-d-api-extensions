# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `GET /plugins/llm-d-server-introspection/config`.

Fast, no engine and no real vLLM/model. `VllmConfig` and its sub configs are
`MagicMock`s so the tests exercise `_build_response` and the FastAPI route in
isolation.
"""

import asyncio
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_d_api_extensions_vllm.server_introspection.config_plugin import (
    EXPOSE_EXTRA_CONFIG_ENV,
    REDACTED,
    ServerConfigPlugin,
    _build_response,
)
from llm_d_api_extensions_vllm.server_introspection.schemas import (
    FeaturesInfo,
    KVCacheInfo,
    KVTransferInfo,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTorchDtype:
    """Stands in for `torch.dtype` without a torch dependency in tests.

    `str(torch.bfloat16) == "torch.bfloat16"`; `_build_response` does
    `str(dtype).removeprefix("torch.")`, so this only needs to match that.
    """

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"


BFLOAT16 = _FakeTorchDtype("bfloat16")
FLOAT16 = _FakeTorchDtype("float16")
FLOAT32 = _FakeTorchDtype("float32")


def _make_vllm_config(
    *,
    model_name: str = "meta-llama/Llama-3.1-8B",
    served_model_name: str = "meta-llama/Llama-3.1-8B",
    model_dtype: _FakeTorchDtype = BFLOAT16,
    quantization: str | None = None,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 128,
    max_num_batched_tokens: int = 2048,
    enable_chunked_prefill: bool = True,
    policy: str = "fcfs",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    disable_hybrid_kv_cache_manager: bool = False,
    speculative_config=None,
    lora_config=None,
    kv_transfer_config=None,
) -> MagicMock:
    model_cfg = MagicMock()
    model_cfg.model = model_name
    model_cfg.served_model_name = served_model_name
    model_cfg.dtype = model_dtype
    model_cfg.quantization = quantization
    model_cfg.max_model_len = max_model_len

    cache_cfg = MagicMock()
    cache_cfg.gpu_memory_utilization = gpu_memory_utilization
    cache_cfg.cache_dtype = cache_dtype
    cache_cfg.enable_prefix_caching = enable_prefix_caching

    scheduler_cfg = MagicMock()
    scheduler_cfg.max_num_seqs = max_num_seqs
    scheduler_cfg.max_num_batched_tokens = max_num_batched_tokens
    scheduler_cfg.enable_chunked_prefill = enable_chunked_prefill
    scheduler_cfg.policy = policy
    scheduler_cfg.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager

    parallel_cfg = MagicMock()
    parallel_cfg.tensor_parallel_size = tensor_parallel_size
    parallel_cfg.pipeline_parallel_size = pipeline_parallel_size
    parallel_cfg.data_parallel_size = data_parallel_size
    parallel_cfg.data_parallel_rank = data_parallel_rank

    vllm_config = MagicMock()
    vllm_config.model_config = model_cfg
    vllm_config.cache_config = cache_cfg
    vllm_config.scheduler_config = scheduler_cfg
    vllm_config.parallel_config = parallel_cfg
    vllm_config.speculative_config = speculative_config
    vllm_config.lora_config = lora_config
    vllm_config.kv_transfer_config = kv_transfer_config

    return vllm_config


def _make_kv_transfer_config(
    *,
    kv_connector: str | None = "NixlConnector",
    kv_role: str | None = "kv_both",
    kv_connector_module_path: str | None = None,
    kv_buffer_device: str = "cuda",
    kv_buffer_size: float = 1e9,
    kv_ip: str = "127.0.0.1",
    kv_port: int = 14579,
    kv_parallel_size: int = 1,
    kv_rank: int | None = None,
    engine_id: str | None = "engine-0",
    kv_connector_extra_config: dict | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.kv_connector = kv_connector
    cfg.kv_role = kv_role
    cfg.kv_connector_module_path = kv_connector_module_path
    cfg.kv_buffer_device = kv_buffer_device
    cfg.kv_buffer_size = kv_buffer_size
    cfg.kv_ip = kv_ip
    cfg.kv_port = kv_port
    cfg.kv_parallel_size = kv_parallel_size
    cfg.kv_rank = kv_rank
    cfg.engine_id = engine_id
    cfg.kv_connector_extra_config = kv_connector_extra_config or {}
    return cfg


def _make_test_app(
    vllm_config: MagicMock,
    served_model_name: list[str] | None = None,
) -> FastAPI:
    """FastAPI app with the plugin's route + state wired through both phases."""
    app = FastAPI()
    plugin = ServerConfigPlugin()
    plugin.attach_router(app)

    app.state.vllm_config = vllm_config
    args = Namespace(served_model_name=served_model_name)
    asyncio.run(plugin.init_state(None, app.state, args))
    return app


# ---------------------------------------------------------------------------
# _build_response unit tests
# ---------------------------------------------------------------------------


class TestBuildResponseModelSection:
    def test_model_name_is_first_served_name(self):
        cfg = _make_vllm_config(model_name="base-model")
        resp = _build_response(cfg, ["alias-a", "alias-b"])
        assert resp.model.name == "alias-a"

    def test_model_served_names_matches_input(self):
        cfg = _make_vllm_config()
        resp = _build_response(cfg, ["m1", "m2", "m3"])
        assert resp.model.served_names == ["m1", "m2", "m3"]

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (BFLOAT16, "bfloat16"),
            (FLOAT16, "float16"),
            (FLOAT32, "float32"),
        ],
    )
    def test_model_dtype_format(self, dtype, expected):
        cfg = _make_vllm_config(model_dtype=dtype)
        resp = _build_response(cfg, ["m"])
        assert resp.model.dtype == expected

    def test_model_quantization_none(self):
        cfg = _make_vllm_config(quantization=None)
        resp = _build_response(cfg, ["m"])
        assert resp.model.quantization is None

    @pytest.mark.parametrize("quant", ["awq", "gptq", "fp8"])
    def test_model_quantization_string(self, quant):
        cfg = _make_vllm_config(quantization=quant)
        resp = _build_response(cfg, ["m"])
        assert resp.model.quantization == quant

    def test_model_max_model_len(self):
        cfg = _make_vllm_config(max_model_len=131072)
        resp = _build_response(cfg, ["m"])
        assert resp.model.max_model_len == 131072


class TestBuildResponseKVCacheSection:
    def test_gpu_memory_utilization(self):
        cfg = _make_vllm_config(gpu_memory_utilization=0.85)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.gpu_memory_utilization == pytest.approx(0.85)

    @pytest.mark.parametrize(
        "model_dtype,expected",
        [
            (BFLOAT16, "bfloat16"),
            (FLOAT16, "float16"),
        ],
    )
    def test_cache_dtype_auto_resolves_to_model_dtype(self, model_dtype, expected):
        cfg = _make_vllm_config(cache_dtype="auto", model_dtype=model_dtype)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.dtype == expected

    def test_cache_dtype_explicit_passes_through(self):
        cfg = _make_vllm_config(cache_dtype="fp8", model_dtype=BFLOAT16)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.dtype == "fp8"

    def test_enable_prefix_caching_true(self):
        cfg = _make_vllm_config(enable_prefix_caching=True)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.enable_prefix_caching is True

    def test_enable_prefix_caching_false(self):
        cfg = _make_vllm_config(enable_prefix_caching=False)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_cache.enable_prefix_caching is False


class TestBuildResponseSchedulerSection:
    def test_max_num_seqs(self):
        cfg = _make_vllm_config(max_num_seqs=256)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.max_num_seqs == 256

    def test_max_num_batched_tokens(self):
        cfg = _make_vllm_config(max_num_batched_tokens=4096)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.max_num_batched_tokens == 4096

    def test_enable_chunked_prefill_true(self):
        cfg = _make_vllm_config(enable_chunked_prefill=True)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.enable_chunked_prefill is True

    def test_enable_chunked_prefill_false(self):
        cfg = _make_vllm_config(enable_chunked_prefill=False)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.enable_chunked_prefill is False

    @pytest.mark.parametrize("policy", ["fcfs", "priority"])
    def test_scheduler_policy(self, policy):
        cfg = _make_vllm_config(policy=policy)
        resp = _build_response(cfg, ["m"])
        assert resp.scheduler.policy == policy


class TestBuildResponseParallelismSection:
    def test_defaults_single_device(self):
        cfg = _make_vllm_config(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
        )
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.tensor_parallel_size == 1
        assert resp.parallelism.pipeline_parallel_size == 1
        assert resp.parallelism.data_parallel_size == 1
        assert resp.parallelism.data_parallel_rank == 0

    def test_tensor_parallel(self):
        cfg = _make_vllm_config(tensor_parallel_size=4)
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.tensor_parallel_size == 4

    def test_data_parallel_with_rank(self):
        cfg = _make_vllm_config(data_parallel_size=4, data_parallel_rank=3)
        resp = _build_response(cfg, ["m"])
        assert resp.parallelism.data_parallel_size == 4
        assert resp.parallelism.data_parallel_rank == 3


class TestBuildResponseFeaturesSection:
    def test_all_features_disabled_by_default(self):
        cfg = _make_vllm_config(
            speculative_config=None,
            lora_config=None,
            disable_hybrid_kv_cache_manager=False,
        )
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is False
        assert resp.features.lora is False
        assert resp.features.hma is True

    def test_speculative_decoding_enabled(self):
        cfg = _make_vllm_config(speculative_config=MagicMock())
        resp = _build_response(cfg, ["m"])
        assert resp.features.speculative_decoding is True

    def test_lora_enabled(self):
        cfg = _make_vllm_config(lora_config=MagicMock())
        resp = _build_response(cfg, ["m"])
        assert resp.features.lora is True

    def test_hma_disabled_when_manager_disabled(self):
        cfg = _make_vllm_config(disable_hybrid_kv_cache_manager=True)
        resp = _build_response(cfg, ["m"])
        assert resp.features.hma is False

    def test_hma_defaults_true_when_field_absent(self):
        # Older vLLM without `disable_hybrid_kv_cache_manager` on
        # SchedulerConfig: getattr(..., default=False) -> hma stays True.
        cfg = _make_vllm_config()
        del cfg.scheduler_config.disable_hybrid_kv_cache_manager
        resp = _build_response(cfg, ["m"])
        assert resp.features.hma is True


class TestBuildResponseKVTransferSection:
    def test_none_when_no_disaggregation_configured(self):
        cfg = _make_vllm_config(kv_transfer_config=None)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_transfer is None

    def test_nixl_connector_populates_fields_and_env_derived_port(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setattr(envs, "VLLM_NIXL_SIDE_CHANNEL_HOST", "10.0.0.5")
        monkeypatch.setattr(envs, "VLLM_NIXL_SIDE_CHANNEL_PORT", 5600)

        kv_transfer_cfg = _make_kv_transfer_config(
            kv_connector="NixlConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"backend": "UCX"},
        )
        cfg = _make_vllm_config(kv_transfer_config=kv_transfer_cfg)
        resp = _build_response(cfg, ["m"])

        assert resp.kv_transfer is not None
        assert resp.kv_transfer.kv_connector == "NixlConnector"
        assert resp.kv_transfer.kv_role == "kv_producer"
        assert resp.kv_transfer.extra_config == {"backend": REDACTED}
        assert resp.kv_transfer.nixl_side_channel_host == "10.0.0.5"
        assert resp.kv_transfer.nixl_side_channel_port == 5600

    def test_extra_config_values_redacted_by_default(self, monkeypatch):
        monkeypatch.delenv(EXPOSE_EXTRA_CONFIG_ENV, raising=False)
        kv_transfer_cfg = _make_kv_transfer_config(
            kv_connector_extra_config={"api_key": "s3cret", "backend": "UCX"},
        )
        cfg = _make_vllm_config(kv_transfer_config=kv_transfer_cfg)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_transfer.extra_config == {"api_key": REDACTED, "backend": REDACTED}

    def test_extra_config_values_exposed_when_opted_in(self, monkeypatch):
        monkeypatch.setenv(EXPOSE_EXTRA_CONFIG_ENV, "1")
        kv_transfer_cfg = _make_kv_transfer_config(
            kv_connector_extra_config={"api_key": "s3cret", "backend": "UCX"},
        )
        cfg = _make_vllm_config(kv_transfer_config=kv_transfer_cfg)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_transfer.extra_config == {"api_key": "s3cret", "backend": "UCX"}

    def test_empty_extra_config_stays_empty(self, monkeypatch):
        monkeypatch.delenv(EXPOSE_EXTRA_CONFIG_ENV, raising=False)
        cfg = _make_vllm_config(kv_transfer_config=_make_kv_transfer_config())
        resp = _build_response(cfg, ["m"])
        assert resp.kv_transfer.extra_config == {}

    def test_non_nixl_connector_leaves_nixl_fields_none(self):
        kv_transfer_cfg = _make_kv_transfer_config(kv_connector="LMCacheConnectorV1")
        cfg = _make_vllm_config(kv_transfer_config=kv_transfer_cfg)
        resp = _build_response(cfg, ["m"])
        assert resp.kv_transfer is not None
        assert resp.kv_transfer.kv_connector == "LMCacheConnectorV1"
        assert resp.kv_transfer.nixl_side_channel_host is None
        assert resp.kv_transfer.nixl_side_channel_port is None


class TestBuildResponseReturnType:
    def test_returns_server_config_response(self):
        cfg = _make_vllm_config()
        resp = _build_response(cfg, ["m"])
        assert isinstance(resp, ServerConfigResponse)
        assert isinstance(resp.model, ModelInfo)
        assert isinstance(resp.kv_cache, KVCacheInfo)
        assert isinstance(resp.scheduler, SchedulerInfo)
        assert isinstance(resp.parallelism, ParallelismInfo)
        assert isinstance(resp.features, FeaturesInfo)
        assert resp.kv_transfer is None

    def test_returns_kv_transfer_info_when_configured(self):
        cfg = _make_vllm_config(kv_transfer_config=_make_kv_transfer_config())
        resp = _build_response(cfg, ["m"])
        assert isinstance(resp.kv_transfer, KVTransferInfo)


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetServerConfigEndpoint:
    def test_response_parses_as_server_config_response(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["my-model"])
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        parsed = ServerConfigResponse.model_validate(resp.json())
        assert parsed.model.name == "my-model"

    def test_served_names_from_args_multiple(self):
        cfg = _make_vllm_config()
        app = _make_test_app(cfg, served_model_name=["alias-a", "alias-b"])
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        data = resp.json()
        assert data["model"]["name"] == "alias-a"
        assert data["model"]["served_names"] == ["alias-a", "alias-b"]

    def test_served_names_fallback_from_model_config(self):
        cfg = _make_vllm_config(served_model_name="model-from-config")
        app = _make_test_app(cfg, served_model_name=None)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        data = resp.json()
        assert data["model"]["name"] == "model-from-config"
        assert data["model"]["served_names"] == ["model-from-config"]

    def test_content_type_is_json(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        assert "application/json" in resp.headers["content-type"]

    def test_top_level_keys_present(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        assert set(resp.json().keys()) == {
            "model",
            "kv_cache",
            "scheduler",
            "parallelism",
            "features",
            "kv_transfer",
        }

    def test_wrong_method_returns_405(self):
        app = _make_test_app(_make_vllm_config(), served_model_name=["m"])
        with TestClient(app) as client:
            resp = client.post("/plugins/llm-d-server-introspection/config")
        assert resp.status_code == 405

    def test_missing_state_returns_500(self):
        # attach_router without ever calling init_state.
        # The handler must not crash on a missing attribute.
        app = FastAPI()
        ServerConfigPlugin().attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        assert resp.status_code == 500

    def test_response_cached_at_init_state_not_rebuilt_per_request(self):
        cfg = _make_vllm_config(max_model_len=32768)
        app = _make_test_app(cfg, served_model_name=["m"])
        cached = app.state.server_config_response
        with TestClient(app) as client:
            first = client.get("/plugins/llm-d-server-introspection/config")
            # A rebuild per request would pick up this change.
            cfg.model_config.max_model_len = 1024
            second = client.get("/plugins/llm-d-server-introspection/config")
        assert first.json()["model"]["max_model_len"] == 32768
        assert second.json()["model"]["max_model_len"] == 32768
        assert app.state.server_config_response is cached

    def test_engine_client_none_on_render_server_is_fine(self):
        # required_tasks=None -> eligible on the CPU only render server,
        # which calls init_state with engine_client=None. This plugin never
        # touches engine_client, so it must not raise.
        app = FastAPI()
        plugin = ServerConfigPlugin()
        plugin.attach_router(app)
        app.state.vllm_config = _make_vllm_config()
        args = Namespace(served_model_name=["m"])
        asyncio.run(plugin.init_state(None, app.state, args))
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/config")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


class TestPluginMetadata:
    def test_name(self):
        assert ServerConfigPlugin().name == "llm_d_server_introspection_config"

    def test_required_tasks_is_none(self):
        assert ServerConfigPlugin().required_tasks is None
