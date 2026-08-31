# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `GET /plugins/llm-d-server-introspection/kv-cache`.

Fast, no engine and no real vLLM/model. `_FakeEngineClient` optionally
exposes `get_kv_cache_group_metadata` so both the "new method present" and
"older vLLM falls back to capacity only" `init_state` paths get exercised.
"""

import asyncio
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_d_api_extensions_vllm.server_introspection.kv_cache_plugin import (
    ServerKVCachePlugin,
    _build_group_spec,
    _build_response,
    _capacity_from_vllm_config,
)
from llm_d_api_extensions_vllm.server_introspection.schemas import (
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

# ---------------------------------------------------------------------------
# Helpers — serialized group dicts (mirrors what
# `get_kv_cache_group_metadata` produces per vllm-project/vllm#48121)
# ---------------------------------------------------------------------------

_BASE_GROUP = {
    "group_id": 0,
    "layer_count": 2,
    "layer_names": ["model.layers.0.self_attn", "model.layers.1.self_attn"],
    "block_size": 16,
    "page_size_bytes": 131072,
    "layer_specs": None,
}


def _full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "full_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        **overrides,
    }


def _mla_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "mla_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 64,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        "cache_dtype_str": "float8_e4m3fn",
        **overrides,
    }


def _sliding_window_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "sliding_window",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "sliding_window": 4096,
        **overrides,
    }


def _sliding_window_mla_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "sliding_window_mla",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 64,
        "dtype": "bfloat16",
        "sliding_window": 4096,
        "cache_dtype_str": "float8_e4m3fn",
        **overrides,
    }


def _chunked_local_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "chunked_local_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "attention_chunk_size": 2048,
        **overrides,
    }


def _mamba_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "mamba",
        "block_size": 1,
        "page_size_bytes": 4096,
        "shapes": [[16, 128], [16, 64]],
        "dtypes": ["float32", "float32"],
        "mamba_type": "mamba2",
        "mamba_cache_mode": "none",
        **overrides,
    }


def _cross_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "cross_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "bfloat16",
        **overrides,
    }


def _encoder_only_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "encoder_only_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "bfloat16",
        **overrides,
    }


def _sink_full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "sink_full_attention",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": 2048,
        "attention_chunk_size": None,
        "sink_len": 4,
        **overrides,
    }


def _unknown_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "kind": "unknown",
        **overrides,
    }


def _uniform_type_dict(**overrides) -> dict:
    # A UniformTypeKVCacheSpecs group resolves to its inner kind (here
    # full_attention) and carries a populated `layer_specs` list.
    return _full_attention_dict(
        layer_specs=[
            {"head_size": 128, "dtype": "bfloat16"},
            {"head_size": 64, "dtype": "bfloat16"},
        ],
        **overrides,
    )


def _capacity_data(**overrides) -> dict:
    return {
        "kv_cache_size_tokens": 16384,
        "max_concurrency": 0.5,
        "num_gpu_blocks": 1024,
        "num_cpu_blocks": 256,
        "groups": [],
        **overrides,
    }


def _kv_cache_data(**overrides) -> dict:
    overrides.setdefault("groups", [_full_attention_dict()])
    return _capacity_data(**overrides)


class _FakeEngineClient:
    """Minimal stand in exercising `get_kv_cache_group_metadata`. Not a real engine.

    `get_kv_cache_group_metadata` is only defined as an attribute when
    `has_new_method=True`, so `hasattr` feature detection in the plugin can
    be exercised both ways.
    """

    def __init__(self, groups: Any = None, has_new_method: bool = True):
        self._groups = groups if groups is not None else []
        self.calls = 0
        if has_new_method:
            self.get_kv_cache_group_metadata = self._get_kv_cache_group_metadata

    async def _get_kv_cache_group_metadata(self) -> Any:
        self.calls += 1
        return self._groups


def _make_vllm_config(
    *,
    num_gpu_blocks: int | None = 1024,
    block_size: int | None = 16,
    num_cpu_blocks: int | None = 0,
    max_model_len: int = 4096,
) -> MagicMock:
    cache_cfg = MagicMock()
    cache_cfg.num_gpu_blocks = num_gpu_blocks
    cache_cfg.block_size = block_size
    cache_cfg.num_cpu_blocks = num_cpu_blocks

    model_cfg = MagicMock()
    model_cfg.max_model_len = max_model_len

    vllm_config = MagicMock()
    vllm_config.cache_config = cache_cfg
    vllm_config.model_config = model_cfg
    return vllm_config


def _make_test_app(
    engine_client: "_FakeEngineClient | None", vllm_config: MagicMock | None = None
) -> FastAPI:
    """FastAPI app with the plugin's route + state wired through both phases."""
    app = FastAPI()
    plugin = ServerKVCachePlugin()
    plugin.attach_router(app)
    app.state.vllm_config = vllm_config if vllm_config is not None else _make_vllm_config()
    asyncio.run(plugin.init_state(engine_client, app.state, Namespace()))
    return app


# ---------------------------------------------------------------------------
# _build_group_spec unit tests — discriminator dispatch
# ---------------------------------------------------------------------------


class TestBuildGroupSpecDispatch:
    def test_full_attention_spec(self):
        result = _build_group_spec(_full_attention_dict())
        assert isinstance(result, FullAttentionGroupSpec)
        assert result.kind == "full_attention"

    def test_mla_attention_spec(self):
        result = _build_group_spec(_mla_attention_dict())
        assert isinstance(result, MLAAttentionGroupSpec)

    def test_sliding_window_spec(self):
        result = _build_group_spec(_sliding_window_dict())
        assert isinstance(result, SlidingWindowGroupSpec)

    def test_sliding_window_mla_spec(self):
        result = _build_group_spec(_sliding_window_mla_dict())
        assert isinstance(result, SlidingWindowMLAGroupSpec)

    def test_chunked_local_attention_spec(self):
        result = _build_group_spec(_chunked_local_attention_dict())
        assert isinstance(result, ChunkedLocalAttentionGroupSpec)

    def test_mamba_spec(self):
        result = _build_group_spec(_mamba_dict())
        assert isinstance(result, MambaGroupSpec)

    def test_cross_attention_spec(self):
        result = _build_group_spec(_cross_attention_dict())
        assert isinstance(result, CrossAttentionGroupSpec)

    def test_encoder_only_attention_spec(self):
        result = _build_group_spec(_encoder_only_attention_dict())
        assert isinstance(result, EncoderOnlyAttentionGroupSpec)

    def test_sink_full_attention_spec(self):
        result = _build_group_spec(_sink_full_attention_dict())
        assert isinstance(result, SinkFullAttentionGroupSpec)

    def test_unknown_spec(self):
        result = _build_group_spec(_unknown_dict())
        assert isinstance(result, UnknownGroupSpec)

    def test_uniform_type_spec_resolves_to_inner_kind(self):
        result = _build_group_spec(_uniform_type_dict())
        assert isinstance(result, FullAttentionGroupSpec)
        assert result.layer_specs == [
            {"head_size": 128, "dtype": "bfloat16"},
            {"head_size": 64, "dtype": "bfloat16"},
        ]

    def test_unhandled_kind_raises_value_error(self):
        group = {**_BASE_GROUP, "kind": "made_up_kind"}
        with pytest.raises(ValueError, match="Unhandled KVCacheSpec kind"):
            _build_group_spec(group)

    def test_preserves_base_fields(self):
        result = _build_group_spec(_full_attention_dict(group_id=3))
        assert result.group_id == 3
        assert result.layer_count == 2
        assert result.layer_names == _BASE_GROUP["layer_names"]
        assert result.block_size == 16
        assert result.page_size_bytes == 131072

    def test_ignores_irrelevant_always_present_fields(self):
        # Real payloads send every field for every kind (irrelevant ones as
        # None) - a full_attention group also carries mamba/sink fields.
        group = _full_attention_dict(shapes=None, dtypes=None, mamba_type=None, sink_len=None)
        result = _build_group_spec(group)
        assert isinstance(result, FullAttentionGroupSpec)


# ---------------------------------------------------------------------------
# _build_response unit tests — pure function
# ---------------------------------------------------------------------------


class TestBuildResponseNullState:
    def test_none_state(self):
        resp = _build_response(None)
        assert isinstance(resp, KVCacheResponse)
        assert resp.kv_cache_size_tokens is None
        assert resp.max_concurrency is None
        assert resp.num_gpu_blocks is None
        assert resp.num_cpu_blocks is None
        assert resp.groups == []


class TestBuildResponseCapacityFields:
    def test_kv_cache_size_tokens(self):
        resp = _build_response(_kv_cache_data(kv_cache_size_tokens=32768))
        assert resp.kv_cache_size_tokens == 32768

    def test_max_concurrency(self):
        resp = _build_response(_kv_cache_data(max_concurrency=1.25))
        assert resp.max_concurrency == pytest.approx(1.25)

    def test_num_gpu_blocks(self):
        resp = _build_response(_kv_cache_data(num_gpu_blocks=2048))
        assert resp.num_gpu_blocks == 2048

    def test_num_cpu_blocks_zero(self):
        resp = _build_response(_kv_cache_data(num_cpu_blocks=0))
        assert resp.num_cpu_blocks == 0

    def test_missing_capacity_fields_resolve_to_none(self):
        resp = _build_response({"groups": []})
        assert resp.kv_cache_size_tokens is None
        assert resp.num_gpu_blocks is None


class TestBuildResponseGroups:
    def test_hybrid_model_two_groups(self):
        groups = [_full_attention_dict(group_id=0), _mamba_dict(group_id=1)]
        resp = _build_response(_kv_cache_data(groups=groups))
        assert len(resp.groups) == 2
        assert isinstance(resp.groups[0], FullAttentionGroupSpec)
        assert isinstance(resp.groups[1], MambaGroupSpec)

    def test_empty_groups_list(self):
        resp = _build_response(_kv_cache_data(groups=[]))
        assert resp.groups == []


# ---------------------------------------------------------------------------
# _capacity_from_vllm_config unit tests
# ---------------------------------------------------------------------------


class TestCapacityFromVllmConfig:
    def test_computes_kv_cache_size_tokens(self):
        cfg = _make_vllm_config(num_gpu_blocks=1024, block_size=16)
        data = _capacity_from_vllm_config(cfg)
        assert data["kv_cache_size_tokens"] == 16384
        assert data["num_gpu_blocks"] == 1024
        assert data["groups"] == []

    def test_computes_max_concurrency(self):
        cfg = _make_vllm_config(num_gpu_blocks=1024, block_size=16, max_model_len=4096)
        data = _capacity_from_vllm_config(cfg)
        assert data["max_concurrency"] == pytest.approx(16384 / 4096)

    def test_none_num_gpu_blocks_yields_none_fields(self):
        cfg = _make_vllm_config(num_gpu_blocks=None)
        data = _capacity_from_vllm_config(cfg)
        assert data["kv_cache_size_tokens"] is None
        assert data["max_concurrency"] is None
        assert data["num_gpu_blocks"] is None

    def test_num_cpu_blocks_passthrough(self):
        cfg = _make_vllm_config(num_cpu_blocks=512)
        data = _capacity_from_vllm_config(cfg)
        assert data["num_cpu_blocks"] == 512


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetKVCacheEndpoint:
    def test_returns_200_with_data_from_new_method(self):
        fake_client = _FakeEngineClient([_full_attention_dict()], has_new_method=True)
        vllm_config = _make_vllm_config(num_gpu_blocks=1024, block_size=16)
        app = _make_test_app(fake_client, vllm_config=vllm_config)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_gpu_blocks"] == 1024
        assert len(data["groups"]) == 1
        assert fake_client.calls == 1

    def test_calls_get_kv_cache_group_metadata_once_and_caches(self):
        fake_client = _FakeEngineClient([_full_attention_dict()], has_new_method=True)
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            client.get("/plugins/llm-d-server-introspection/kv-cache")
            client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert fake_client.calls == 1

    def test_degrades_to_capacity_only_without_new_method(self):
        fake_client = _FakeEngineClient(has_new_method=False)
        vllm_config = _make_vllm_config(num_gpu_blocks=2048, block_size=16)
        app = _make_test_app(fake_client, vllm_config=vllm_config)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_gpu_blocks"] == 2048
        assert data["kv_cache_size_tokens"] == 32768
        assert data["groups"] == []

    def test_engine_client_none_returns_503(self):
        app = _make_test_app(None)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert resp.status_code == 503

    def test_missing_state_returns_500(self):
        app = FastAPI()
        ServerKVCachePlugin().attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert resp.status_code == 500

    def test_content_type_is_json(self):
        fake_client = _FakeEngineClient([_full_attention_dict()])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert "application/json" in resp.headers["content-type"]

    def test_wrong_method_returns_405(self):
        app = _make_test_app(_FakeEngineClient([_full_attention_dict()]))
        with TestClient(app) as client:
            resp = client.post("/plugins/llm-d-server-introspection/kv-cache")
        assert resp.status_code == 405

    def test_groups_serialized_correctly(self):
        groups = [_full_attention_dict(group_id=0), _mamba_dict(group_id=1)]
        fake_client = _FakeEngineClient(groups)
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        data = resp.json()["groups"]
        assert data[0]["kind"] == "full_attention"
        assert data[1]["kind"] == "mamba"

    def test_top_level_keys_present(self):
        fake_client = _FakeEngineClient([_full_attention_dict()])
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/llm-d-server-introspection/kv-cache")
        assert set(resp.json().keys()) == {
            "kv_cache_size_tokens",
            "max_concurrency",
            "num_gpu_blocks",
            "num_cpu_blocks",
            "groups",
        }


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


class TestPluginMetadata:
    def test_name(self):
        assert ServerKVCachePlugin().name == "llm_d_server_introspection_kv_cache"

    def test_required_tasks_excludes_render(self):
        assert "render" not in ServerKVCachePlugin().required_tasks

    def test_required_tasks_includes_generate(self):
        assert "generate" in ServerKVCachePlugin().required_tasks
