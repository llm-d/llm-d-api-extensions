# SPDX-License-Identifier: Apache-2.0
"""Pydantic model tests for the `/plugins/llm-d-server-introspection/config` response schema."""

import pytest
from kv_group_fixtures import ALL_KINDS, SPEC_CLS_BY_KIND, group_dict

from llm_d_api_extensions_vllm.server_introspection.schemas import (
    ComputeCapability,
    DeviceInfo,
    DevicesResponse,
    FeaturesInfo,
    FullAttentionGroupSpec,
    KVCacheInfo,
    KVCacheResponse,
    MambaGroupSpec,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
    SinkFullAttentionGroupSpec,
)


def _server_config_response() -> ServerConfigResponse:
    return ServerConfigResponse(
        model=ModelInfo(
            name="llama",
            served_names=["llama"],
            dtype="bfloat16",
            quantization=None,
            max_model_len=4096,
        ),
        kv_cache=KVCacheInfo(
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            enable_prefix_caching=True,
        ),
        scheduler=SchedulerInfo(
            max_num_seqs=64,
            max_num_batched_tokens=1024,
            enable_chunked_prefill=True,
            policy="fcfs",
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
        features=FeaturesInfo(
            speculative_decoding=False,
            lora=False,
            hma=True,
        ),
    )


def test_server_config_response_roundtrip():
    original = _server_config_response()
    restored = ServerConfigResponse.model_validate(original.model_dump())
    assert restored == original


def test_quantization_null_serialization():
    dumped = _server_config_response().model_dump()
    assert dumped["model"]["quantization"] is None


def test_quantization_string_serialization():
    resp = _server_config_response()
    resp.model.quantization = "gptq"
    assert resp.model_dump()["model"]["quantization"] == "gptq"


def test_features_info_all_bool():
    feat = FeaturesInfo(speculative_decoding=True, lora=False, hma=True)
    assert feat.model_dump() == {
        "speculative_decoding": True,
        "lora": False,
        "hma": True,
    }


def test_server_config_response_json_schema_has_required_sections():
    schema = ServerConfigResponse.model_json_schema()
    assert set(schema["properties"]) == {
        "model",
        "kv_cache",
        "scheduler",
        "parallelism",
        "features",
        "kv_transfer",
    }


def _device_info(**overrides) -> DeviceInfo:
    fields = dict(
        rank=0,
        name="A100-PCIE-40GB",
        total_memory_bytes=42_949_672_960,
        compute_capability=ComputeCapability(major=8, minor=0),
        num_compute_units=108,
    )
    fields.update(overrides)
    return DeviceInfo(**fields)


def test_devices_response_roundtrip():
    original = DevicesResponse(devices=[_device_info(), _device_info(rank=1)])
    restored = DevicesResponse.model_validate(original.model_dump())
    assert restored == original


def test_devices_response_empty():
    assert DevicesResponse(devices=[]).devices == []


def test_device_info_compute_capability_defaults_to_none():
    device = DeviceInfo(rank=0, name="cpu", total_memory_bytes=1024)
    assert device.compute_capability is None
    assert device.num_compute_units is None


def test_device_info_null_capability_serialization():
    dumped = _device_info(compute_capability=None).model_dump()
    assert dumped["compute_capability"] is None


def test_devices_response_json_schema_shape():
    schema = DevicesResponse.model_json_schema()
    assert set(schema["properties"]) == {"devices"}


# ---------------------------------------------------------------------------
# kv-cache group spec + response tests
# ---------------------------------------------------------------------------


def _spec(kind: str, **kwargs):
    return SPEC_CLS_BY_KIND[kind].model_validate(group_dict(kind, **kwargs))


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_group_spec_roundtrip(kind):
    spec = _spec(kind)
    restored = type(spec).model_validate(spec.model_dump())
    assert restored == spec
    assert spec.kind == kind


def test_full_attention_optional_fields_default_none():
    spec = _spec("full_attention", omit=("sliding_window", "attention_chunk_size"))
    assert spec.sliding_window is None
    assert spec.attention_chunk_size is None


def test_sink_full_attention_sink_len_optional():
    spec = _spec("sink_full_attention", omit=("sink_len",))
    assert isinstance(spec, SinkFullAttentionGroupSpec)
    assert spec.sink_len is None


def test_full_attention_layer_specs_populated_for_uniform_type_group():
    # A UniformTypeKVCacheSpecs group resolves to its inner kind but keeps
    # layer_specs populated (None for a regular, non fanned out group).
    spec = _spec("full_attention", layer_specs=[{"head_size": 128}, {"head_size": 64}])
    restored = FullAttentionGroupSpec.model_validate(spec.model_dump())
    assert restored == spec
    assert spec.layer_specs == [{"head_size": 128}, {"head_size": 64}]


def test_kv_cache_response_defaults():
    resp = KVCacheResponse()
    assert resp.kv_cache_size_tokens is None
    assert resp.max_concurrency is None
    assert resp.num_gpu_blocks is None
    assert resp.num_cpu_blocks is None
    assert resp.groups == []


def test_kv_cache_response_roundtrip_with_groups():
    original = KVCacheResponse(
        kv_cache_size_tokens=16384,
        max_concurrency=0.5,
        num_gpu_blocks=1024,
        num_cpu_blocks=256,
        groups=[_spec("full_attention")],
    )
    restored = KVCacheResponse.model_validate(original.model_dump())
    assert restored == original


def test_kv_cache_response_json_schema_has_expected_properties():
    schema = KVCacheResponse.model_json_schema()
    assert set(schema["properties"]) == {
        "kv_cache_size_tokens",
        "max_concurrency",
        "num_gpu_blocks",
        "num_cpu_blocks",
        "groups",
    }


def test_kv_cache_response_discriminates_group_spec_types_on_validate():
    # A discriminated union must reconstruct the correct concrete class from
    # raw JSON, not just accept already typed model instances.
    dumped = KVCacheResponse(
        groups=[_spec("full_attention"), _spec("mamba", group_id=1)]
    ).model_dump()
    restored = KVCacheResponse.model_validate(dumped)
    assert isinstance(restored.groups[0], FullAttentionGroupSpec)
    assert isinstance(restored.groups[1], MambaGroupSpec)
