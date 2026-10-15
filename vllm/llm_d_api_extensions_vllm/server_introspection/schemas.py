# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the `/plugins/llm-d-server-introspection/*` introspection endpoints.

Shared across all endpoint plugins in this package.

Every field sourced from vLLM is nullable. vLLM's config objects and engine
metadata are internal API, so a field vLLM renames, removes or retypes is
reported as `null` rather than failing the response (see `common.py`).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str | None = None
    served_names: list[str] | None = None
    dtype: str | None = None
    quantization: str | None = None
    max_model_len: int | None = None


class KVCacheInfo(BaseModel):
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    enable_prefix_caching: bool | None = None


class SchedulerInfo(BaseModel):
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    policy: str | None = None


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    data_parallel_rank: int | None = None


class FeaturesInfo(BaseModel):
    speculative_decoding: bool | None = None
    lora: bool | None = None
    hma: bool | None = None


class KVTransferInfo(BaseModel):
    kv_connector: str | None = None
    kv_role: str | None = None
    kv_connector_module_path: str | None = None
    kv_buffer_device: str | None = None
    kv_port: int | None = None
    engine_id: str | None = None
    extra_config: dict = Field(default_factory=dict)
    # NIXL side channel base endpoint (env derived only when connector is
    # NixlConnector). This is the base host/port. NixlConnector derives the
    # actual per rank port as base_port + rank_offset inside the worker.
    nixl_side_channel_host: str | None = None
    nixl_side_channel_port: int | None = None


class ServerConfigResponse(BaseModel):
    model: ModelInfo
    kv_cache: KVCacheInfo
    scheduler: SchedulerInfo
    parallelism: ParallelismInfo
    features: FeaturesInfo
    kv_transfer: KVTransferInfo | None = None


class ComputeCapability(BaseModel):
    major: int | None = None
    minor: int | None = None


class DeviceInfo(BaseModel):
    rank: int | None = None
    name: str | None = None
    total_memory_bytes: int | None = None
    compute_capability: ComputeCapability | None = None
    num_compute_units: int | None = None


class DevicesResponse(BaseModel):
    devices: list[DeviceInfo]


# ---------------------------------------------------------------------------
# `/plugins/llm-d-server-introspection/kv-cache`: group specs are a
# discriminated union keyed on `kind`, matching the field names and
# `KVCacheSpecKind` string values `EngineClient.get_kv_cache_group_metadata()`
# serializes groups under (vllm-project/vllm#48121). Every group dict carries
# the full field superset (irrelevant fields sent as `None`). Pydantic's
# default `extra="ignore"` behavior drops whichever ones a given `kind`
# doesn't declare. A `kind` this package does not know is reported as
# `UnknownGroupSpec` with the original string in `vllm_kind`.
# ---------------------------------------------------------------------------


class _KVCacheGroupBase(BaseModel):
    group_idx: int | None = None
    layer_count: int | None = None
    layer_names: list[str] | None = None
    block_size: int | None = None
    page_size_bytes: int | None = None
    # Present (non-None) only when the group fans out a UniformTypeKVCacheSpecs
    # into its per layer specs.
    layer_specs: list[dict] | None = None


class _FullAttentionBaseSpec(_KVCacheGroupBase):
    num_kv_heads: int | None = None
    head_size: int | None = None
    head_size_v: int | None = None
    dtype: str | None = None
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


class FullAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["full_attention"] = "full_attention"


class MLAAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["mla_attention"] = "mla_attention"
    cache_dtype_str: str | None = None


class SlidingWindowGroupSpec(_KVCacheGroupBase):
    kind: Literal["sliding_window"] = "sliding_window"
    num_kv_heads: int | None = None
    head_size: int | None = None
    dtype: str | None = None
    sliding_window: int | None = None


class SlidingWindowMLAGroupSpec(_KVCacheGroupBase):
    kind: Literal["sliding_window_mla"] = "sliding_window_mla"
    num_kv_heads: int | None = None
    head_size: int | None = None
    head_size_v: int | None = None
    dtype: str | None = None
    sliding_window: int | None = None
    cache_dtype_str: str | None = None


class ChunkedLocalAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["chunked_local_attention"] = "chunked_local_attention"
    num_kv_heads: int | None = None
    head_size: int | None = None
    dtype: str | None = None
    attention_chunk_size: int | None = None


class MambaGroupSpec(_KVCacheGroupBase):
    kind: Literal["mamba"] = "mamba"
    shapes: list[list[int]] | None = None
    dtypes: list[str] | None = None
    mamba_type: str | None = None
    mamba_cache_mode: str | None = None


class CrossAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["cross_attention"] = "cross_attention"
    num_kv_heads: int | None = None
    head_size: int | None = None
    dtype: str | None = None


class EncoderOnlyAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["encoder_only_attention"] = "encoder_only_attention"
    num_kv_heads: int | None = None
    head_size: int | None = None
    dtype: str | None = None


class SinkFullAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["sink_full_attention"] = "sink_full_attention"
    sink_len: int | None = None


class UnknownGroupSpec(_KVCacheGroupBase):
    kind: Literal["unknown"] = "unknown"
    # The `kind` vLLM reported when this package has no schema for it.
    vllm_kind: str | None = None


KVCacheGroupSpec = Annotated[
    FullAttentionGroupSpec
    | MLAAttentionGroupSpec
    | SlidingWindowGroupSpec
    | SlidingWindowMLAGroupSpec
    | ChunkedLocalAttentionGroupSpec
    | MambaGroupSpec
    | CrossAttentionGroupSpec
    | EncoderOnlyAttentionGroupSpec
    | SinkFullAttentionGroupSpec
    | UnknownGroupSpec,
    Field(discriminator="kind"),
]


class KVCacheResponse(BaseModel):
    kv_cache_size_tokens: int | None = None
    max_concurrency: float | None = None
    num_gpu_blocks: int | None = None
    num_cpu_blocks: int | None = None
    groups: list[KVCacheGroupSpec] = Field(default_factory=list)
