# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the `/plugins/llm-d-server-introspection/*` introspection endpoints.

Shared across all endpoint plugins in this package.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    served_names: list[str]
    dtype: str
    quantization: str | None
    max_model_len: int


class KVCacheInfo(BaseModel):
    gpu_memory_utilization: float
    dtype: str
    enable_prefix_caching: bool


class SchedulerInfo(BaseModel):
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool
    policy: str


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_rank: int


class FeaturesInfo(BaseModel):
    speculative_decoding: bool
    lora: bool
    hma: bool


class KVTransferInfo(BaseModel):
    kv_connector: str | None
    kv_role: str | None
    kv_connector_module_path: str | None = None
    kv_buffer_device: str | None = None
    kv_buffer_size: float | None = None
    kv_ip: str | None = None
    kv_port: int | None = None
    kv_parallel_size: int | None = None
    kv_rank: int | None = None
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
    major: int
    minor: int


class DeviceInfo(BaseModel):
    rank: int
    name: str
    total_memory_bytes: int
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
# doesn't declare.
# ---------------------------------------------------------------------------


class _KVCacheGroupBase(BaseModel):
    group_id: int
    layer_count: int
    layer_names: list[str]
    block_size: int
    page_size_bytes: int
    # Present (non-None) only when the group fans out a UniformTypeKVCacheSpecs
    # into its per layer specs.
    layer_specs: list[dict] | None = None


class _FullAttentionBaseSpec(_KVCacheGroupBase):
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


class FullAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["full_attention"] = "full_attention"


class MLAAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["mla_attention"] = "mla_attention"
    cache_dtype_str: str | None = None


class SlidingWindowGroupSpec(_KVCacheGroupBase):
    kind: Literal["sliding_window"] = "sliding_window"
    num_kv_heads: int
    head_size: int
    dtype: str
    sliding_window: int


class SlidingWindowMLAGroupSpec(_KVCacheGroupBase):
    kind: Literal["sliding_window_mla"] = "sliding_window_mla"
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int
    cache_dtype_str: str | None = None


class ChunkedLocalAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["chunked_local_attention"] = "chunked_local_attention"
    num_kv_heads: int
    head_size: int
    dtype: str
    attention_chunk_size: int


class MambaGroupSpec(_KVCacheGroupBase):
    kind: Literal["mamba"] = "mamba"
    shapes: list[list[int]]
    dtypes: list[str]
    mamba_type: str
    mamba_cache_mode: str


class CrossAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["cross_attention"] = "cross_attention"
    num_kv_heads: int
    head_size: int
    dtype: str


class EncoderOnlyAttentionGroupSpec(_KVCacheGroupBase):
    kind: Literal["encoder_only_attention"] = "encoder_only_attention"
    num_kv_heads: int
    head_size: int
    dtype: str


class SinkFullAttentionGroupSpec(_FullAttentionBaseSpec):
    kind: Literal["sink_full_attention"] = "sink_full_attention"
    sink_len: int | None = None


class UnknownGroupSpec(_KVCacheGroupBase):
    kind: Literal["unknown"] = "unknown"


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
