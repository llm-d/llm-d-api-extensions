# SPDX-License-Identifier: Apache-2.0
"""Serialized KV cache group dicts shared by the kv-cache plugin and schema tests.

Mirrors what `get_kv_cache_group_metadata` produces per vllm-project/vllm#48121.
"""

from llm_d_api_extensions_vllm.server_introspection.schemas import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    EncoderOnlyAttentionGroupSpec,
    FullAttentionGroupSpec,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    SlidingWindowMLAGroupSpec,
    UnknownGroupSpec,
)

BASE_GROUP = {
    "group_id": 0,
    "layer_count": 2,
    "layer_names": ["model.layers.0.self_attn", "model.layers.1.self_attn"],
    "block_size": 16,
    "page_size_bytes": 131072,
    "layer_specs": None,
}

_ATTENTION = {"num_kv_heads": 8, "head_size": 128, "dtype": "bfloat16"}

# Fields each kind adds on top of BASE_GROUP.
_KIND_FIELDS: dict[str, dict] = {
    "full_attention": {
        **_ATTENTION,
        "head_size_v": 128,
        "sliding_window": None,
        "attention_chunk_size": None,
    },
    "mla_attention": {
        **_ATTENTION,
        "head_size_v": 64,
        "sliding_window": None,
        "attention_chunk_size": None,
        "cache_dtype_str": "float8_e4m3fn",
    },
    "sliding_window": {**_ATTENTION, "dtype": "float16", "sliding_window": 4096},
    "sliding_window_mla": {
        **_ATTENTION,
        "head_size_v": 64,
        "sliding_window": 4096,
        "cache_dtype_str": "float8_e4m3fn",
    },
    "chunked_local_attention": {**_ATTENTION, "dtype": "float16", "attention_chunk_size": 2048},
    "mamba": {
        "block_size": 1,
        "page_size_bytes": 4096,
        "shapes": [[16, 128], [16, 64]],
        "dtypes": ["float32", "float32"],
        "mamba_type": "mamba2",
        "mamba_cache_mode": "none",
    },
    "cross_attention": dict(_ATTENTION),
    "encoder_only_attention": dict(_ATTENTION),
    "sink_full_attention": {
        **_ATTENTION,
        "head_size_v": 128,
        "sliding_window": 2048,
        "attention_chunk_size": None,
        "sink_len": 4,
    },
    "unknown": {},
}

SPEC_CLS_BY_KIND: dict[str, type] = {
    "full_attention": FullAttentionGroupSpec,
    "mla_attention": MLAAttentionGroupSpec,
    "sliding_window": SlidingWindowGroupSpec,
    "sliding_window_mla": SlidingWindowMLAGroupSpec,
    "chunked_local_attention": ChunkedLocalAttentionGroupSpec,
    "mamba": MambaGroupSpec,
    "cross_attention": CrossAttentionGroupSpec,
    "encoder_only_attention": EncoderOnlyAttentionGroupSpec,
    "sink_full_attention": SinkFullAttentionGroupSpec,
    "unknown": UnknownGroupSpec,
}

ALL_KINDS = list(SPEC_CLS_BY_KIND)


def group_dict(kind: str, *, omit: tuple[str, ...] = (), **overrides) -> dict:
    group = {**BASE_GROUP, "kind": kind, **_KIND_FIELDS[kind], **overrides}
    for key in omit:
        group.pop(key)
    return group
