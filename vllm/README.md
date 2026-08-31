# llm-d API extensions for vLLM

`llm-d-api-extensions-vllm` packages llm-d's vLLM extensions. They integrate through vLLM's [endpoint plugin framework](https://docs.vllm.ai/en/latest/design/endpoint_plugins/) and `--worker-extension-cls`, so nothing here requires a change to vLLM core.

## Server introspection

Three read only endpoints describing the server llm-d is talking to. Each is a separate plugin so a deployment can enable only what it needs.

### `GET /plugins/llm-d-server-introspection/config`

Operator supplied, config time values of how the server was launched. Nothing profiled or derived from model internals. Requires no engine, so it also works on the CPU only render server.

```jsonc
{
  "model": {
    "name": "llama3",
    "served_names": ["llama3"],
    "dtype": "bfloat16",
    "quantization": null,
    "max_model_len": 8192
  },
  "kv_cache": {
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "enable_prefix_caching": true
  },
  "scheduler": {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": true,
    "policy": "fcfs"
  },
  "parallelism": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "data_parallel_rank": 0
  },
  "features": {
    "speculative_decoding": false,
    "lora": true,
    "hma": true
  },
  "kv_transfer": {
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": null,
    "kv_buffer_device": "cuda",
    "kv_buffer_size": 1000000000.0,
    "kv_ip": "127.0.0.1",
    "kv_port": 14579,
    "kv_parallel_size": 1,
    "kv_rank": null,
    "engine_id": "engine-0",
    "extra_config": {},
    "nixl_side_channel_host": "localhost",
    "nixl_side_channel_port": 5600
  }
}
```

> [!NOTE]
> `kv_transfer` is `null` when no disaggregation/KV offloading connector is configured (`vllm_config.kv_transfer_config` is `None`). `nixl_side_channel_host`/`nixl_side_channel_port` are only populated when `kv_connector` is `"NixlConnector"`. They report the env derived *base* host/port (`VLLM_NIXL_SIDE_CHANNEL_HOST`/`VLLM_NIXL_SIDE_CHANNEL_PORT`). `NixlConnector` derives the actual per rank bound port as `base_port + rank_offset` inside the worker.

> [!WARNING]
> `kv_connector_extra_config` is free form and operator controlled. It could theoretically contain sensitive data.

### `GET /plugins/llm-d-server-introspection/devices`

Per rank hardware properties, gathered once at startup via `collective_rpc` and cached for the server's lifetime. Requires an engine (`503` on the CPU only render server) and the worker side `get_device_properties` method installed by `device_worker_ext.DeviceInfoWorkerExtension` via `--worker-extension-cls`.

```jsonc
{
  "devices": [
    {
      "rank": 0,
      "name": "A100-PCIE-40GB",
      "total_memory_bytes": 42949672960,
      "compute_capability": { "major": 8, "minor": 0 },
      "num_compute_units": 108
    }
  ]
}
```

### `GET /plugins/llm-d-server-introspection/kv-cache`

Post profiling KV cache capacity and attention group structure, gathered once at startup and cached for the server's lifetime. Requires an engine (`503` on the CPU only render server). `groups` is a discriminated union keyed on `kind`, correctly representing hybrid models with multiple attention/mamba groups.

Against a vLLM build without `EngineClient.get_kv_cache_group_metadata()` ([vllm-project/vllm#48121](https://github.com/vllm-project/vllm/pull/48121)), this plugin falls back to capacity only fields read directly off `vllm_config.cache_config` and returns an empty `groups`.

```jsonc
{
  "kv_cache_size_tokens": 393216,
  "max_concurrency": 48.0,
  "num_gpu_blocks": 24576,
  "num_cpu_blocks": 0,
  "groups": [
    {
      "group_id": 0,
      "kind": "full_attention",
      "layer_count": 32,
      "layer_names": ["model.layers.0.self_attn", "..."],
      "block_size": 16,
      "page_size_bytes": 131072,
      "num_kv_heads": 8,
      "head_size": 128,
      "head_size_v": 128,
      "dtype": "bfloat16",
      "sliding_window": null,
      "attention_chunk_size": null
    }
  ]
}
```

## Install

```bash
pip install .

# or for development
pip install -e .
```

## Enable

Endpoint plugins load only when explicitly named in `VLLM_PLUGINS` (off by default), so installing this package alone changes nothing:

```bash
# config only
VLLM_PLUGINS=llm_d_server_introspection_config vllm serve <model>

# devices only
VLLM_PLUGINS=llm_d_server_introspection_devices vllm serve <model> \
  --worker-extension-cls llm_d_api_extensions_vllm.server_introspection.device_worker_ext.DeviceInfoWorkerExtension

# kv cache only
VLLM_PLUGINS=llm_d_server_introspection_kv_cache vllm serve <model>

# all three
VLLM_PLUGINS=llm_d_server_introspection_config,llm_d_server_introspection_devices,llm_d_server_introspection_kv_cache \
  vllm serve <model> \
  --worker-extension-cls llm_d_api_extensions_vllm.server_introspection.device_worker_ext.DeviceInfoWorkerExtension
```

```bash
curl http://localhost:8000/plugins/llm-d-server-introspection/config
curl http://localhost:8000/plugins/llm-d-server-introspection/devices
curl http://localhost:8000/plugins/llm-d-server-introspection/kv-cache
```

## Test

```bash
pip install -e .[test]

# fast unit tests (schema + FastAPI route, no engine, no real model)
pytest tests/ -v

# real server e2e (downloads model weights, launches a subprocess server)
RUN_VLLM_E2E=1 pytest tests/server_introspection/test_e2e.py -v
```

Lint and format with [ruff](https://docs.astral.sh/ruff/), matching what CI runs:

```bash
ruff check .
ruff format --check .
```

## Adding an extension

Add a subpackage under `llm_d_api_extensions_vllm/`, register its plugin class as a `vllm.endpoint_plugins` entry point in [`pyproject.toml`](pyproject.toml) and namespace its routes under `/plugins/llm-d-<extension>/...`. Keep the entry point key identical to the plugin class's `name` attribute. The key is what operators put in `VLLM_PLUGINS`.
