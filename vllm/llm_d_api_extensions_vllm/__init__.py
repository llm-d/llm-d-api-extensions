# SPDX-License-Identifier: Apache-2.0
"""llm-d API extensions for the vLLM inference server.

Each subpackage is one extension family. Extensions integrate with vLLM
through documented extension points (`vllm.endpoint_plugins` entry points,
`--worker-extension-cls`) rather than by patching vLLM core.

- `server_introspection`: read-only `GET /plugins/llm-d-server-introspection/*`
  endpoints exposing how the server was launched, what hardware it is on and
  what its KV cache looks like after profiling.
"""
