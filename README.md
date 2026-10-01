# llm-d API Extensions

API extensions for inference servers, packaged for [llm-d](https://github.com/llm-d/llm-d).

llm-d needs data out of the inference server. Information like how it was launched, what hardware it sits on, how big its KV cache turned out to be. This data is then used to make routing and scheduling decisions. Rather than adding those endpoints to an inference server's core, this repo hosts them as **extensions** that are installed alongside the server and loaded through the server's own extension points.

For vLLM, that means using the [endpoint plugin framework](https://docs.vllm.ai/en/latest/design/endpoint_plugins/). The framework is an extension mechanism where a Python package is registered as a `vllm.endpoint_plugins` entry point and vLLM attaches its routes at startup. No fork, no patch and no core change required on the server.

## Layout

The repo is inference server agnostic at the top level. Each supported server gets its own directory holding an independently installable package:

| Directory | Package | Server |
|---|---|---|
| [`vllm/`](vllm/) | `llm-d-api-extensions-vllm` | [vLLM](https://github.com/vllm-project/vllm) |

Support for another server (for example SGLang) would land as a sibling directory once that server exposes a comparable extension point.

## Extensions

| Extension | Server | Endpoints |
|---|---|---|
| [Server introspection](vllm/README.md) | vLLM | `GET /plugins/llm-d-server-introspection/{config,devices,kv-cache}` |

## Quick start

```bash
pip install ./vllm

VLLM_PLUGINS=llm_d_server_introspection_config vllm serve <model>

curl http://localhost:8000/plugins/llm-d-server-introspection/config
```

Extensions are opt-in. vLLM only loads the plugins named in `VLLM_PLUGINS`, so installing this package does not change the behaviour of a server that does not ask for it. See [`vllm/README.md`](vllm/README.md) for the full endpoint reference, the response schemas and the flags each extension needs.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Maintainers are listed in [MAINTAINERS.md](MAINTAINERS.md). To report a vulnerability, follow [SECURITY.md](SECURITY.md) rather than opening an issue.

## License

Apache 2.0 — see [LICENSE](LICENSE).
