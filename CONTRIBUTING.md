# Contributing to llm-d-api-extensions

Thank you for your interest in contributing. This repo hosts API extensions that llm-d loads into inference server. Contributions of all kinds — bug reports, docs, extensions for new servers — are welcome.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md). Maintainers are listed in [MAINTAINERS.md](MAINTAINERS.md) and approvers/reviewers in [OWNERS](OWNERS).

## Scope

Extensions in this repo integrate with an inference server through that server's own documented extension points. A change that requires forking or patching the inference server does not belong here. Go upstream to the server instead.

The top level of the repo is inference server agnostic. Server specific code lives in a directory named for that server (for example, [`vllm/`](vllm/)).

## Reporting issues

Before filing, search the [issue tracker](https://github.com/llm-d/llm-d-api-extensions/issues) for an existing report. When you do file one, include the inference server version, how the server was launched (the `VLLM_PLUGINS` value and relevant flags), what you expected, and what happened instead.

Do **not** open a public issue for a security vulnerability. Follow [SECURITY.md](SECURITY.md).

## Development

The vLLM extensions are a standard Python package. From `vllm/`:

```bash
pip install -e .[test]

# fast unit tests: no engine, no model weights
pytest tests/ -v

# lint and format, exactly what CI runs
ruff check .
ruff format --check .
```

Formatting and linting are enforced in CI by [ruff](https://docs.astral.sh/ruff/). The configuration lives in [`vllm/ruff.toml`](vllm/ruff.toml). Run `ruff format .` to fix formatting and `ruff check --fix .` for the auto-fixable lint rules before you push.

The end-to-end tests launch a real vLLM server and download model weights, so they are opt-in and are not run by CI:

```bash
RUN_VLLM_E2E=1 pytest tests/server_introspection/test_e2e.py -v
```

### Adding an endpoint

New endpoints should:

- Be **opt-in.** Register as a separate `vllm.endpoint_plugins` entry point so operators enable exactly what they want through `VLLM_PLUGINS`.
- **Namespace their routes** under `/plugins/llm-d-<extension>/...`. vLLM does not enforce route conflicts and a later route silently shadows an earlier one.
- Keep the entry point key identical to the plugin class's `name` attribute, since that key is what operators put in `VLLM_PLUGINS`.
- Set `required_tasks` honestly. Use `None` only if the endpoint genuinely needs no engine and return `503` rather than crashing when a dependency is absent.
- **Degrade, not crash.** If the endpoint depends on a vLLM API that is not in a released build yet, feature detect it and fall back.
- Ship a Pydantic response model and unit tests that need neither an engine nor model weights.
- Be documented in [`vllm/README.md`](vllm/README.md) with an example response.

## Submitting a pull request

1. **Discuss first for anything significant.** Opening an issue before a large change avoids rework and makes sure the direction fits llm-d's needs.
2. Fork the repo and work on a branch.
3. **Sign off your commits.** This project requires the [Developer Certificate of Origin](https://developercertificate.org/); use `git commit -s`. CI rejects unsigned commits.
4. Make sure `pytest`, `ruff check` and `ruff format --check` all pass locally.
5. Update the docs alongside the code. A new endpoint without a README entry is incomplete.
6. Open the PR with a clear description of what changed and why, linking the issue it addresses.

## Code review

Pull requests are reviewed by the maintainers listed in [OWNERS](OWNERS). Reviews arr done on a best effort basis. Maintainers may prioritise their own work. Addressing review feedback promptly is the fastest path to a merge.

## Community

- **Slack:** the llm-d developer Slack at [llm-d.slack.com](https://llm-d.slack.com)
- **Project:** [llm-d/llm-d](https://github.com/llm-d/llm-d)

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE). Source files carry an `# SPDX-License-Identifier: Apache-2.0` header. Please add that on new files.
