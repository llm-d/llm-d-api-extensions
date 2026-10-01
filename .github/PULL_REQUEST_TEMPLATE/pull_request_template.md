---
name: Pull request
about: Create a pull request
---

## Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce.

### Test configuration

- Inference server and version:
- Python version and platform:
- `VLLM_PLUGINS` value and relevant `vllm serve` flags:

## Checklist

- [ ] My changes follow the style guidelines of this project
- [ ] I have performed a self-review of my own changes
- [ ] I have signed off my commits (`git commit -s`, per the [DCO](https://developercertificate.org/))
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest tests/ -v` passes
- [ ] Any new endpoint is opt-in, namespaced under `/plugins/llm-d-<extension>/...` and documented in `vllm/README.md`
- [ ] I have updated the documentation accordingly
