# llm-d-api-extensions
#
# Convenience targets that wrap the per server packages. Each package is also a
# standard Python project you can drive directly from its own directory.

VLLM_DIR := vllm
RUFF_VERSION := 0.15.11

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the vLLM extensions with test extras (editable)
	pip install -e '$(VLLM_DIR)[test]'

.PHONY: lint
lint: ## Run ruff lint and format checks (what CI runs)
	cd $(VLLM_DIR) && ruff check . && ruff format --check --diff .

.PHONY: fix
fix: ## Apply ruff auto-fixes and formatting
	cd $(VLLM_DIR) && ruff check --fix . && ruff format .

.PHONY: test
test: ## Run the fast unit tests (no engine, no model weights)
	cd $(VLLM_DIR) && pytest tests/ -v

.PHONY: test-e2e
test-e2e: ## Run the opt-in e2e tests (downloads model weights, launches a server)
	cd $(VLLM_DIR) && RUN_VLLM_E2E=1 pytest tests/server_introspection/test_e2e.py -v

.PHONY: build
build: ## Build the vLLM extensions distribution
	cd $(VLLM_DIR) && python -m build

.PHONY: clean
clean: ## Remove build and test artifacts
	rm -rf $(VLLM_DIR)/build $(VLLM_DIR)/dist $(VLLM_DIR)/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(VLLM_DIR)/.pytest_cache $(VLLM_DIR)/.ruff_cache
