BIN      := heracles
CARGO    ?= cargo
ARGS     ?=

.DEFAULT_GOAL := help
.PHONY: help build release run test fmt fmt-check lint check install uninstall clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Debug build
	$(CARGO) build

release: ## Optimized build
	$(CARGO) build --release

run: ## Run with ARGS, e.g. make run ARGS="search 'auth flow'"
	$(CARGO) run -- $(ARGS)

test: ## Run tests
	$(CARGO) test

fmt: ## Format code
	$(CARGO) fmt

fmt-check: ## Verify formatting
	$(CARGO) fmt --check

lint: ## Clippy with warnings as errors
	$(CARGO) clippy --all-targets -- -D warnings

check: fmt-check lint test ## Everything CI should run

install: ## Install the binary to ~/.cargo/bin and initialize ~/.heracles
	$(CARGO) install --path . --locked
	$(BIN) init

uninstall: ## Remove the binary (keeps ~/.heracles)
	$(CARGO) uninstall $(BIN)

clean: ## Remove build artifacts
	$(CARGO) clean
