# -----------------------------
# Configuration
# -----------------------------

.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

UV := uv

DOCS_BUILDDIR = docs/_build
DOCS_SOURCEDIR = docs/source
CODEDIR = ptyrax
TESTDIR = ptyrax

# -----------------------------
# Utility
# -----------------------------

.PHONY: help
help:
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  install        Install uv (if needed) and sync deps"
	@echo "  sync           Sync dependencies (dev)"
	@echo ""
	@echo "  test           Run tests"
	@echo "  lint           Run linters"
	@echo "  format         Format code"
	@echo "  docs           Build docs"
	@echo ""
	@echo "  ci             Run all CI checks"
	@echo ""

# -----------------------------
# Bootstrap
# -----------------------------

.PHONY: install-uv
install-uv:
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv already installed."; \
	fi

.PHONY: install
install: install-uv sync

.PHONY: sync
sync:
	$(UV) sync

# CI-safe dependency install
.PHONY: ci-install
ci-install: install-uv
	$(UV) sync --frozen

# -----------------------------
# Actions (assume deps present)
# -----------------------------

.PHONY: test
test:
	$(UV) run --dev pytest $(TESTDIR)

.PHONY: lint
lint:
	$(UV) run --dev ruff check $(CODEDIR)

.PHONY: format
format:
	$(UV) run --dev ruff format $(CODEDIR)
	$(UV) run --dev docformatter -r --in-place $(CODEDIR)

.PHONY: docs
docs: clean format
	$(UV) run --extra docs python -m sphinx -b html $(DOCS_SOURCEDIR) $(DOCS_BUILDDIR)

.PHONY: docs-serve
docs-serve:
	$(UV) run --extra docs sphinx-autobuild  $(DOCS_SOURCEDIR) $(DOCS_BUILDDIR) --open-browser
# -----------------------------
# CI Targets
# -----------------------------

.PHONY: ci-lint
ci-lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: ci-test
ci-test:
	$(UV) run pytest --maxfail=1 --junitxml=test_results.xml

format-docstrings:
	$(UV) run docformatter \
		--in-place \
		--wrap-summaries 120 \
		--wrap-descriptions 120 \
		--recursive \
		-v \
		$(CODEDIR)

.PHONY: ci-docs
ci-docs: clean docs ci-prune
	echo "Docs built successfully"

.PHONY: ci-prune
ci-prune:
	$(UV) cache prune --ci

.PHONY: ci
ci: ci-install ci-lint ci-test ci-docs

.PHONY: clean
clean:
	rm -rf $(DOCS_BUILDDIR)
	rm -rf ./junit.xml