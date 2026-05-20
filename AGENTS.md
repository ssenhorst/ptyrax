<!-- Auto-generated guidance for AI coding agents working on Ptyrax -->
# Copilot instructions for Ptyrax

This file contains concise, actionable guidance for AI coding agents to be immediately productive in this repository.

- **Big picture:** `ptyrax` is a JAX-based ptychography reconstruction framework. Core runtime is a CLI/UV entrypoint (`uv run ptyrax reconstruct`) which loads a dataset, applies preprocessing, generates and reconstructs a `PtychographyModel`, and writes reconstructions and TensorBoard logs into a `log_dir`.

- **Where to look first:**
  - Project root README: [README.md](../README.md) — quickstart and `uv` usage.
  - Runner / CLI source: `ptyrax/` (package) — search for `__main__` and `reconstruct` entrypoints.
  - Configs: [configs/](../configs/) — experiment YAMLs show typical preprocessing and optimizer patterns (example: [configs/lenspaper.yaml](../configs/lenspaper.yaml)).
  
- **Coordinate conventions:**
  - The project uses the coordinate system conventions from the CXI format. This means that the global coordinates are defined such z is along the incoming beam direction and
  y is chosen vertically.
  - 2D arrays are indexed with x along dim 0 and y along dim 1 (equivalent to indexing parameter 'ij' for numpy)
  - Samples in simple reflection geometries only have a rotation around the y-axis. The z-axis is chosen mostly along the direction of the incoming ray, so positive in the global coordinates
  - Detectors in simple reflection geometries are placed (near to) the specular reflection. The detector z-direction is also chosen along the direction of the incoming beam, but since this beam originates from a reflection at the sample, it may have either positive or negative z coefficients.

- **Run / build / test workflows (explicit):**
  - Local install and dependency sync: run the Makefile targets in [Makefile](../Makefile): `make install` (installs `uv` and runs `uv sync`).
  - Run a reconstruction (example):

    uv run ptyrax data/lenspaper.hdf5 data/lenspaper_reconstruction.hdf5 --log_dir logs/lenspaper --config configs/lenspaper.yaml

  - GPU runs: include the `cuda` extra for JAX: `uv run --extra "cuda" ptyrax ...`.
  - Tests: `make test` (runs `uv run --dev pytest tests`).
  - Docs: `make docs` / `make docs-serve` (Sphinx with `sphinx-book-theme` + MyST/Myst-NB). Use `uv run --extra docs` or `make docs` to build.

- **Config conventions and patterns:**
  - Experiment configs are YAML and use `!@` tags to reference constructors/functions registered in code (see `configs/EX0094/*.yaml` for complex examples). Treat YAML values as the canonical source for experiment behavior.
  - Common sections: `__main__` (top-level runtime overrides), `PtychographyModel`, `train_session`, `probe`, `sample`, and optimizer groups like `fast` / `sparse`.

- **Ptyrax CLI**
  - The CLI takes commands in the format `ptyrax [experiment | simulate | reconstruct] [...]`. The different entrypoints are found in `ptyrax/simulate.py`, `ptyrax/experiment.py` and `ptyrax/reconstruct.py`.

- **Code patterns to respect:**
  - Uses JAX, `equinox` for model state/serialization — continuations/restarts expect `.eqx` artifacts alongside HDF5 outputs in the `log_dir`.
  - Optimizers and schedules are configured via factory functions referenced from YAML (`!@ fast/optax.sgdr_schedule()`), so changes should preserve the outward YAML API.
  - Logging: TensorBoard logs are created to `log_dir` and converted to HDF5; use the same directory layout when adding features.
  - Adhere to SOLID code principles. Minimize the use of helper functions. If the best solution to a problem requires some refactoring, ask for feedback on the refactor rather than going for the simple but messy solution.
  - Avoid acronyms except for very special cases where they are used in common speech.

- **Debugging workflow**
  - After making changes that could be an entire commit, also write several tests for this code part. This will ensure good code coverage.
  - Always execute the tests you created to ensure that they pass. If not, fix the errors.

- **Important files and places to edit:**
  - [Makefile](../Makefile) — developer shortcuts and CI targets.
  - [pyproject.toml](../pyproject.toml) — dependency groups and extras (`cuda`, `docs`).
  - [configs/](../configs/) — canonical experiment configurations; prefer editing/adding configs over embedding parameters in code.

- **CI / docs publishing:**
  - This project uses GitLab CI (see `.gitlab-ci.yml`) — docs are published via GitLab Pages. The docs build wrapper is `scripts/docs_build.sh` and `Makefile` targets invoke Sphinx; update the GitLab pipeline if you change build steps.

- **When modifying configs or model code:**
  - Preserve `!@` factory signatures used by YAML (search for usages in `configs/` before renaming functions).
  - Add unit tests under `tests/` that exercise the YAML-backed configuration flow (use `make test`).

- **Quick checks to run after changes:**
  - `make lint` and `make format` (via `uv`) — repository expects `ruff` formatting rules from `pyproject.toml`.
  - `make test` — run tests with the dev `uv` environment.
  - Example-reconstruction smoke test: run `uv run python scripts/run_reconstruction_example.py` against a small dataset.
  
- **Package manager:**
  - Uses `uv` package manager - All CLI commands in the environment should be prepended by `uv run [command]`.
  - Requirements are stored in `pyproject.toml`.
  - Additional requirements (e.g. for documentation) are stored as extras - use `uv run --extra docs [command]` to add
  these to the environment

If any section is unclear or you want more examples (e.g., specific `ptyrax/` modules to inspect), tell me which area to expand.
