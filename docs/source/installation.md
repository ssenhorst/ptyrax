# Installation

## Using Make (recommended)

If you are on a system which has the `make` command, installation of the virtual environment is easiest

```{code} bash
make install
```

## Local installation (uv)

We recommend using the [uv](https://docs.astral.sh/uv/) package manager. To install, use

### macOS and Linux

```{code} bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

Use `irm` to download the script and execute it with `iex`:

```{code} powershell
PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, ptyrax may be added to your `uv`-managed environment using

```{code}
uv add git+https://github.com/ssenhorst/ptyrax
```

Then, to run the base version of ptyrax without GPU support, use

```{code} bash
uv run ptyrax [reconstruct, simulate, experiment] [PTYCHOGRAM] [OUTPUT_FILENAME] [OPTIONS]
```

If you wish to make use of the GPU, you must install the CUDA-enabled version of jax. Instead use

```{code} bash
uv run --extra "cuda" ptyrax [reconstruct, simulate, experiment] [PTYCHOGRAM] [OUTPUT_FILENAME] [OPTIONS]
```

## Local installation (pip)

To install the package locally, use

```{bash}
python3 -m venv .venv
source ./.venv/bin/activate
```

to create and activate a virtual environment. To install non-GPU version, use

```{bash}
python3 -m pip install .
```

and for the GPU version use

```{bash}
python3 -m pip install .[cuda]
```
