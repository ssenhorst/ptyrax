# Ptyrax

Ptyrax is a inverse-problem solver using Automatic Differentiation built on Jax. Mainly developed for ptychography, but with support for simple extension to custom imaging models.

## Quickstart

Reconstructing a dataset is easy using [uv package manager](https://docs.astral.sh/uv/). A basic reconstruction can be initiated using

```{code} bash
uvx --python 3.12 --from 'tensorboard' --with 'setuptools<82' tensorboard --logdir logs/ & \
uvx git+https://github.com/ssenhorst/ptyrax.git reconstruct https://surfdrive.surf.nl/public.php/dav/files/sakpFtVESDmncRH -c https://surfdrive.surf.nl/public.php/dav/files/2W3HjTfprrX5fLn reconstruction.hdf5 -l logs/
```

This will:

- Download an example dataset to a temporary directory
- Create a logs/ folder to track the reconstruction progress
- Download a configuration file and save it to the log folder for the current run
- Start reconstructing the example dataset
- Start a background tensorboard instance (usually on port `6006`) to track the progress during reconstruction
- Save the reconstruction in binary (`reconstruction.eqx`) and hdf5 (`reconstruction.hdf5`) format

Note that the tensorboard instance will start in the background. To close it, it can be brought back to the foreground using the `fg` command.
To enable hardware acceleration based on CUDA, it is better to install the package in a virtual environment.

## Installation

### Using Make (recommended)

If you are on a system which has the `make` command, installation of the virtual environment is easiest

```{code} bash
make install
```

### Local installation (uv)

We recommend using the [uv](https://docs.astral.sh/uv/) package manager. To install, use

#### macOS and Linux

```{code} bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows

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

### Local installation (pip)

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

## Usage

If Ptyrax was not installed via uv, replace all `uv run ptyrax` commands with `python3 -m ptyrax`.
Ptyrax can be invoked using

```{bash}
uv run ptyrax [reconstruct, simulate, experiment] [ptychogram_path] [reconstruction_filename] {opts}
```

For example, you can reconstruct the example dataset in `data/lenspaper.hdf5` using

```{bash}
uv run ptyrax reconstruct https://surfdrive.surf.nl/public.php/dav/files/sakpFtVESDmncRH data/lenspaper_reconstruction.hdf5 --log_dir logs/lenspaper --config configs/lenspaper.yaml
```

This will generate a reconstruction output `lenspaper_reconstruction.hdf5` in the `log_dir` directory. The specific parameters used for the reconstruction are set using the [gin-config](https://github.com/google/gin-config) file specified by `config`. The reconstruction process may be monitored by use of [tensorboard](https://github.com/tensorflow/tensorboard) using (in a new shell):

```{bash}
uv run python -m tensorboard.main --logdir logs/lenspaper --samples_per_plugin images=200
```

After the reconstruction process has completed, the output will become available in the `log_dir` in hdf5 format (`lenspaper_reconstruction.hdf5`) and as a binary [equinox](https://docs.kidger.site/equinox/all-of-equinox/) file (`lenspaper_reconstruction.eqx`, for use when restarting from an intermediate solution). In addition to the output, everything that is written to `stdout` and `stderr` is logged to `out.log`. The tensorboard logs are converted to hdf5 `tensorboard_logs.hdf5` for easy loading, and `metadata.yaml` specifies the exact version of the code that was used ptyrax, including possible diffs if under development.

## Documentation

The documentation is hosted on [readthedocs](https://ptyrax.readthedocs.io/en/latest/). We recommend getting started with a [basic reconstruction tutorial](https://ptyrax.readthedocs.io/en/latest/tutorials/basic_reconstruction.html).

Alternatively, the docs can be built and viewed using

```{bash}
make docs-serve
```

## Contributing

Contributions welcome!
