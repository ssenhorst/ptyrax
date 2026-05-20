# Quickstart — Minimal Reconstruction

Ptyrax reconstructs ptychographic datasets from a configuration and a dataset file. This guide shows the minimal steps to run a reconstruction locally and inspect the result. We will assume you use the [uv package manager](https://docs.astral.sh/uv/).
A basic reconstruction can be initiated using

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
To enable hardware acceleration based on CUDA, it is better to install the package in a virtual environment. In your `uv` project directory, use: 

```{code} bash
uv add git+https://github.com/ssenhorst/ptyrax.git --with cuda
```

then run

```{code} bash
uvx --python 3.12 --from 'tensorboard' --with 'setuptools<82' tensorboard --logdir logs/ & \
uv run ptyrax reconstruct https://surfdrive.surf.nl/public.php/dav/files/sakpFtVESDmncRH -c https://surfdrive.surf.nl/public.php/dav/files/2W3HjTfprrX5fLn reconstruction.hdf5 -l logs/
```
