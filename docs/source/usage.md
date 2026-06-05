# Usage

If Ptyrax was not installed via uv, replace all `uv run ptyrax` commands with `python3 -m ptyrax`.
Ptyrax can be invoked using

```{code} bash
uv run ptyrax [reconstruct, simulate, experiment] [ptychogram_path] [reconstruction_filename] {opts}
```

For example, you can reconstruct the example dataset in `data/lenspaper.hdf5` using

```{code} bash
uv run ptyrax reconstruct https://surfdrive.surf.nl/public.php/dav/files/sakpFtVESDmncRH data/lenspaper_reconstruction.hdf5 --log_dir logs/lenspaper --config configs/lenspaper.yaml
```

This will generate a reconstruction output `lenspaper_reconstruction.hdf5` in the `log_dir` directory. The specific parameters used for the reconstruction are set using the [gin-config](https://github.com/google/gin-config) file specified by `config`. The reconstruction process may be monitored by use of [tensorboard](https://github.com/tensorflow/tensorboard) using (in a new shell):

```{code} bash
uvx --python 3.12 --from 'tensorboard' --with 'setuptools<82' tensorboard --logdir logs/lenspaper --samples_per_plugin images=200
```

After the reconstruction process has completed, the output will become available in the `log_dir` in hdf5 format (`lenspaper_reconstruction.hdf5`) and as a binary [equinox](https://docs.kidger.site/equinox/all-of-equinox/) file (`lenspaper_reconstruction.eqx`, for use when restarting from an intermediate solution). In addition to the output, everything that is written to `stdout` and `stderr` is logged to `out.log`. The tensorboard logs are converted to hdf5 `tensorboard_logs.hdf5` for easy loading, and `metadata.yaml` specifies the exact version of the code that was used ptyrax, including possible diffs if under development.
