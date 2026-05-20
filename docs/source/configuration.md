# Configuration Guide

This page explains how ptyrax uses configuration files to wire datasets, models and training. It is written for users who want to run reconstructions by editing a YAML file rather than changing code. Ptyrax accepts one or more configuration files in yaml format, passed with `--config` on the CLI. Configuration may also be loaded interactively in a notebook using
```{code} python
import gin
from ptyrax.__main__ import initialize_gin_configs
gin.enter_interactive_mode()
initialize_gin_configs("path/to/config.yaml")
```

A full configuration file, containing all the configurable parameters, is available here {download}`configs/all_keys_example.yaml`.

## The yaml format

The custom yaml format differs from the defaults used by [gin-config](https://github.com/google/gin-config). Instead of adding scopes using `scope/target`, in ptyrax, the scopes are top level keys. The global scope is `__main__`, so all globally replaced configuration values must be within `__main__`.

[gin-config](https://github.com/google/gin-config) allows referencing functions as configurable items by prefixing it with '`@`'. In our yaml format, this is achieved by the special `!@` tag.

## Example configuration file (minimal)
Let's go through the configuration file for the example lenspaper dataset, line by line. The full configuration is:
```yaml
__main__:
  train_session:
    num_epochs: 50
  reconstruct:
    preprocess_functions:
      - !@ center_scan_positions
      - !@ wavelength_units
      - !@ normalize_by_mean_intensity
      - !@ apply_orientation

  apply_orientation:
    orientation: 4
  
  PtychographyModel_initializer:
    probe_initializer: !@ probe/aperture

  initialize_optimizer_and_state:
    optimizers:
      - !@ fast/make_optimizer_specification()
      - !@ slow/make_optimizer_specification()

slow:
  make_optimizer_specification:
    base_optimizer: !@ optax.adam
    match_patterns:
      - ".*reflection_coefficient.*"
    learn_rate: -1e-5

fast:
  make_optimizer_specification:
    base_optimizer: !@ optax.adam
    match_patterns:
      - ".*probe.data"
    learn_rate: -1e-4
probe:
  aperture:
    radius: [20, 20]
    normalize: True
```

The first top level key

```yaml
__main__:
```

Specifies that we are in the global scope.

```yaml
__main__:
  train_session:
    num_epochs: 50
```

This will replace the keyword argument `num_epochs` for the {py:func}`~ptyrax.reconstruct.train_session` function to 50.
This means that, if the configuration is loaded and the {py:func}`~ptyrax.reconstruct.train_session` function is called, the `num_epochs` argument has already been filled in.
It may still be overrided, but the default is set.

```yaml
__main__:
  reconstruct:
    preprocess_functions:
      - !@ center_scan_positions
      - !@ wavelength_units
      - !@ normalize_by_mean_intensity
      - !@ apply_orientation
```

Now we are specifying a different type of function argument to {py:func}`~ptyrax.reconstruct.reconstruct`: a list of functions. This is the power of gin: we can specify the behavior of our code chaining together different functions that will be executed. In this case, we specify a list of preprocessing functions. These are all functions that take as input instances of {py:class}`~ptyrax.dataset.ImageDataset`, and output preprocessed versions of this. For example {py:class}`~ptyrax.dataset.apply_orientation` takes a dataset and changes its axes and coordinates to match the convention used in ptyrax. Now, we can even change the behavior of these functions, by configuring them in the same way:

```yaml
__main__:
  apply_orientation:
    orientation: 4
```

This sets the specific orientation to be used in the preprocessing function that we just added to our list of preprocessing functions. But what if we want our functions to do behave one way when referenced in one position, but a different way for a different reference? This is possible using scoping:

```yaml
__main__:
  train_session:
    PtychographyModel_initializer:
      probe_initializer: !@ probe/aperture
```

Instead of passing the {py:func}`~ptyrax.initializers.aperture` function directly, we have added the `probe` scope. This means that the aperture function that will be used here, will have its arguments configured by the configuration under the `probe` top-level key in the yaml file:

```yaml
probe:
  aperture:
    radius: [20, 20]
    normalize: True
```

We also use scoping to define our optimizer:

```yaml
  initialize_optimizer_and_state:
    optimizers:
      - !@ fast/make_optimizer_specification()
      - !@ sparse/make_optimizer_specification()
```

Here we reference the same function, {py:func}`~ptyrax.training.make_optimizer_specification`, two times with different scopes. Although it is the same function, it will be called with different arguments according to the scope

```yaml
slow:
  make_optimizer_specification:
    base_optimizer: !@ optax.adam
    match_patterns:
      - ".*reflection_coefficient.*"
    learn_rate: -1e-5

fast:
  make_optimizer_specification:
    base_optimizer: !@ optax.adam
    match_patterns:
      - ".*probe.data"
    learn_rate: -1e-4
```

As we see, the slow optimizer, which matches `reflection_coefficient` parameters of the model, will be optimized with a lower `learn_rate` than the fast optimizer, which matches the `probe` parameters of the model.

## Advanced: multiple config files and experiment sweeps

You can pass multiple `--config` files to compose behaviour; one file may contain base settings and another variable overrides. The order here matters: config files passed last will override config files passed first.
