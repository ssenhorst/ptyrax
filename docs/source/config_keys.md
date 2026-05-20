# Configuration Keys Reference (generated)

This page lists the `@gin.configurable` symbols exposed across the codebase and suggests a recommended `__main__` key for use in YAML configs. The table helps map callable names to the keys you'd bind under the `__main__` scope.

Notes
## Overview

**Table: configurable symbols and recommended __main__ keys**

| Recommended `__main__` key | Callable (module) | Signature (short) | Description |
|---|---|---|---|
| `__main__.regularize` | `ptyrax/training.py` | `regularize(field, weight=...)` | Top-level regulariser aggregator |
| `__main__.support_overlap` | `ptyrax/training.py` | `support_overlap(field, weight=...)` | Support-overlap penalty |
| `__main__.tv` | `ptyrax/training.py` | `tv(field, weight=..., tv_mode=...)` | Total-variation regulariser |
| `__main__._tv` | `ptyrax/training.py` | `_tv(a, weight)` | Internal TV helper |
| `__main__.scale_by_adam_thresholded` | `ptyrax/training.py` | `scale_by_adam_thresholded(...)` | Optimizer state scaler (Adam variant) |
| `__main__.adam_thresholded` | `ptyrax/training.py` | `adam_thresholded(...)` | Adam-like optimizer factory |
| `__main__.loss` | `ptyrax/training.py` | `loss(y_true, y_pred, ...)` | Loss function (multiple variants available) |
| `__main__.debug_gradient` | `ptyrax/training.py` | `debug_gradient(gradient, path_matchers=[])` | Helper to print/inspect gradients |
| `__main__.load_model_from_reconstruction` | `ptyrax/training.py` | `load_model_from_reconstruction(...)` | Load model helper from previous run |
| `__main__.loop_schedule` | `ptyrax/training.py` | `loop_schedule(...)` | Epoch/loop scheduling helper |
| `__main__.scale_schedule_steps_by_epoch` | `ptyrax/training.py` | `scale_schedule_steps_by_epoch(...)` | Adjust schedule per epoch |
| `__main__.make_optimizer_specification` | `ptyrax/training.py` | `make_optimizer_specification(...)` | Construct optimizer spec used by `initialize_optimizer_and_state` |
| `__main__.initialize_optimizer_and_state` | `ptyrax/training.py` | `initialize_optimizer_and_state(model, ...)` | Create optimizer objects and state |
| `__main__.preprocess_model` | `ptyrax/training.py` | `preprocess_model(model, ...)` | Model pre-processing hook |
| `__main__.simulation_postprocessing` | `ptyrax/training.py` | `simulation_postprocessing(...)` | Postprocess simulated outputs |
| `__main__.ptyrax` | `ptyrax/__main__.py` | `ptyrax(...)` | Main CLI entrypoint factory (bindings under `__main__.ptyrax.*`) |
| `__main__.main` | `ptyrax/__main__.py` | `main(*args, **kwargs)` | CLI `main` wrapper |
| `__main__.dataset.batch` | `ptyrax/dataset.py` | `batch(batch_size=..., shuffle_mode=...)` | Dataset batching helper (method on `Ptychogram`) |
| `__main__.from_hdf5` | `ptyrax/dataset.py` | `from_hdf5(path, convert_to_standard=True)` | Generic HDF5 loader (supports flat/CXI variants) |
| `__main__.scale` | `ptyrax/dataset.py` | `scale(ptychogram, scale)` | Scale transform for `Ptychogram` |
| `__main__.log_image` | `ptyrax/logger.py` | `log_image(writer, tag, tensor, step, ...)` | TensorBoard image logger helper |
| `__main__.log_on_train_start` | `ptyrax/logger.py` | `log_on_train_start(writer, dataset, debug=False)` | Hook executed on training start |
| `__main__.log_on_epoch_end` | `ptyrax/logger.py` | `log_on_epoch_end(writer, ...)` | Hook executed at epoch end |
| `__main__.binary_complex_image` | `ptyrax/initializers.py` | `binary_complex_image(...)` | Initializer: binary complex image |
| `__main__.binary_reflection_image` | `ptyrax/initializers.py` | `binary_reflection_image(...)` | Initializer: reflection image |
| `__main__.from_test_images` | `ptyrax/initializers.py` | `from_test_images(...)` | Create initial fields from test images |
| `__main__.aperture` | `ptyrax/initializers.py` | `aperture(...)` | Aperture initializer |
| `__main__.gaussian` | `ptyrax/initializers.py` | `gaussian(...)` | Gaussian initializer |
| `__main__.custom_initializer` | `ptyrax/initializers.py` | `custom(...)` | Custom initializer factory |
| `__main__.compute_center_of_mass_shift` | `ptyrax/utils.py` | `compute_center_of_mass_shift(...)` | Compute center-of-mass shift helper |
| `__main__.plot_complex` | `ptyrax/utils.py` | `plot_complex(...)` | Plotting helper for complex arrays |
| `__main__.plot_real` | `ptyrax/utils.py` | `plot_real(...)` | Plotting helper for real arrays |
| `__main__.load_hdf5` | `ptyrax/utils.py` | `load_hdf5(file_path, key_translation=None)` | Low-level HDF5 loader (used by dataset loaders) |
| `__main__.scaled_mean` | `ptyrax/utils.py` | `scaled_mean(a, scale=1.0)` | Small numeric helper |
| `__main__.custom_loader` | `ptyrax/equinox_model.py` | `custom_loader(...)` | Custom model loader for Equinox models |
| `__main__.interpolated_shift` | `ptyrax/spatial.py` | `shift_with_interpolation(x, center, target_shape)` | Shift helper (registered as `interpolated_shift`) |
| `__main__.shift_with_interpolation_unequal_pixel_size` | `ptyrax/spatial.py` | `shift_with_interpolation_unequal_pixel_size(...)` | Shift helper for unequal pixel sizes |
