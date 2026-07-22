# Herculens Wrapper Implementation Overview

This note is for quickly understanding and extending this repository on another
machine. It focuses on code structure, data flow, sampler behavior, and the
important implementation conventions used by the wrapper.

## Repository Layout

```text
herculens_wrapper/
├── run.py                         # Main entry point and orchestration
├── hmc_pipeline.md                # Detailed HMC sampling note
├── configs/
│   ├── config_svi.py              # Typical SVI configuration
│   ├── config_hmc.py              # Typical HMC configuration
│   ├── config_pipeline.py         # Multi-run SVI -> HMC pipeline config
│   └── config_param.py            # Parametric/non-pixelated style config
├── herculens_wrapper/
│   ├── models.py                  # LensImage wrapper, NumPyro prob_model, init loading
│   ├── samplers.py                # SVI, Optax, HMC, diagnostics, metrics
│   ├── custom_gibbs.py            # Gibbs-within-HMC wrapper around NUTS kernels
│   ├── visualizations.py          # Image/source/model/residual/corner plots
│   ├── utils.py                   # Config/data/json/path/helper utilities
│   └── __init__.py
└── utils/                         # Standalone helper scripts and checks
```

## Running

Typical command:

```bash
python run.py configs/config_svi.py
python run.py configs/config_hmc.py
python run.py configs/config_pipeline.py
```

`run.py` loads the config module, reads FITS image/noise/PSF data, builds the
Herculens `LensImage`, builds the NumPyro `prob_model`, loads or samples initial
parameters, runs the requested sampler, and writes outputs into `save_path`.

The config file normally provides:

- `lens_mass_config()`
- `lens_light_config()`
- `source_light_config()`
- `point_source_config()`
- `config_data()`
- `config_noise()`
- `config_psf()`
- `config_numerics()`
- `config_pipeline()` or sampler-specific runtime options

## Main Data Flow

```text
config_*.py
  -> run.py: build_and_run()
    -> load image/noise/PSF data
    -> assemble type_list and param_list
    -> models.py: create_lens_image()
    -> models.py: create_prob_model()
    -> models.py: get_init_params()
    -> samplers.py: run_svi(), run_optax(), or run_hmc()
    -> save kwargs_result.json, arrays, metrics, plots, diagnostics
```

## Key Concepts

### Constrained vs Unconstrained Parameters

Most wrapper-level logic uses constrained physical parameters. NumPyro/NUTS
internally works in unconstrained space.

Important helpers in `samplers.py`:

- `to_unconstrained(prob_model, params)`
- `to_constrained(prob_model, params)`

Use constrained parameters for:

- `prob_model.params2kwargs(...)`
- `prob_model.log_likelihood(...)`
- saved `kwargs_result.json`
- initialization loaded from prior runs

### `params2kwargs`

`prob_model.params2kwargs(params)` converts sampled parameter dictionaries into
Herculens model kwargs:

- `kwargs_lens`
- `kwargs_lens_light`
- `kwargs_source`
- `kwargs_point_source`

For a Matérn pixelated source, it reconstructs physical source pixels from:

```text
pixels_wn_source_grid, n_source_grid, rho_source_grid, sigma_source_grid
```

### Deterministic Outputs

The NumPyro model registers derived products using `numpyro.deterministic`.
Important deterministic sites are:

- `pixels_source_grid`: physical source pixels derived from source parameters
- `model_image`: full model image derived from lens/source/light parameters

For HMC final results and batch diagnostics, source/model outputs use posterior
deterministic summaries:

```text
median(pixels_source_grid samples)
median(model_image samples)
```

This is preferred over:

```text
FFT(median(pixels_wn), median(n), median(rho), median(sigma))
```

because source pixels are nonlinear derived quantities. Median of the derived
quantity is a better posterior image summary than deriving once from median
inputs.

For SVI, the current final deterministic output is evaluated at the guide median
parameter set. SVI does not currently sample many guide draws for final source
and model summaries.

## SVI Flow

Implemented in `samplers.py:run_svi`.

High-level flow:

```text
init params
  -> AutoLowRankMultivariateNormal guide
  -> NumPyro SVI
  -> guide.median(result.params)
  -> deterministic outputs evaluated at guide median
  -> save kwargs_result.json, kwargs_sigma.json, plots, metrics
```

SVI also saves:

- `svi_loss_history.json`
- `svi_guide_params.pkl`
- `kwargs_sigma.json`

`svi_guide_params.pkl` is useful for uncertainty estimates, but HMC
warm-starting mainly uses the saved result kwargs and pixel arrays.

## HMC Flow

Implemented in `samplers.py:run_hmc`. See `hmc_pipeline.md` for more detail.

HMC requires a previous run through `init_params_path`, usually an SVI run.

High-level flow:

```text
load constrained init params from previous run
  -> trace prob_model to identify active latent sample sites
  -> remove deterministic sites from init params
  -> optionally jitter initial lens-mass parameters per chain
  -> build Gibbs-within-HMC kernels
  -> warmup on batch 0
  -> sample in checkpointed batches
  -> save batch diagnostics
  -> concatenate all batches
  -> save final samples, deterministic summaries, metrics, plots
```

Default grouping when `disable_gibbs=False`:

```text
Group 1:
  pixels_wn_source_grid
  n_source_grid, rho_source_grid, sigma_source_grid
  lens_light_* parameters
  other non-lens-mass parameters

Group 2:
  lens_* mass parameters, excluding lens_light_*
```

Each group is updated by a separate NUTS kernel inside `custom_gibbs.py`.
Warmup/adaptation also uses this grouped Gibbs-within-HMC structure.

Batching is controlled by:

- `num_warmup_hmc_numpyro`
- `num_samples_hmc_numpyro`
- `checkpoint_interval_hmc_numpyro`
- `num_chains_hmc_numpyro`

After each batch:

- samples are moved to CPU NumPy arrays
- `hmc_samples_batch_{i}.npz` is saved
- `hmc_checkpoint.pkl` is updated
- diagnostic plots and summaries are generated

If a run is interrupted, rerunning with the same `save_path` resumes from the
checkpoint. Later batches use `num_warmup=0` and continue from the last state.

## Source Plane and Adaptive Grid

Adaptive pixelated sources are configured in `source_light_config()` with:

```python
'pixel_adaptive_grid': True
'pixel_grid_shape': ...
```

The source-plane extent is determined from the ray-traced image-plane source
mask and `source_grid_scale`. If `source_arc_mask_path` is not provided, the
wrapper now falls back to a full-image mask before constructing Herculens
`LensImage`, so adaptive source grids can still run.

The support mask is visualization-only. It should not zero out saved source
arrays or HMC/SVI model parameters.

## Outputs

Common final outputs:

```text
kwargs_result.json                  # Saved constrained model kwargs
kwargs_source_pixels.npy            # Physical source pixels
kwargs_source_pixels_wn.npy         # White-noise source parameters
kwargs_sigma.json                   # Parameter uncertainty summary
metrics.json                        # Chi2, reduced chi2, BIC, log likelihood
modeling_result.npz                 # best_fit_model, data, noise, mask
best_fit_model_linear.png
best_fit_model_log.png
image_plane.png
source_plane_linear.png
source_plane_log.png
lens_light_subtracted_image.png
lens_light_subtracted_image_log.png
ring_model_comparison_linear.png
ring_model_comparison_log.png
mass_profile_convergence.png
```

HMC final outputs also include:

```text
hmc_numpyro_samples.npz
hmc_samples_batch_{i}.npz
hmc_checkpoint.pkl
mcmc_summary_final.txt
mcmc_diagnostics_final.png
kwargs_loglike.json
kwargs_loglike_source_pixels.npy
kwargs_loglike_source_pixels_wn.npy
best_fit_model_loglike_linear.png
best_fit_model_loglike_log.png
hmc_log_likelihoods.npy
```

Diagnostic `.npy` files inside `diagnostics/` are currently disabled/commented
in `samplers.py` to reduce disk usage. The comments are left near the call sites
so they can be uncommented later.

## Important Visualizations

### `best_fit_model_*.png`

Panels:

```text
model image | image data | normalized residual
```

Residual convention:

```text
(model - data) / noise_map
```

### `ring_model_comparison_*.png`

Used to check ring modeling after lens-light subtraction.

Panels:

```text
model without lens light | image - lens light | residual
```

The residual is:

```text
(model_without_lens_light - (image - lens_light)) / noise_map
```

### `source_plane_*.png`

Shows reconstructed source-plane pixels. Visualization masks may set inactive
regions to zero for display only. Saved final `.npy` source arrays are not
masked by visualization support masks.

## Where to Edit Common Features

Add or change priors:

- `models.py:create_prob_model`
- `models.py:PowerSpectrum`

Change initialization / loading from previous runs:

- `models.py:get_init_params`
- `models.py:resolve_fixed_kwargs`

Change SVI behavior:

- `samplers.py:run_svi`

Change HMC grouping, jitter, checkpointing, diagnostics:

- `samplers.py:run_hmc`
- `custom_gibbs.py:MultiHMCGibbs`

Change final plots:

- `visualizations.py:generate_run_plots`

Change HMC batch diagnostic plots:

- `samplers.py:run_hmc`, intermediate diagnostics section

Change JSON and pixel-array saving:

- `utils.py:kwargs_best_to_json_pixelated_npy`

## Practical Notes

- Keep final source arrays unmasked. Apply support masks only for plots.
- HMC warm-start can load median parameters and pixel arrays from SVI outputs.
- HMC final source/model plots should use deterministic posterior medians when
  available.
- Convergence diagnostics should focus on sampled latent parameters, not huge
  deterministic arrays like `model_image` or `pixels_source_grid`.
- If HMC memory is high, reduce `checkpoint_interval_hmc_numpyro`; deterministic
  arrays are stored per sample within each batch.
- `source_arc_mask_path` is optional for adaptive grids; missing masks fall back
  to full-image support.

## Minimal Extension Checklist

When implementing a new feature:

1. Add config options in the relevant `configs/config_*.py`.
2. Read them in `run.py` or pass them through existing config dictionaries.
3. Add model behavior in `models.py` if it affects priors or model construction.
4. Add sampler behavior in `samplers.py` if it affects inference.
5. Add or update plots in `visualizations.py`.
6. Make sure saved kwargs stay JSON-compatible via `utils.py`.
7. Run:

```bash
python -m py_compile run.py herculens_wrapper/*.py
git diff --check
```

