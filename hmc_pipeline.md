# HMC Sampling Pipeline

The pipeline is implemented in [`run.py`](file:///users/xczhou/system/herculens_wrapper/run.py) (orchestration) and [`samplers.py`](file:///users/xczhou/system/herculens_wrapper/herculens_wrapper/samplers.py) (core logic), with the custom Gibbs kernel in [`custom_gibbs.py`](file:///users/xczhou/system/herculens_wrapper/herculens_wrapper/custom_gibbs.py).

---

## Stage 0 — Prerequisites (prior SVI run)

The HMC sampler **requires** a completed SVI run as a warm-start. The SVI run saves:
- `svi_guide_params.pkl` — the fitted `AutoLowRankMultivariateNormal` guide parameters
- `kwargs_result.json` — the best-fit parameter dict

Set `init_params_path` in [`config_hmc.py`](file:///users/xczhou/system/modeling_F277W/config_hmc.py) (line 381) to point to this directory.

---

## Stage 1 — Configuration & Model Construction (`run.py`)

```
build_and_run(config_hmc.py)
  ├── Load image / noise / PSF FITS data
  ├── Build param/type lists from config functions:
  │     lens_mass_config()  →  EPL + SHEAR
  │     lens_light_config() →  GAUSSIAN_ELLIPSE × N
  │     source_light_config() → PIXELATED (Matérn prior)
  ├── create_lens_image(...)       ← herculens LensImage
  └── create_prob_model(...)       ← NumPyro probabilistic model
```

---

## Stage 2 — Warm-start Initialization (`run_hmc` in `samplers.py`, L268)

```python
# Load SVI guide params pickle
guide_params = pickle.load("svi_guide_params.pkl")

# Recreate the AutoLowRankMultivariateNormal guide
guide = AutoLowRankMultivariateNormal(prob_model.model)

# Extract MAP medians in constrained (physical) space
init_params = guide.median(guide_params)    # → dict of physical params

# Convert to unconstrained space for NUTS
init_params_unconst = to_unconstrained(prob_model, init_params)
```

---

## Stage 3 — Parameter Grouping for Gibbs-within-HMC (L322)

Parameters are classified into blocks, each handled by a separate NUTS kernel:

| Group | Variables | Kernel |
|---|---|---|
| **Pixelated pixels** | `pixels_wn_*` | NUTS kernel 1 |
| **Matérn power spectrum** | `n_source_grid`, `rho_source_grid`, `sigma_source_grid` | NUTS kernel 1 |
| **Lens light** | `lens_light_*` | NUTS kernel 1 |
| **Other** | remaining | NUTS kernel 1 |
| **Lens mass** | `lens_*` (excl. `lens_light_*`) | NUTS kernel 2 |

---

## Stage 4 — Kernel Setup (L350–L405)

### Kernel 1 — Pixels + Power spectrum + Lens light
```python
kernel_1 = NUTS(
    prob_model.model,
    target_accept_prob=0.95,
    max_tree_depth=10,
    dense_mass=[ tuple(vars_power),                  # Matérn block
                 tuple(lens_light_params_per_comp) ]  # per-component blocks
)
```

### Kernel 2 — Lens mass
```python
kernel_2 = NUTS(
    prob_model.model,
    target_accept_prob=0.90,
    max_tree_depth=10,
    dense_mass=[ tuple(mass_params_per_component) ]
)
```

### Outer Gibbs wrapper
```python
outer_kernel = MultiHMCGibbs(
    inner_kernels=[kernel_1, kernel_2],
    gibbs_sites_list=[
        vars_pixel + vars_power + vars_lens_light + vars_other,  # → kernel_1
        vars_mass                                                  # → kernel_2
    ]
)
```

At each Gibbs step, kernel_1 updates its sites while conditioning on the current mass params (fixed), then kernel_2 updates its sites while conditioning on the current pixel/light params (fixed). This alternating update is implemented in [`custom_gibbs.py`](file:///users/xczhou/system/herculens_wrapper/herculens_wrapper/custom_gibbs.py).

---

## Stage 5 — Batched MCMC Execution with Checkpointing (L407–L637)

The total `num_samples` is split into `checkpoint_interval`-sized batches. From `config_hmc.py`:

| Setting | Value |
|---|---|
| `num_warmup` | 500 |
| `num_samples_total` | 3000 |
| `checkpoint_interval` | 500 |
| `num_chains` | 4 |

```
Batch 0: MCMC(..., num_warmup=500, num_samples=500) ← full warmup + 500 samples
  └─ saves hmc_samples_batch_0.npz + hmc_checkpoint.pkl

Batch 1: MCMC(..., num_warmup=0, num_samples=500)   ← resumes from last state
  └─ saves hmc_samples_batch_1.npz + hmc_checkpoint.pkl

... (6 batches total for 3000 samples at 500/batch)
```

If interrupted, the run **resumes automatically** from the checkpoint file on the next invocation.

---

## Stage 6 — Intermediate Diagnostics (per batch)

After each batch, the following are generated in `diagnostics/`:

| File | Description |
|---|---|
| `image_plane_batch_N.png` | Lens plane visualization using current median params |
| `best_fit_model_linear_batch_N.png` | Data / model / residuals (linear scale) |
| `best_fit_model_log_batch_N.png` | Data / model / residuals (log scale) |
| `source_plane_linear_batch_N.png` | Reconstructed source (linear) |
| `source_plane_log_batch_N.png` | Reconstructed source (log) |
| `mcmc_summary_batch_N.txt` | ArviZ `az.summary()` — R̂, ESS, mean, sd for lens mass + Matérn params |
| `mcmc_diagnostics_batch_N.png` | ArviZ trace + density plots |

---

## Stage 7 — Post-sampling Outputs

```
Concatenate all batch samples
  ↓
map_params = median over all samples (parameter point estimate)
  ↓
Save:
  hmc_samples.npz          ← full sample array
  kwargs_result.json        ← MAP (median) best-fit params
  kwargs_sigma.json         ← asymmetric 1σ uncertainties [p16, p84]
  metrics.json              ← chi², reduced chi², BIC, log-likelihood
  mcmc_summary_final.txt    ← final ArviZ convergence summary
  mcmc_diagnostics_final.png← final trace plots
  modeling_result.npz       ← best_fit_model, image_data, noise_map
```

---

## Pipeline Summary (data flow)

```
config_hmc.py
    │
    ▼
run.py: build_and_run()
    │  ← FITS data, lens model components, prob_model
    ▼
samplers.py: run_hmc()
    │
    ├─ [Stage 2] Load svi_guide_params.pkl → warm-start init_params
    │
    ├─ [Stage 3] Group params: pixels / power / lens_light / lens_mass
    │
    ├─ [Stage 4] Build NUTS kernel_1 + kernel_2
    │             wrapped in MultiHMCGibbs (custom_gibbs.py)
    │
    ├─ [Stage 5] Batched MCMC loop with checkpoint/resume
    │             (warmup only on batch 0)
    │
    ├─ [Stage 6] Per-batch diagnostics → diagnostics/
    │
    └─ [Stage 7] Concatenate, compute MAP median, save outputs
```
