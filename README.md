# Herculens Wrapper API

本文只介绍公开 API 的方法和用法。所有公开对象均从 `herculens_wrapper.api` 导入。

## 1. 基本流程

```python
from herculens_wrapper.api import (
    LensProfileCollection,
    LightProfile,
    MassProfile,
    SamplerConfig,
    SingleBandData,
    SingleBandModel,
)

# 1. 读取数据
data = SingleBandData.from_fits(
    "data.fits",
    "noise.fits",
    "psf.fits",
    pixel_scale=0.05,
)

# 2. 定义 profiles
lens_mass = MassProfile(
    "EPL",
    prior={
        "theta_E": [1.0, 0.2, 0.2, 2.5],
        "gamma": [2.0, 0.1, 1.5, 2.5],
        "q": [0.6, 0.9],
        "phi": [80.0, 100.0],
        "center_x": [0.0, 0.1, -0.5, 0.5],
        "center_y": [0.0, 0.1, -0.5, 0.5],
    },
)

source = LightProfile(
    "SERSIC_ELLIPSE",
    prior={
        "amp": [1.0, 0.5, 0.0, 10.0],
        "R_sersic": [0.2, 0.1, 0.01, 2.0],
        "n_sersic": [2.0, 0.5, 0.5, 8.0],
        "e1": [0.0, 0.1, -0.5, 0.5],
        "e2": [0.0, 0.1, -0.5, 0.5],
        "center_x": [0.0, 0.1, -0.5, 0.5],
        "center_y": [0.0, 0.1, -0.5, 0.5],
    },
)

profiles = LensProfileCollection(
    lens_mass=lens_mass,
    source_light=source,
)

# 3. 创建 model
model = SingleBandModel(
    observation=data,
    profiles=profiles,
    numerics={"supersampling_factor": 1},
)

# 4. 初始化并选择 sampler
initial = model.initialize(seed=42)
sampler = SamplerConfig.svi(
    max_iterations=5000,
    learning_rate=1e-2,
    random_seed=42,
)

# 5. 采样并输出结果
result = model.run(
    sampler,
    init_params=initial,
    save_path="results/run_0",
)
result.output()
```

参数使用四元列表 `[mean, sigma, lower, upper]` 表示采样 prior；标量表示固定参数。

## 2. 数据 API

### 从独立 FITS 文件读取

```python
data = SingleBandData.from_fits(
    image_path="data.fits",
    noise_path="noise.fits",
    psf_path="psf.fits",
    pixel_scale=0.05,
    psf_supersampling_factor=1,
    crop_size=120,
    background_subtract={"num_pixels": 10, "corner": "upper left"},
    source_arc_mask_path="mask_1.fits",
    contaminate_mask_path="mask_out.fits",
)
```

### 从一个多 HDU FITS 文件读取

```python
data = SingleBandData.from_fits_hdus(
    "observation.fits",
    pixel_scale=0.05,
    image_hdu=0,
    noise_hdu="NOISE",
    psf_hdu="PSF",
)
```

### 直接使用数组

```python
data = SingleBandData(
    image=image_array,
    noise=noise_array,
    psf=psf_array,
    pixel_scale=0.05,
)
```

### 常用数据方法

```python
data.show(scale="linear", save_path="input.png")
data.save("results/data")

mask = data.set_source_arc_mask_from_snr(
    threshold=2.0,
    smoothing_sigma_pixels=10.0,
    dilate_pixels=2,
    save_path="source_mask.fits",
)

image = data.likelihood_image
noise = data.likelihood_noise
mask = data.likelihood_mask
```

## 3. Profile API

### Mass 和 light profiles

```python
mass = MassProfile("SIE", prior={...})
light = LightProfile("SERSIC_ELLIPSE", prior={...})

# 一次定义多个 profile
mass_components = MassProfile(
    ["EPL", "SHEAR"],
    prior=[epl_prior, shear_prior],
)
```

### EPL/SIE 的轴比和方向

EPL 和 SIE 推荐直接使用 `q` 和 `phi`：

```python
mass = MassProfile(
    "EPL",
    prior={
        "theta_E": [0.2, 0.8],
        "gamma": [1.2, 2.8],
        "q": [0.6, 0.9],
        "phi": [80.0, 100.0],
        "center_x": [0.0, 0.1, -0.3, 0.3],
        "center_y": [0.0, 0.1, -0.3, 0.3],
    },
)
```

`q` 是短轴/长轴比，必须满足 `0 < q <= 1`。`phi` 的单位是度，
从模型坐标的 `+x` 轴逆时针测量；因此 `phi=90` 表示长轴沿 `+y` 方向。
模型内部自动转换为 Herculens 使用的 `e1/e2`。旧的 `e1/e2` 输入仍然兼容，
但同一个 profile 不能同时提供两套参数。

### 修改 prior 或当前值

```python
mass.set_prior(theta_E=[1.0, 0.2, 0.2, 2.5])
mass.set_value(theta_E=1.05)

mass.theta_E.prior = [1.0, 0.2, 0.2, 2.5]
mass.theta_E.value = 1.05

print(mass.priors)
print(mass.values)
```

### 参数链接

把一个参数赋值为另一个 `Parameter` 即可建立链接：

```python
source.center_x = mass.center_x
source.center_y = mass.center_y

# 等价写法
source.center_x.link_to(mass.center_x)
```

被链接参数不作为独立采样参数。

### 从已有结果固定初始化

```python
mass.initialize_from(
    "previous_run/kwargs_result.json",
    component="lens_mass",
)

mass.clear_initialization()
```

### 固定 profile 或部分参数

```python
fixed_all = profiles.freeze()

fixed_light = profiles.with_fixed(lens_light=True)

fixed_centers = profiles.with_fixed(
    lens_mass={0: ["center_x", "center_y"]},
)

fixed_values = profiles.with_fixed(
    lens_mass={0: {"center_x": 0.0, "center_y": 0.0}},
)
```

### 查看完整 profile 配置

```python
print(profiles.configuration)
print(profiles.priors)
print(profiles.values)
```

## 4. MPPL multipole

推荐直接输入 `a_m` 和 `phi_m`。模型使用
`cos[m(theta - phi_m)]`：API 中的 `phi_m` 从模型坐标 `+x` 轴逆时针测量，
单位为度，等价方向的周期为 `360°/m`。进入 MPPL 内部计算前会自动转换为弧度。
`m` 必须是固定正整数。两元素列表
`[low, high]` 表示该区间上的均匀 prior。

```python
epl = MassProfile("EPL", prior=epl_prior)

multipole = MassProfile(
    "MPPL",
    prior={
        "m": 4,
        "a_m": [0.02, 0.01, 0.0, 0.10],
        "phi_m": [0.0, 10.0, -45.0, 45.0],
    },
)

# MPPL 与主 EPL 共用几何/斜率参数
multipole.gamma = epl.gamma
multipole.center_x = epl.center_x
multipole.center_y = epl.center_y
multipole.b = epl.theta_E

profiles = LensProfileCollection(
    lens_mass=[epl, multipole],
    source_light=source,
)
```

例如，对 `m=1` 将方向限制在 80°–100°：

```python
"phi_m": [80.0, 100.0]
```

如果需要把方向固定为单一角度，应使用标量：

```python
"phi_m": 90.0
```

也支持等价的 `e_x`、`e_y` 输入，但不能与 `a_m`、`phi_m` 同时使用：

```python
multipole = MassProfile(
    "MPPL",
    prior={
        "m": 4,
        "e_x": [0.01, 0.01, -0.1, 0.1],
        "e_y": [0.01, 0.01, -0.1, 0.1],
    },
)
```

## 5. Pixelated source

```python
from herculens_wrapper.api import PixelatedSource

source = PixelatedSource(
    pixel_grid={
        "grid_kind": "uniform",  # 或 ray_transformed_uniform
        "pixel_grid_shape": 80,
        "pixel_interpol": "fast_bilinear",
        "pixel_scale_factor": 0.5,
        "grid_center": (0.0, 0.0),
        "grid_shape": (2.0, 2.0),
        "rtu_polynomial_order": 11,
    },
    pixelated_prior={
        "prior_type": "matern",
        "regul_strengths": (3.0, 3.0),
        "positive": True,
    },
)
```

`prior_type` 支持 `matern`、`wavelet_sparsity` 和 `wavelet_penalty`。

## 6. SingleBandModel

### 创建和查看模型

```python
model = SingleBandModel(
    profiles=profiles,
    observation=data,
    numerics={
        "supersampling_factor": 1,
        "supersampling_convolution": False,
    },
    source_grid_scale=1.0,
    likelihood_scale=1.0,
)

print(model.configuration())
print(model.describe())
print(model.num_sampling_parameters)
```

### 初始化

```python
initial = model.initialize(seed=42)

# 从以前的 SVI 结果初始化
initial = model.initialize(
    seed=42,
    init_params_path="previous_svi/run_0",
    pixelated_init_match="image",
    num_iterations_warmup=2000,
)
```

### 初始模型图

```python
model.plot_initial_model(
    scale="linear",
    save_path="initial_model.png",
)

model.plot_initial_source(
    scale="linear",
    save_path="initial_source.png",
)
```

### 载入已有结果

```python
model.load("previous_svi/run_0", seed=42)
result = model.get_results(random_seed=42)
```

如果给出包含多个 `run_i` 的目录，`load()` 会选择最高 likelihood 的 run。

## 7. SVI

### 单次 SVI

```python
sampler = SamplerConfig.svi(
    max_iterations=5000,
    learning_rate=1e-2,
    init_scale=0.1,
    loss_kind="trace_elbo",
    num_particles=10,
    random_seed=42,
)

result = model.run(
    sampler,
    init_params=initial,
    save_path="svi/run_0",
)
result.output()
```

### 多次串行 SVI

```python
results = model.run(
    sampler,
    save_path="svi",
    n_runs=4,
    parallel=False,
)
results.output("svi")
```

### 多 GPU/MIG 并行 SVI

```python
results = model.run(
    sampler,
    save_path="svi",
    n_runs=4,
    parallel=True,
    gpus=["0", "1"],
    # MIG 也可写为 ["MIG-...", "MIG-..."]
)
results.output("svi")
```

每个 restart 保存到 `svi/run_i`。

## 8. HMC/NUTS

HMC 需要先从已有 SVI 结果初始化。

```python
initial = model.initialize(
    seed=42,
    init_params_path="svi/run_0",
    num_iterations_warmup=0,
)

hmc = SamplerConfig.hmc(
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    checkpoint_interval=250,
    chain_method="parallel",
    progress_bar=True,
    disable_gibbs=False,
    random_seed=42,
)

result = model.run(
    hmc,
    init_params=initial,
    save_path="hmc",
)
result.output()
```

多 GPU/MIG HMC 使用 `chain_method="parallel"`，每条 chain 占用一个可见 JAX device。启动程序前设置可见设备：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py
```

### 恢复中断的 HMC

```python
hmc = SamplerConfig.hmc(
    num_samples=2000,  # 恢复后希望达到的总样本数/chain
    num_chains=4,
    checkpoint_interval=250,
    chain_method="parallel",
)

result = model.resume_hmc(hmc, save_path="hmc")
result.output()
```

### 载入 HMC posterior

```python
model.load_hmc("hmc")
result = model.get_results(random_seed=42)

metrics = model.recompute_hmc_metrics("hmc", write=True)
```

## 9. Result API

### 路径和标准输出

```python
print(result.run_directory)
print(result.log_path)

# 沿用 model.run(save_path=...) 的目录
products = result.output()

# 保留旧接口：显式指定其他输出目录
products = result.output("other_output")
```

### 数值结果

```python
parameters = result.parameters
samples = result.samples       # HMC posterior；SVI 通常为 None
loss = result.loss_history
metrics = result.metrics()
source = result.get_source_plane()
convergence = result.mass_component_convergence()
```

### 保存和绘图

```python
result.save_parameters("parameters.json")
result.save_metrics("metrics.json")
result.save_history("loss.json")
result.save_svi_guide("guide.pkl")

result.plot_best_fit(save_path="best_fit.png")
result.plot_loss_curve(save_path="loss.png")
result.plot_image_plane(save_path="image_plane.png")
result.plot_composite(save_path="composite.png")
result.plot_source_plane(save_path="source_plane.png")
result.plot_corner(save_path="corner.png")
result.plot_mass_profile_convergence(save_path="convergence.png")
```

## 10. 自动 logging

不需要在 run file 中导入或配置 logging：

```python
result = model.run(sampler, save_path="results/run_0")
result.output()

print(model.log_path)   # results/run_0/log.txt
print(result.log_path)  # results/run_0/log.txt
```

`model.run()` 自动写入：

- `log.txt`
- `model_configuration.json`

串行运行同时输出到终端和日志；并行 SVI 的每个 worker 独立写入 `run_i/log.txt`。

## 11. 多波段 API

```python
from herculens_wrapper.api import (
    LensProfileCollection,
    MultiBandData,
    MultiBandModel,
    MultiBandProfileCollection,
)

observations = MultiBandData(
    F150W=data_f150w,
    F277W=data_f277w,
)

shared = LensProfileCollection(lens_mass=lens_mass)

band_profiles = {
    "F150W": LensProfileCollection(
        lens_light=lens_light_f150w,
        source_light=source_f150w,
    ),
    "F277W": LensProfileCollection(
        lens_light=lens_light_f277w,
        source_light=source_f277w,
    ),
}

profiles = MultiBandProfileCollection(
    shared=shared,
    bands=band_profiles,
)

model = MultiBandModel(
    observations=observations,
    profiles=profiles,
    numerics={"supersampling_factor": 1},
)

initial = model.initialize(seed=42)
result = model.run(
    SamplerConfig.svi(random_seed=42),
    init_params=initial,
    save_path="multiband/run_0",
)
result.output()

print(model.configuration())
print(model.describe())
print(result.kwargs_by_band())
print(result.metrics())
```

### 波段独立的 lens center

```python
lens_mass.set_independent("F150W", "center_x", [-0.05, 0.02, -0.2, 0.2])
lens_mass.set_independent("F150W", "center_y", [0.03, 0.02, -0.2, 0.2])
```

## 12. 多次结果比较

```python
from herculens_wrapper.api import (
    MultiBandResultsCombination,
    SingleBandResultsCombination,
)

SingleBandResultsCombination(single_band_results).output("svi")
MultiBandResultsCombination(multiband_results).output("multiband_svi")
```
