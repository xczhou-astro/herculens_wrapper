# Herculens Wrapper API

本文只介绍公开 API 的方法和用法。所有公开对象均从 `herculens_wrapper.api` 导入。

## 交互式点源 ray-tracing viewer

仓库包含一个独立的本地界面，用于从已有质量模型检查点源及其扩展
Gaussian 源的像。启动后选择观测图像和包含 `kwargs_result.json` 的拟合目录：

```bash
python point_source_gui/start_gui.py
```

页面会优先从结果目录附近的 `model_configuration.json` 或 `config.json`
读取 `lens_mass_type_list`。如果该配置文件不在目录中，在 **Mass types**
字段按拟合时相同的顺序填入类型，例如 `EPL, SHEAR`。设置拟合使用的
pixel scale 后，点击左侧 image-plane 图中的点：页面会显示该点的
source-plane 坐标和 caustic，并给出以该位置为中心、可调 `sigma` 的 Gaussian
在 image plane 中形成的 1σ、2σ、3σ 等值线。图像必须与拟合时使用的
中心、裁切和 pixel scale 相同。

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

### 联合采样 lens light 与 stellar mass MGE

`StellarMassMGE` 默认读取固定的 lens-light MGE，适合将先前结果用于
后续阶段。若希望在 parametric 或 pixelated SVI/HMC 中让 stellar mass 随
同一组 sampled lens-light Gaussian 参数更新，使用 `follow_lens_light=True`：

```python
from herculens_wrapper.api import LightProfile, StellarMassMGE

lens_light = LightProfile.mge(15, (0.001, 3.0))
stellar = StellarMassMGE(
    lens_light,
    follow_lens_light=True,
    prior={
        "upsilon_kappa": [0.1, 0.05, 0.001, 1.0],
        "ml_gradient": [0.0, 0.2, -1.0, 1.0],
    },
)
profiles = LensProfileCollection(lens_mass=[stellar], lens_light=lens_light)
```

每次 likelihood evaluation 都会从当前 lens-light MGE 重建 stellar Gaussian
amplitudes；`upsilon_kappa` 则设定全局 stellar-mass normalization。若在后续
阶段希望固定 light 但保留这条 dependency，在写入或载入 lens-light 参数值后使用
`profiles.with_fixed(lens_light=True)`。

### 标准椭圆 NFW halo

`NFWEllipseHalo` 对应 `NFW_ELLIPSE_KAPPA`。其 3-D density 是标准 NFW
`rho ∝ 1 / (x * (1 + x)**2)`；椭圆化的 projected convergence 由
JAX-Lensing-Profiles 的 3-D MGE 计算。`R_s`（注意大写）是角尺度半径，
而 `kappa_s` 是无量纲 normalization：

```python
from herculens_wrapper.api import NFWEllipseHalo

halo = NFWEllipseHalo(prior={
    "kappa_s": [0.03, 0.02, 1e-4, 0.5],
    "R_s": [3.0, 2.0, 0.3, 30.0],
    "e1": [0.06, 0.06, -0.3, 0.3],
    "e2": [0.02, 0.06, -0.3, 0.3],
    "center_x": [0.0, 0.03, -0.1, 0.1],
    "center_y": [0.0, 0.03, -0.1, 0.1],
})
```

这里不要把 `e1=e2=0` 固定：MGE 椭圆 Gaussian backend 在严格圆极限没有
数值 branch。若 halo 必须严格球对称，改用解析的 `MassProfile("NFW")`。
`GNFWHaloMGE` 是另一个、具有不同 outer-transition 定义的 profile，不能通过
固定其参数来替代这个标准 NFW。

若 lens light 是一组 `GAUSSIAN_ELLIPSE`，可以让 halo 中心等于整组 MGE 的
光通量加权中心（`amp` 是各 Gaussian 的积分通量）：

```python
halo.center_x.prior = ["correlated", "lens_light", "flux_centroid", "center_x"]
halo.center_y.prior = ["correlated", "lens_light", "flux_centroid", "center_y"]
```

计算使用当前采样值：`center_x = sum(amp_i * center_x_i) / sum(amp_i)`，
`center_y` 同理。这个链接不额外采样 halo 中心；如果 lens light 在后续阶段
固定，halo 中心也会随之固定。此链接要求 `lens_light` 全部由
`GAUSSIAN_ELLIPSE` 组成。

### EPL-referenced multipole phase

`MPPL_OFFSET` samples a relative multipole phase while keeping it close to an
EPL orientation.  All public-facing angles are in degrees.  `phi_ref` is an
exact link and therefore does not create a second sampled PA:

```python
epl = MassProfile(
    "EPL",
    prior={
        "theta_E": [0.5, 2.0], "gamma": [1.6, 2.4],
        "q": [0.3, 0.9], "phi": [-90.0, 90.0],
        "center_x": [-0.2, 0.2], "center_y": [-0.2, 0.2],
    },
)
mppl4 = MassProfile(
    "MPPL_OFFSET",
    prior={
        "m": 4,
        "a_m": [0.0, 0.03],
        "delta_phi_m": [0.0, 10.0, -20.0, 20.0],
    },
)
mppl4.phi_ref = epl.phi
mppl4.gamma = epl.gamma
mppl4.center_x = epl.center_x
mppl4.center_y = epl.center_y
mppl4.b = epl.theta_E

profiles = LensProfileCollection(lens_mass=[epl, mppl4])
```

At every likelihood evaluation, the profile uses
`phi_m = phi_ref + delta_phi_m`.  For `m=4`, use a narrow offset prior around
zero to avoid the 90-degree phase periodicity.  The same
`mppl4.phi_ref = epl.phi` link also works when the EPL is declared with
native `e1`/`e2`: in that case `phi` is derived internally as
`0.5 * atan2(e2, e1)` and is used only for the link.

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

### 均匀 exposure time 的 Poisson 修正

若图像和模型的单位都是电子计数率（例如 `e-/s`），可不用固定
`noise` map，而是给定背景 RMS（同为 `e-/s`）和统一曝光时间（秒）：

```python
data = SingleBandData.from_fits(
    "image.fits", None, "psf.fits",
    pixel_scale=0.03,
    background_rms=0.012,  # e-/s
    exposure_time=1200.0,  # s
)
```

似然采用模型依赖的 Gaussian--Poisson 方差
`background_rms**2 + maximum(model, 0) / exposure_time`。`noise` map 与
`background_rms`/`exposure_time` 两种输入互斥；原有的 `noise` map 接口完全保留。
当前只支持标量、空间均匀的 exposure time，暂不支持 variance boost map。

### 采样 background RMS

若背景 Gaussian RMS 本身不确定，可令它成为一个全图共享的 SVI/HMC
参数。此模式仍需要已知的曝光时间：

```python
data = SingleBandData.from_fits(
    "image.fits", None, "psf.fits",
    pixel_scale=0.03,
    exposure_time=1200.0,
    background_rms_prior={
        "kind": "log_uniform",
        "low": 5e-5,
        "high": 2e-4,
    },
)
```

简写 `background_rms_prior=[5e-5, 2e-4]` 等价于上述 LogUniform prior。

SVI/HMC 的 `kwargs_result.json` 会在 `likelihood_parameters.background_rms`
记录该参数的后验代表值；将该结果作为 `initialize(init_params_path=...)`
的输入时会自动恢复，供 HMC 初始状态及其回退路径使用。
结果参数中会包含 `background_rms`。它与固定 `noise` map、固定
`background_rms` 两种输入均互斥；目前仅支持 `SingleBandModel`。

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

### 从多个已有结果 warm-start，但继续采样

`warm_start_from()` 只提供数值初值；它**不会**修改当前 profile 的
prior，也不会固定参数。文件可为 `kwargs_result.json`，或包含它的 run
directory。实际读取在 `model.initialize()`（以及未传 `init_params` 的
`model.run()`）时发生。

这适合把独立的 lens-light 拟合与 ring/arc 拟合拼成最终联合模型：

```python
from herculens_wrapper.api import (
    LensProfileCollection, ProfileCollection,
    LightProfile, MassProfile, PixelatedLensLight, PixelatedSource,
)

# 两个 lens-light 成分共享一个保存结果；顺序必须与原拟合一致。
lens_light = ProfileCollection([
    LightProfile("SERSIC_ELLIPSE", prior=bulge_prior),
    PixelatedLensLight(scale_factor=lens_scale, matern_prior=disk_prior),
]).warm_start_from(
    "lens_light_pixelated_svi/run_0", component="lens_light",
)

# 可分别来自另一轮 ring/arc 的结果。
lens_mass = MassProfile("SIE", prior=mass_prior).warm_start_from(
    "ring_pixelated_hmc", component="lens_mass",
)
source_light = PixelatedSource(
    pixel_grid=source_grid, pixelated_prior=source_prior,
).warm_start_from("ring_pixelated_hmc", component="source_light")

profiles = LensProfileCollection(
    lens_mass=lens_mass,
    lens_light=lens_light,
    source_light=source_light,
)
```

`ProfileCollection(...).warm_start_from(...)` 是多成分（例如 parametric +
pixelated lens light）的推荐写法。单一成分可直接调用
`profile.warm_start_from(...)`。所有当前 profile 的顺序和 pixel grid 必须与
各自保存结果一致；否则 `initialize()` 会给出具体 latent-site shape 错误。
使用 `clear_warm_start()` 可撤销声明。

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

### JAXtronomy EPL plus elliptical multipoles (`EPL_MULTIPOLE_M1M3M4_ELL`)

`EPL_MULTIPOLE_M1M3M4_ELL` is the direct JAXtronomy-style combination of an
EPL and its elliptical `m=1`, `m=3`, and `m=4` perturbations.  It is one
mass component; do not add a separate EPL or link geometry to other terms:

```python
epl = MassProfile("EPL_MULTIPOLE_M1M3M4_ELL", prior={
    "theta_E": [0.5, 2.0], "gamma": [1.6, 2.4],
    "q": [0.3, 0.9], "phi": [-90.0, 90.0],
    "center_x": [-0.2, 0.2], "center_y": [-0.2, 0.2],
    "a1_a": [-0.02, 0.02], "delta_phi_m1": [-15.0, 15.0],
    "a3_a": [-0.02, 0.02], "delta_phi_m3": [-15.0, 15.0],
    "a4_a": [-0.02, 0.02], "delta_phi_m4": [-15.0, 15.0],
})
profiles = LensProfileCollection(lens_mass=[epl])
```

All delta angles in the API are degrees and are eccentric anomalies measured
from the EPL semi-major axis, not sky position angles.  Each dimensionless
amplitude becomes the physical elliptical-multipole amplitude
`a_m = a*_a * theta_E` internally.

`mass_profile_convergence.png` automatically expands this joint profile into
one total-model row and separate `EPL`, `m=1`, `m=3`, and `m=4` rows.  Each
row contains a convergence map, a magnification map, and a radial convergence
profile.  The convergence maps add exactly to the joint profile.  The
multipole magnification panels are their *isolated* lensing responses; only
the total-model row has physical critical lines and the full-model
magnification.

The same diagnostic also expands independently declared mass components such
as `EPL + MPPL(m=1) + MPPL(m=3)`.  A `SHEAR` component remains only in the
total-model row: its convergence is identically zero, while its physically
meaningful effect is already present in the total critical lines and
magnification.

### SIE elliptical multipole with an offset phase (`ELL_MPPL_OFFSET`)

`ELL_MPPL_OFFSET` is a standalone elliptical multipole for an **SIE**
reference lens.  It deliberately has no `gamma`: link its reference geometry
to a companion `SIE`, whose slope is fixed to 2.  `delta_varphi_m` and
`phi_ref` are degrees in the API; the former is an eccentric anomaly from the
reference ellipse semi-major axis, not a sky PA.

```python
sie = MassProfile("SIE", prior={
    "theta_E": [0.5, 2.0],
    "q": [0.3, 0.9], "phi": [-90.0, 90.0],
    "center_x": [-0.2, 0.2], "center_y": [-0.2, 0.2],
})
ell_m4 = MassProfile("ELL_MPPL_OFFSET", prior={
    "m": 4,
    "a_m_frac": [-0.05, 0.05],
    "delta_varphi_m": [0.0, 10.0, -20.0, 20.0],
})
ell_m4.q = sie.q
ell_m4.phi_ref = sie.phi
ell_m4.center_x = sie.center_x
ell_m4.center_y = sie.center_y
ell_m4.r_E = sie.theta_E

profiles = LensProfileCollection(lens_mass=[sie, ell_m4])
```

Only `m=1`, `m=3`, and `m=4` are supported.  Supply exactly one of `a_m`
(arcsec) or `a_m_frac` (dimensionless).  Do not add `gamma`; the profile is
not a generalized-EPL elliptical multipole.

### Legacy note

`ELL_MPPL_OFFSET` supersedes the earlier `ELL_MPPL` public name.  The older
`ELL_MPPL` and `EPL_M1M3M4` public profile names are not registered.

`EPL_MULTIPOLE_M3M4_ELL` uses the same JAXtronomy elliptical construction but
omits m1.  It takes the same parameters except `a1_a` and `delta_phi_m1`:
`theta_E`, `gamma`, `e1`, `e2`, `center_x`, `center_y`, `a3_a`,
`delta_phi_m3`, `a4_a`, and `delta_phi_m4`.

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

### Parametric + pixelated light

Lens light 和 source light 都可以同时包含 parametric profile 与一个
pixelated profile；各分量的亮度会相加：

```python
from herculens_wrapper.api import PixelatedLensLight, PixelatedSource

profiles = LensProfileCollection(
    lens_mass=lens_mass,
    lens_light=[
        lens_bulge,  # 例如 LightProfile("SERSIC_ELLIPSE", ...)
        PixelatedLensLight(
            scale_factor=1.0,
            matern_prior={"positive": True},
        ),
    ],
    source_light=[
        source_core,  # 例如 LightProfile("SERSIC_ELLIPSE", ...)
        PixelatedSource(
            pixel_grid={"pixel_grid_shape": 80},
            pixelated_prior={"prior_type": "matern", "positive": True},
        ),
    ],
)
```

每个 light block 目前最多包含一个 `PIXELATED` profile，但可以同时包含任意数量的
parametric profiles，且 `PIXELATED` 可以位于列表中的任意位置。RTU source grid 目前仍要求
source light 中只有 `PixelatedSource`；uniform/adaptive pixel grid 支持上述混合模型。

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

# 跨波段：仅以 F277W 的 mass 结果 warm-start F150W。
# mass 仍是 F150W SVI 的自由参数；不会继承 F277W source/light。
initial = model.initialize(
    seed=42,
    init_lens_mass_path="F277W/pixelated_svi/run_0",
    pixelated_init_match="image",
    num_iterations_warmup=2000,
)

# 同一件事也可直接交给 run（包括 n_runs>1 的独立 SVI restarts）。
result = model.run(
    SamplerConfig.svi(max_iterations=5000),
    save_path="F150W/pixelated_svi",
    init_lens_mass_path="F277W/pixelated_svi/run_0",
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

`result.output()` 写出的 `modeling_result.fits` 包含 `BEST_FIT_MODEL`、
`IMAGE_DATA`、`NOISE_MAP`、`PSF`、`LENS_LIGHT` 和 `SOURCE_PLANE` 扩展，
以及已有的掩膜。`PSF` 保留输入核，`PSFSSAMP` 记录其相对图像像素的超采样倍数。
`SOURCE_PLANE` 的物理坐标保存在 `SOURCE_X`/`SOURCE_Y`；RTU 网格则使用
`SOURCE_X_CORNERS`/`SOURCE_Y_CORNERS`。源平面数值以图像像素通量为单位。

`SUMMARY` 头关键字注明图像汇总方式：SVI 的 `GUIDEMED` 表示在 guide 的
中值参数处计算图像；HMC 的 `PIXMED` 表示对每次后验抽样生成的图像逐像素取
中值。HMC 的 `BEST_FIT_MODEL`、`LENS_LIGHT` 和 `SOURCE_PLANE` 都采用后者，
它们不是最大 log likelihood 样本的图像。

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

任何 mass 参数都可用同一方法设为某个波段独立；未声明的其它波段仍共享：

```python
lens_mass.set_independent("F150W", "gamma", [1.8, 2.4])
```

### 从已有质量模型开始 pixelated SVI

`init_lens_mass_path` 只读取已有结果的 `kwargs_lens`，并不固定它们，
因此新联合拟合仍会重新采样全部质量参数。它可接受 single-band 或
multi-band result directory：

```python
result = model.run(
    SamplerConfig.svi(max_iterations=10_000, random_seed=42),
    save_path="F150W_F277W/pixelated_svi",
    init_lens_mass_path="F277W/parametric_svi/run_0",
    pixelated_init_match="image",
    num_iterations_warmup=2_000,
)
```

若初始化目录是联合 parametric SVI，pixelated source 还可用每个 band 的
analytic source 进行 power-spectrum 初始化：

```python
initial = model.initialize(
    init_params_path="multiband/parametric_svi/run_0",
    pixelated_init_match="source",
    num_iterations_warmup=2_000,
)
```

多次独立的 joint SVI 可直接运行：

```python
results = model.run(
    SamplerConfig.svi(random_seed=42),
    save_path="multiband/svi_restarts",
    n_runs=4,
)
results.output("multiband/svi_restarts")
```

每个 band 也可以在 `SingleBandData` 中给出自己的
`background_rms_prior` 和 `exposure_time`；这会采样独立的
`F150W/background_rms`、`F277W/background_rms` likelihood nuisance 参数。

## 12. 多次结果比较

```python
from herculens_wrapper.api import (
    MultiBandResultsCombination,
    SingleBandResultsCombination,
)

SingleBandResultsCombination(single_band_results).output("svi")
MultiBandResultsCombination(multiband_results).output("multiband_svi")
```
