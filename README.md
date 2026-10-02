# Herculens Wrapper API

本文只介绍公开 API 的方法和用法。所有公开对象均从 `herculens_wrapper.api` 导入。

## Sky Workbench：从 FITS 大图生成 STARRED PSF

本地网页按「载入 sky 图像 → 查找点源候选 → 人工选择 → STARRED 拟合 PSF」
的顺序工作。PSF 优化步骤参考 `coadd/psf_modelling.py`。启动：

```bash
python -m pip install -r sky_workbench/requirements.txt
python -m sky_workbench.app
```

打开 <http://127.0.0.1:5073>，选择本机 FITS（默认识别 `SCI` 和 `ERR` HDU）。
图像使用与 `coadd/starfinder.py` 相同的 `viridis` 对数配色，并保持原始像素比例。
滚轮缩放、拖动平移，在当前视野查找候选，点击选用或取消点源。
设置奇数像素的输出 PSF 尺寸并选择输出文件夹后运行 STARRED。拟合切片比
输出 PSF 每边多 10 像素，输出从中心裁取，仅保存与原图相同像素尺度的
`psf_modelled.fits`。同时保存 `selected_stars.csv`、`selection.json` 和
`starred.log`。拟合时为每颗星估计并拟合局部常数背景，读取可用的
`COVERAGE_MASK` / `DQ`，屏蔽附近明显点源。页面显示 PSF 和观测、模型、
`残差 / ERR` 对比图；输出目录还有 `quality.json`、`quality.png`、
`star_residuals.fits` 和 `radial_profiles.csv`。选中至少 3 颗星时，先留出
最后一颗做独立验证，再用全部星拟合最终 PSF，并保存 `holdout_quality.png`。
边缘点源的缩略图可显示缺失区域；输出 PSF 覆盖的区域仍须完全落在原图内。

`coadd/coadd.py` 使用 WCS 双线性重采样和逆方差合并，并未调用 AstroDrizzle。
重采样仍会产生相邻像素间的噪声相关性；`ERR` 只描述逐像素标准差，因此
质量图中的 `残差 / ERR` 是诊断量，不能单独当作严格的独立像素卡方检验。

STARRED 需要 `starred-astro` 和 JAX。程序优先使用当前 Python；也会查找本机
Conda 的 `utils` 或 `herculens` 环境。可用 `STARRED_PYTHON=/path/to/python`
指定拟合所用的解释器。网页默认只监听 `127.0.0.1`。

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

### 独立的批量质量 MGE

`MassProfile.mge()` 返回一个 `MASS_MGE` profile，用数组批量计算 Gaussian
偏折，与 lens light 没有参数链接。Gaussian 数目在构建模型时指定；改变数目
需要重建模型和重新编译。

```python
mass_mge = MassProfile.mge(
    n_gauss=10,
    sigma_lims=(0.5 * args.pixel_scale,
                0.5 * args.pixel_scale * args.crop_size),
    amp_prior=(-3.0, 1.0),
    shared_center=True,
    shared_ellipticity=True,
    center_prior=(0.0, 0.1, -0.2, 0.2),
    ellipticity_prior=(0.0, 0.1, -0.5, 0.5),
)
profiles = LensProfileCollection(
    lens_mass=[mass_mge, shear], lens_light=lens_light, source_light=source_light,
)
```

`amp_prior=(log_loc, log_scale)` 与 `LightProfile.mge` 使用同样的 LogNormal
约定：每个 Gaussian 独立满足 `ln(amp_i) ~ Normal(log_loc, log_scale)`，
使用自然对数。例如 `(-3, 1)` 的中位数约为 `0.05`。
质量 Gaussian 的振幅是积分 convergence，坐标为 arcsec 时单位为 arcsec²；
因此数值需按质量尺度设置，不能直接照搬 lens-light 的 flux 振幅。
`sigma_lims` 被分为 N 个非重叠对数区间，各宽度在自己的区间内按 LogUniform 采样。
中心和椭率使用 `[mean, std, lower, upper]` TruncatedNormal 先验，分别应用于
`center_x/center_y` 和 `e1/e2`。

两个共享开关独立：共享时每个坐标/椭率分量只采样一个标量；不共享时采样 N
个独立值，并且必须显式提供相应的 `center_prior` 或 `ellipticity_prior`。
共享且未提供先验时，默认分别为 `(0, 0.1, -0.2, 0.2)` 和 `(0, 0.1, -0.5, 0.5)`。
两者均共享时有 `2*N+4` 个质量参数，均不共享时有 `6*N` 个；剪切另计。
底层椭圆 Gaussian 没有精确圆形分支，不应将 `e1=e2=0` 同时固定。
向量化缩小了计算图，具体 GPU 编译和执行速度仍需在目标环境计时。

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

质量 Gaussian 的积分振幅为
`A_i = upsilon_kappa * (F_i / sum(F)) * (sigma_i / sigma_ref)**(-ml_gradient)`，
其中 `F_i=amp_i` 是光的积分通量，`sigma_ref` 是所有 sigma 的几何平均。
最简单的常数 M/L 情形应固定 `ml_gradient=0.0`；此时
`kappa_star = upsilon_kappa * I_light / sum(F)`。
`upsilon_kappa` 的量纲是坐标单位的平方（通常 arcsec²），不是物理 M/L，
只有梯度为零时才等于总 stellar convergence integral。非零梯度的权重没有
再次归一化。底层 stellar Gaussian 也没有精确圆形分支，不能同时固定
某个 Gaussian 的 `e1=e2=0`。

独立验证入口是 `utils/validate_stellar_mge_nfw.py`，在 Herculens 环境运行：

```bash
python utils/validate_stellar_mge_nfw.py --results-dir /path/to/pixelated_svi
```

它使用独立的 projected-density 积分验证偏折、收敛度和 Hessian，核对
sampled lens-light dependency，并生成独立模拟及六参数回收结果。
NFW 是有限个 Gaussian 的近似，其误差随 `r/R_s` 变化；验证输出保留失败的
精度门槛、严格圆形边界和径向误差图。不能用拟合优劣或一次回收替代这些检验。

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

### 多个点源

```python
point_prior = {
    "ra": [0.0, 0.03, -0.1, 0.1],
    "dec": [0.0, 0.03, -0.1, 0.1],
    "amp": [2.0, 0.2],
}
point_sources = PointSourceProfile(
    ["SOURCE_POSITION"] * 2,
    prior=[point_prior] * 2,
)
profiles = LensProfileCollection(point_source=point_sources)
```

批量构造返回 `ProfileCollection`；两个源的参数分别采样，配置也分别复制。
如果已知两个源的位置不同，可用 `prior=[first_prior, second_prior]` 分别指定。
`amp` 的两项是对数正态分布的 `[log_loc, log_scale]`，所以上例的振幅中位数为 `exp(2)`。

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

在构建模型之前调用 `initialize_from()`。`components` 选择当前组内唯一的
profile 名称；有重复名称时用从 0 开始的 `component_idx`。两个选择器不能同时使用，
省略两者会固定所有 profile。所选 profile 的参数名必须完全一致，缺失或多余都会报错；
未选中的 profile 保留原先的 prior，继续采样。

```python
mass = MassProfile(["EPL", "SHEAR"], prior=[epl_prior, shear_prior])
mass.initialize_from({"EPL": epl_values}, components=["EPL"])

# 重复的 EPL 必须使用索引；只有第一个 EPL 被固定。
mass = MassProfile(["EPL", "EPL", "SHEAR"], prior=[epl_prior, epl_prior, shear_prior])
mass.initialize_from(initial_parameters, component_idx=[0])

# initial_parameters 可为整个组的有序参数列表，也可仅包含所选项的参数列表。
# kwargs_result.json 的物理成分列表始终按原始 profile 索引读取。
mass.initialize_from("previous_run/kwargs_result.json", component_idx=[0])

# LightProfile 组使用相同接口。PixelatedSource 固定物理源图，保留当前网格配置；
# 像素和正则化参数都不再采样，pixels 的二维形状必须匹配模型源网格。
source = PixelatedSource(pixel_grid=source_grid)
source.initialize_from({"pixels": source_pixels}, component_idx=[0])
```

无需专用的 mass collection，也无需在 `LensProfileCollection` 上添加此方法。
下面的旧版声明方式继续兼容：

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

`MPPL(m=1)` 支持自由采样 `gamma`，包括穿过 `gamma=2`。
令 `p=3-gamma`，其势函数采用连续规范：
`psi_1 = b**(gamma-1) * a_1 * r*cos(theta-phi_1) * p/(p+1)`
`* log(r) * exprel((p-1)*log(r))`，其中 `exprel(z)=expm1(z)/z`，
零点用 Taylor 展开计算，保留正确的 `gamma` 梯度。
半径以 arcsec 表示，对数的参考尺度固定为 1 arcsec；
`gamma=2` 时连续回到 `a_1*b*r*log(r)*cos(theta-phi_1)/2`。
实现使用 Cartesian dipole 与已有的微小半径软化，以避免中心角度奇点。

这个规范从旧的非等温 `m=1` 解中减去线性势，去掉发散的常量偏折；
在远离软化中心处保留同一个密度／剪切扰动。`MPPL_OFFSET(m=1)` 同样采用该解。
旧的非等温 `m=1` 拟合结果若继续使用，需要同步平移源坐标；
建议新采样重新初始化，避免直接沿用旧规范下的源位置或源像素图。
`m=3,4` 的势函数保持原公式。可运行 `utils/validate_mppl.py` 验证密度与导数。

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

### EPL plus elliptical m=3,4 multipoles (`EPL_MULTIPOLE_M3M4_ELL`)

`EPL_MULTIPOLE_M3M4_ELL` is the wrapper's JAX-compatible adaptation of the
lenstronomy/JAXtronomy joint profile. It includes the EPL and both elliptical
perturbations in one mass component:

```python
from herculens_wrapper.api import MassProfile, LensProfileCollection

epl_m3m4 = MassProfile("EPL_MULTIPOLE_M3M4_ELL", prior={
    "theta_E": [0.5, 2.0], "gamma": [1.6, 2.4],
    "q": [0.3, 0.9], "phi": [-90.0, 90.0],
    "center_x": [-0.2, 0.2], "center_y": [-0.2, 0.2],
    "a3_a": [-0.02, 0.02], "delta_phi_m3": [-15.0, 15.0],
    "a4_a": [-0.02, 0.02], "delta_phi_m4": [-15.0, 15.0],
})
profiles = LensProfileCollection(lens_mass=[epl_m3m4])
```

`gamma` is sampled freely for the EPL. The perturbations remain isothermal,
with convergence proportional to `1/R`, and do not receive `gamma`. Both terms
share the EPL center, axis ratio, and reference direction. API angles are in
degrees; `delta_phi_m3` and `delta_phi_m4` are eccentric anomalies relative to
the EPL major axis. Amplitudes are dimensionless and become
`a_m = a*_a * theta_E` internally. Native `e1`, `e2` may replace `q`, `phi`.

The local circular fallback evaluates the perturbation in the reference
ellipse's frame, preserving its orientation and consistency between the
potential, deflection, and Hessian as `q` approaches one.

To reproduce the checks against a local lenstronomy source checkout, run
`python utils/validate_epl_m3m4_ell.py --reference-root ../lenstronomy` in the
Herculens environment. This checks potential, deflection, and Hessian values,
the circular limit, derivatives through `gamma=2`, and public API sampling.
The report is saved to `results/epl_m3m4_validation/validation.json`.

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

# 按成分从旧模型的 kwargs_result.json warm start；也可传结果目录。
# 这里的 mass 参数仍然是自由参数。新模型多出的参数保留 init_to_median 初值，
# 旧文件多出的参数会被忽略。省略 components 时加载所有匹配的成分。
initial = model.initialize(
    seed=42,
    init_path="previous_svi/run_0/kwargs_result.json",
    components=["lens_mass"],
)
result = model.run(SamplerConfig.svi(max_iterations=5000), init_params=initial)

# 也可以直接交给 run；n_runs>1 时各次运行分别按自己的 seed 初始化。
result = model.run(
    SamplerConfig.svi(max_iterations=5000),
    save_path="new_svi",
    n_runs=3,
    init_path="previous_svi/run_0",
    components=["lens_mass", "lens_light"],
)

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

`result.output()` 写出的 `modeling_result.fits` 包含 `MEDIAN_MODEL`、
`IMAGE_DATA`、`NOISE_MAP`、`PSF`、`LENS_LIGHT` 和 `SOURCE_PLANE` 扩展，
以及已有的掩膜。`PSF` 保留输入核，`PSFSSAMP` 记录其相对图像像素的超采样倍数。
`SOURCE_PLANE` 的物理坐标保存在 `SOURCE_X`/`SOURCE_Y`；RTU 网格则使用
`SOURCE_X_CORNERS`/`SOURCE_Y_CORNERS`。源平面数值以图像像素通量为单位。

`SUMMARY` 头关键字注明图像汇总方式：SVI 的 `GUIDEMED` 表示在 guide 的
中值参数处计算图像；HMC 的 `PIXMED` 表示对每次后验抽样生成的图像逐像素取
中值。HMC 的 `MEDIAN_MODEL`、`LENS_LIGHT` 和 `SOURCE_PLANE` 都采用后者，
它们不是最大 log likelihood 样本的图像。没有后验中值的单点优化结果仍使用
`BEST_FIT_MODEL`（`SUMMARY=PARAMSET`）。

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
