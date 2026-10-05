"""Relate saved MASS_MGE widths to half-mass and Einstein radii.

No inference is rerun. Circular formulae and independent elliptical aperture
integrals are compared with the installed backend's deflection/Hessian.
The critical curve may be outside the observed field: this is extrapolation.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from herculens_wrapper.profiles import register_mass_profiles
from herculens.MassModel.mass_model import MassModel


def enclosed(radius, amp, sigma, q=1., order=128):
    """Exact radial integral followed by periodic angular quadrature.

    Coordinates use the backend's area-preserving ellipse definition.
    The aperture is a circle centred on the common MGE centre.
    """
    radius = np.asarray(radius)
    angle = np.arange(order) * (2*np.pi/order)
    g = q*np.cos(angle)**2 + np.sin(angle)**2/q
    exponent = radius[..., None, None]**2 * g / (2*sigma[:, None]**2)
    fractions = (-np.expm1(-exponent)/g).mean(axis=-1)
    return (fractions*amp).sum(axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results/mge_sigma_radius_diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    register_mass_profiles()
    mass = MassModel(["MASS_MGE", "SHEAR"])

    @jax.jit
    def tangential_eigenvalue(radius, angles, kwargs):
        x = kwargs[0]['center_x'] + radius*jnp.cos(angles)
        y = kwargs[0]['center_y'] + radius*jnp.sin(angles)
        xx, xy, yx, yy = mass.hessian(x, y, kwargs)
        kappa = (xx+yy)/2
        gamma = jnp.hypot((xx-yy)/2, (xy+yx)/2)
        return 1-kappa-gamma

    radii = np.geomspace(.003, 8., 400)
    records = []
    curves = []
    for run in sorted(args.results_dir.glob("run_*")):
        kwargs = json.loads((run/"kwargs_result.json").read_text())["kwargs_lens"]
        p = kwargs[0]
        amp, sigma = np.asarray(p['amp']), np.asarray(p['sigma'])
        e = np.hypot(p['e1'], p['e2']); q = (1-e)/(1+e)
        total = amp.sum()
        half = brentq(lambda r: enclosed(r, amp, sigma, q)-total/2, 1e-8, 8.)
        aperture_radius = brentq(lambda r: enclosed(r, amp, sigma, q)/(np.pi*r*r)-1, 1e-4, 8.)
        check_r = np.array([.3, .4, .5])
        area = enclosed(check_r, amp, sigma, q)
        np.testing.assert_allclose(area, enclosed(check_r, amp, sigma, q, order=256), rtol=1e-12)
        # Flux of alpha through a circular boundary equals twice enclosed kappa.
        angle = np.arange(256)*2*np.pi/256
        cx, cy = p['center_x'], p['center_y']
        x = cx+check_r[:, None]*np.cos(angle)
        y = cy+check_r[:, None]*np.sin(angle)
        ax, ay = mass.alpha(jnp.asarray(x), jnp.asarray(y), kwargs, k=0)
        radial_alpha = np.asarray(ax)*np.cos(angle)+np.asarray(ay)*np.sin(angle)
        backend_area = np.pi*check_r*radial_alpha.mean(axis=-1)
        np.testing.assert_allclose(backend_area, area, rtol=2e-6, atol=1e-8)
        # Find the actual outer lambda_t=0 curve including saved shear.
        # Search beyond the image only to quantify the saved model prediction.
        kw = [{k:jnp.asarray(v) for k,v in component.items()} for component in kwargs]
        lo, hi = np.ones(256), np.full(256, 8.)
        assert np.all(tangential_eigenvalue(lo, angle, kw) < 0)
        assert np.all(tangential_eigenvalue(hi, angle, kw) > 0)
        for _ in range(48):
            mid = (lo+hi)/2
            inside = np.asarray(tangential_eigenvalue(mid, angle, kw)) < 0
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        critical_radii = (lo+hi)/2
        critical_radius = np.sqrt(np.mean(critical_radii**2))
        scaled_amp = amp/(area[1]/(np.pi*.4**2))
        local_kappa = np.sum(scaled_amp/(2*np.pi*sigma**2)*np.exp(-.4**2/(2*sigma**2)))
        record = {
            "run":run.name, "sigma_arcsec":sigma.tolist(), "amp_arcsec2":amp.tolist(),
            "total_convergence_area_arcsec2":float(total), "axis_ratio_q":float(q),
            "circular_aperture_half_mass_radius_arcsec":float(half),
            "circular_aperture_mean_kappa_one_radius_arcsec":float(aperture_radius),
            "critical_curve_area_radius_arcsec_extrapolated":float(critical_radius),
            "aperture_radii_arcsec":check_r.tolist(), "mean_kappa":(area/(np.pi*check_r**2)).tolist(),
            "backend_enclosed_area_relative_error":float(np.max(np.abs(backend_area/area-1))),
            "amp_scale_to_mean_kappa_one_at_0p4_arcsec":float(np.pi*.4**2/area[1]),
            "scaled_amp_median_arcsec2":float(np.median(scaled_amp)),
            "circularized_local_kappa_after_rescaling_at_0p4":float(local_kappa),
            "circularized_alpha_log_slope_after_rescaling_at_0p4":float(2*local_kappa-1),
        }
        records.append(record)
        curves.append(enclosed(radii, amp, sigma, q)/(np.pi*radii**2))
        print(json.dumps(record), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for record, curve in zip(records, curves):
        axes[0].loglog(radii, curve, label=record['run'], lw=1.5)
    axes[0].axhline(1, color='black', ls='--', lw=1)
    axes[0].axvspan(.3, .5, color='grey', alpha=.2)
    axes[0].set(xlabel='Circular aperture radius [arcsec]', ylabel='Mean convergence',
                title='Saved models: mass scale exceeds arc scale')
    axes[0].legend(fontsize=8)
    first = records[0]
    axes[1].loglog(radii, curves[0], label='Saved run_0', color='C0')
    axes[1].loglog(radii, curves[0]*first['amp_scale_to_mean_kappa_one_at_0p4_arcsec'],
                   label='Same widths; amplitudes rescaled', color='C1')
    axes[1].axhline(1, color='black', ls='--', lw=1)
    axes[1].axvspan(.3, .5, color='grey', alpha=.2)
    axes[1].set(xlabel='Circular aperture radius [arcsec]', ylabel='Mean convergence',
                title='Changing amplitudes changes the lensing radius')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'sigma_radius.png', dpi=180)
    plt.close(fig)
    (args.output/'diagnostics.json').write_text(json.dumps(records, indent=2)+'\n')
    rows = [f"| {r['run']} | {r['total_convergence_area_arcsec2']:.4f} | {r['axis_ratio_q']:.6f} | {r['circular_aperture_half_mass_radius_arcsec']:.5f} | {r['critical_curve_area_radius_arcsec_extrapolated']:.5f} | {r['mean_kappa'][1]:.3f} |" for r in records]
    report = r"""# MGE sigma 与半质量 / 爱因斯坦半径诊断

本报告只重新评估保存的 parametric SVI 点估计，不重新拟合，不将 SVI 当作可信后验。
临界曲线计算延伸到观测图之外，是保存模型的外推预测，不是图内测量。

Gaussian amp 是积分 convergence A_i，单位 arcsec²。面积保持的椭圆半径为
R²=q X²+Y²/q，κ_i=A_i/(2πσ_i²) exp[-R²/(2σ_i²)]。
因此 σ_major=σ/sqrt(q)，σ_minor=σ sqrt(q)。单 Gaussian 的等面积半质量半径
为 sqrt(2 ln 2) σ ≈1.17741σ；MGE 没有单一 σ，其半质量半径需要累计所有振幅。

圆对称情形累计 convergence 为
Mκ(<r)=Σ A_i [1-exp(-r²/(2σ_i²))]。
半质量条件是 Mκ(<R_half)=ΣA_i/2；爱因斯坦半径条件是 Mκ(<R_E)=πR_E²。
后一个条件只对圆对称、无外剪切情形严格等于切向临界半径。
椭圆及剪切情况下另用 Hessian 的 λ_t=1-κ-|γ|=0 定义临界曲线，
计算 sqrt(area/pi)。两种定义在本例近圆形、弱剪切结果中数值接近。

| run | ΣA [arcsec²] | q | 圆孔径 R_half [arcsec] | 外推临界曲线 R_E,eff [arcsec] | mean κ(<0.4″) |
|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + r"""

此次 σ prior 是 [0.006,0.246] arcsec，而不是原先的 [0.03,1.23]；
点估计宽度约 [0.0072,0.2044]。amp prior 为 LogNormal(2,0.2)，
每个振幅中位数 e²≈7.389，总积分 convergence 约74。
在0.3–0.5 arcsec内 mean κ约94–250，远离产生该尺度圆形爱因斯坦环的 mean κ=1。
因此本次首先暴露的是质量归一化不匹配，不能据此判定 MGE 表达能力或偏折实现失败。

仅把 run_0 振幅整体缩放到 mean κ(<0.4″)=1，需要乘约0.006918，
即每个振幅约0.051 arcsec²；这只是诊断尺度，不是一次新的拟合。
保持这些等权且紧凑的 Gaussian，缩放后 κ(0.4″)约0.0363，偏折对数斜率约-0.927，
接近质量集中在环内的情形。恢复较宽 σ 范围并放宽各分量权重后才能探索更广的径向形状。
σ_max 不必等于或大于 R_E；即使所有 Gaussian 都很紧凑，足够质量也能给出较大的 R_E。

检查：128/256角向积分一致；独立孔径积分与实际 MASS_MGE 偏折边界积分一致；
外推临界曲线在256个方位上求根，并包含保存的 SHEAR。输出未包含 SVI 不确定性。

理论参考：[Shajib 2019](https://arxiv.org/abs/1906.08263)；
[Kochanek 的圆对称透镜讲义](https://ned.ipac.caltech.edu/level5/March04/Kochanek2/Kochanek3_2.html)。
详细数值见 diagnostics.json，图见 sigma_radius.png。
"""
    (args.output/'REPORT.md').write_text(report)


if __name__ == '__main__':
    main()
