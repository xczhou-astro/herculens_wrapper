"""Audit saved spherical-NFW SVI point estimates, without rerunning inference.

Uses an independent NumPy NFW formula and stellar-density quadrature. Aperture
fractions are properties of the saved parameter vectors, not posterior limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.validate_stellar_mge_nfw import (
    errors, nfw_density, stellar_alpha,
)
from jax_lensing_profiles.MassModel.Profiles.NFW import NFW


def read(path):
    return json.loads(path.read_text())


def spherical_alpha(r, p):
    z = np.asarray(r, dtype=float) / p['R_s']
    f = np.ones_like(z)
    low, high = z < 1, z > 1
    f[low] = np.arccosh(1/z[low]) / np.sqrt(1-z[low]**2)
    f[high] = np.arccos(1/z[high]) / np.sqrt(z[high]**2-1)
    return 4*p['kappa_s']*p['R_s']*(np.log(z/2)+f)/z


def aperture(radii, st, halo, angular_order=128, density_order=128):
    """Integral kappa dA inside circles centered at the halo/light centroid.

    Divergence theorem: integral kappa dA = pi R mean(alpha dot radial_unit).
    The independent stellar oracle integrates Gaussian surface densities.
    """
    radii = np.atleast_1d(radii)
    phi = (np.arange(angular_order)+.5)*2*np.pi/angular_order
    c, s = np.cos(phi), np.sin(phi)
    x = halo['center_x'] + radii[:, None]*c
    y = halo['center_y'] + radii[:, None]*s
    a = stellar_alpha(x, y, st, order=density_order)
    ms = np.pi*radii*np.mean(a[0]*c+a[1]*s, axis=1)
    mh = np.pi*radii*spherical_alpha(radii, halo)
    return ms, mh, mh/(ms+mh)


def inspect_run(path):
    kw = read(path/'kwargs_result.json')
    st, halo, shear = kw['kwargs_lens']
    assert st['ml_gradient'] == 0
    # Check saved dynamic stellar-light linking and centroid, independently.
    light = kw['kwargs_lens_light']
    for key in ('amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y'):
        np.testing.assert_array_equal(st['light_'+key], [p[key] for p in light])
    flux = np.array(st['light_amp'])
    for key in ('center_x', 'center_y'):
        centroid = np.dot(flux, st['light_'+key])/flux.sum()
        np.testing.assert_allclose(halo[key], centroid, atol=1e-14, rtol=0)

    te = read(path/'lens_mass_parameters.json')['einstein_radius']['theta_E_eff_arcsec']
    radii = np.array([te, .4, .5, .8])
    ms, mh, fraction = aperture(radii, st, halo)
    ms_high, _, _ = aperture(radii, st, halo, 256, 256)

    # Compare the installed analytic-potential/autodiff backend against an
    # independent density/deflection oracle over a broad relevant radius range.
    r = np.unique(np.r_[np.geomspace(.03, 1.5, 100), halo['R_s']])
    angle = np.arange(r.size)*.73
    x, y = halo['center_x']+r*np.cos(angle), halo['center_y']+r*np.sin(angle)
    expected = spherical_alpha(r, halo)*np.array([np.cos(angle), np.sin(angle)])
    actual = np.array(NFW().derivatives(x, y, **halo))
    h = np.array(NFW().hessian(x, y, **halo))
    check = {'alpha': errors(actual, expected, vector=True),
             'kappa': errors((h[0]+h[1])/2, nfw_density(r, halo['kappa_s'], halo['R_s'])),
             'stellar_aperture_order_doubling': errors(ms, ms_high)}
    metrics = read(path/'metrics.json')
    with fits.open(path/'modeling_result.fits') as hdus:
        residual = (hdus['MEDIAN_MODEL'].data-hdus['IMAGE_DATA'].data)/hdus['NOISE_MAP'].data
        chi2 = float(np.sum(residual**2))
    np.testing.assert_allclose(chi2, metrics['CHI2_MEDIAN'], rtol=1e-10)
    history = np.array(read(path/'svi_loss_history.json')['loss_history'])
    row = dict(run=path.name, upsilon_kappa=st['upsilon_kappa'], **halo,
               shear_strength=float(np.hypot(shear['gamma1'], shear['gamma2'])),
               theta_E_eff=te, sigma_halo=read(path/'kwargs_sigma.json')['kwargs_lens'][1],
               chi2_from_fits=chi2, chi2_per_pixel=chi2/residual.size,
               aperture_radii=radii.tolist(), stellar_area=ms.tolist(),
               halo_area=mh.tolist(), halo_fraction=fraction.tolist(), numerical_checks=check,
               final_loss_median=float(np.median(history[-500:])),
               preceding_loss_median=float(np.median(history[-1500:-1000])))
    return row, (st, halo, shear), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/sph_nfw_svi_diagnostics')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.results_dir.glob('run_*'))
    inspected = [inspect_run(p) for p in runs]
    rows = [item[0] for item in inspected]
    starts = [read(p/'kwargs_init.json') for p in runs]
    same_start = all(s['kwargs_lens'] == starts[0]['kwargs_lens'] and
                     s['kwargs_lens_light'] == starts[0]['kwargs_lens_light'] for s in starts)
    config = read(args.results_dir/'config.json')
    comparison = read(args.results_dir/'comparison.json')
    context = []
    parametric = args.results_dir.parent/'parametric_svi'
    for p in sorted(parametric.glob('run_*')):
        if (p/'kwargs_result.json').exists():
            row, _, _ = inspect_run(p)
            row['matches_pixelated_start'] = read(p/'kwargs_result.json')['kwargs_lens'] == starts[0]['kwargs_lens']
            context.append(row)
    summary = dict(results_dir=str(args.results_dir.resolve()),
                   interpretation='Saved SVI parameter vectors only; no posterior calibration.',
                   aperture_definition='Circular aperture about the light flux centroid; not the critical-curve interior.',
                   same_mass_light_initialization=same_start, init_params_path=config['init_params_path'],
                   runs=rows, parametric_context=context, comparison_json=comparison,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir/'diagnostics.json').write_text(json.dumps(summary, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout='constrained')
    radii = np.geomspace(.12, 1.0, 36)
    for row, (st, halo, shear), history in inspected:
        ms, mh, frac = aperture(radii, st, halo)
        axes[0].plot(radii, 100*frac, label=row['run'])
        # Radial mean stellar deflection, spherical halo and shear magnitude.
        if row is rows[0]:
            axes[1].plot(radii, ms/(np.pi*radii), label='Stellar radial mean')
            axes[1].plot(radii, mh/(np.pi*radii), label='Spherical NFW')
            axes[1].plot(radii, row['shear_strength']*radii, label='External shear magnitude')
        blocks = history[-3000:].reshape(-1, 100)
        axes[2].plot(np.arange(7050, 10000, 100), np.median(blocks, axis=1), label=row['run'])
    axes[0].set(xlabel='Circular aperture radius [arcsec]', ylabel='Halo projected mass fraction [%]',
                title='Saved SVI vectors; no uncertainty bands')
    axes[1].set(xlabel='Radius [arcsec]', ylabel='Deflection [arcsec]', yscale='log', title='run_0 component contributions')
    axes[2].set(xlabel='SVI iteration', ylabel='Negative ELBO (100-step median)', title='Final 3,000 iterations')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(args.output_dir/'diagnostics.png', dpi=180)
    plt.close(fig)

    lines = ['# StellarMassMGE + spherical NFW：SVI 诊断', '',
             f'输入：`{args.results_dir.resolve()}`', '',
             '这些数值描述保存的 SVI 参数向量，不是经验证的真实后验，也不是 halo 质量上限。', '',
             '| Run | upsilon_kappa [arcsec²] | kappa_s | R_s [arcsec] | gamma_ext | theta_E_eff [arcsec] | f_halo(<theta_E_eff) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['upsilon_kappa']:.6f} | {r['kappa_s']:.6f} | {r['R_s']:.6f} | {r['shear_strength']:.6f} | {r['theta_E_eff']:.6f} | {100*r['halo_fraction'][0]:.3f}% |")
    lines += ['', '质量分数为同一圆孔径内 halo / (halo + stellar) 的投影质量；圆心为光度加权中心。',
              '孔径半径采用保存的临界曲线面积等效 Einstein radius，不等于沿非圆形临界曲线积分。',
              '面积归一化 integral(kappa d²theta) 的单位为 arcsec²；计算比例无需红移或宇宙学。剪切的 convergence 为零，不加入质量分母。', '',
              f"四次 mass/light 初值完全相同：{same_start}；config 的 init_params_path 为 `{config['init_params_path']}`。",
              '这可以检验同一初值附近的随机种子稳定性，不能当作四个独立起点探索后验。', '',
              '给定先验下限 kappa_s=1e-5、R_s=0.03 arcsec，这四次点估计偏小但未贴边。',
              'kwargs_sigma.json 是 2,000 次 guide 抽样的标准差；非对称、长尾分布不宜用 median ± sigma 表示可信区间。', '']
    for r in rows:
        lines.append(f"- {r['run']}：guide SD(kappa_s)={r['sigma_halo']['kappa_s']:.5f}、SD(R_s)={r['sigma_halo']['R_s']:.5f} arcsec；最后 500 步 loss 中位数 {r['final_loss_median']:.2f}，前一窗口 {r['preceding_loss_median']:.2f}。")
    lines += ['', '独立检查：保存的 stellar MGE 与 lens light 参数逐项相同；halo 中心等于独立计算的光度加权中心。',
              'NFW 后端偏折和 convergence 对照 NumPy 解析式，在 0.03–1.5 arcsec 上检查，包含 r=R_s；下面误差含后端中心平滑 epsilon 的影响。', '']
    for r in rows:
        c = r['numerical_checks']
        lines.append(f"- {r['run']}：偏折 max rel={c['alpha']['max_relative']:.3g}，kappa max rel={c['kappa']['max_relative']:.3g}，恒星孔径积分阶数翻倍 max rel={c['stellar_aperture_order_doubling']['max_relative']:.3g}。")
    lines += ['', '数值检查支持这一分支的质量计算和参数链接正确；不证明 SVI 找到了全局解或给出正确后验。', '',
              '邻近 parametric_svi 的结果仅作初始化对照（source 模型不同，不作模型证据比较）：', '',
              '| Run | kappa_s | R_s [arcsec] | gamma_ext | chi²/N_pixel | matches pixelated init |',
              '|---|---:|---:|---:|---:|---|']
    for r in context:
        lines.append(f"| {r['run']} | {r['kappa_s']:.6f} | {r['R_s']:.6f} | {r['shear_strength']:.6f} | {r['chi2_per_pixel']:.6f} | {r['matches_pixelated_start']} |")
    lines += ['', '指标检查：直接从 FITS 的 data/model/noise 重算 chi²，与每个 run 的 metrics.json 一致。',
              'pixelated 的 nominal 自由参数数大于数据像素数，保存的 CHI2_DOF 被截为 1；REDUCED_CHI2 不具有通常的拟合优度含义。',
              'comparison.json 的参数计数/BIC 与各 run/metrics.json 不一致，不能据此作模型选择；仅用重算的 chi² 作条件拟合诊断。', '',
              '解释：这四次 SVI 参数向量对应较弱的圆形 halo 和 gamma_ext≈0.25 的角结构。恒星 MGE、剪切和 source 重建的分配需联合诊断；大剪切是否物理可信需要环境信息。',
              '优先的下一步：分别从 parametric run_1、run_2 和显著 halo 初值启动 pixelated；固定若干 R_s，逐一联合重拟合 stellar/shear/source，检查 likelihood 与正则项；与 halo=0 的控制模型对照。',
              '模拟验证应以独立渲染器注入非零 spherical halo，再检查实现的偏折、Hessian、梯度和参数恢复；同一径向区域无法分解 stellar/halo 时，应检验可识别的总偏折和孔径质量组合。', '',
              '参考：[Keeton NFW 与圆对称质量关系](https://arxiv.org/html/astro-ph/0102341v2)；[NumPyro AutoGuide](https://num.pyro.ai/en/stable/autoguide.html)。', '',
              '![诊断图](diagnostics.png)', '']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'output_dir':str(args.output_dir.resolve()), 'runs':rows}, indent=2))


if __name__ == '__main__':
    main()
