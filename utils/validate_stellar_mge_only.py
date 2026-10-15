#!/usr/bin/env python3
"""Audit archived STELLAR_MGE + SHEAR fits using independent density integrals.

This reevaluates saved parameters with the installed backend; it neither refits
the data nor asserts that a light-tracing mass model measures physical stars.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from herculens.LightModel.Profiles.gaussian import GaussianEllipse
from herculens.MassModel.mass_model import MassModel
from herculens_wrapper.models import _materialize_dynamic_stellar_mge
from herculens_wrapper.profiles.composite import StellarMGE
from herculens_wrapper.profiles.registry import register_mass_profiles
from utils.validate_stellar_mge_nfw import (
    errors, reference_hessian, shear_alpha, stellar_alpha, stellar_amplitudes,
    stellar_density,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_fields(output, name, shape, extent, actual_k, reference_k, actual_a, reference_a):
    alpha, reference = np.linalg.norm(actual_a, axis=0), np.linalg.norm(reference_a, axis=0)
    density_error = (actual_k-reference_k)/max(np.max(np.abs(reference_k)), 1e-30)
    alpha_error = np.linalg.norm(actual_a-reference_a, axis=0)/max(np.max(reference), 1e-30)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), layout="constrained")
    rows = ((actual_k, reference_k, density_error), (alpha, reference, alpha_error))
    titles = (("Backend stellar convergence", "Independent Gaussian density", "Normalized density difference"),
              ("Backend deflection magnitude", "Independent integral", "Normalized deflection difference"))
    for row, (fields, labels) in enumerate(zip(rows, titles)):
        limit = max(np.max(fields[0]), np.max(fields[1]))
        for col, (field, title) in enumerate(zip(fields, labels)):
            kwargs = dict(cmap="viridis", vmin=0, vmax=limit)
            if col == 2:
                maximum = max(np.max(np.abs(field)), 1e-16)
                kwargs = dict(cmap="RdBu_r" if row == 0 else "magma",
                              vmin=-maximum if row == 0 else 0, vmax=maximum)
            im = axes[row, col].imshow(np.reshape(field, shape), origin="lower", extent=extent,
                                       interpolation="nearest", **kwargs)
            axes[row, col].set(title=title, xlabel="x (arcsec)", ylabel="y (arcsec)", aspect="equal")
            bar = fig.colorbar(im, ax=axes[row, col], fraction=.046, pad=.035)
            bar.set_label("Dimensionless" if row == 0 else ("Arcsec" if col < 2 else "Dimensionless"))
    fig.suptitle(f"Stellar MGE implementation audit: {name}\nSame saved parameters; independent references", fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"field_comparison.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def audit_run(path, mask):
    configuration = json.loads((path/"model_configuration.json").read_text())
    saved = json.loads((path/"kwargs_result.json").read_text())
    types = configuration["backend_definition"]["types"]["lens_mass_type_list"]
    if types != ["STELLAR_MGE", "SHEAR"]:
        raise ValueError(f"{path}: expected STELLAR_MGE + SHEAR, found {types}")
    stellar, shear = saved["kwargs_lens"]
    light = saved["kwargs_lens_light"]
    definition = configuration["backend_definition"]["parameters"]["lens_mass_params_list"][0]
    materialized = _materialize_dynamic_stellar_mge(
        {k: stellar[k] for k in ("upsilon_kappa", "ml_gradient")}, definition, light)
    fields = {key: errors(materialized["light_"+key], stellar["light_"+key])
              for key in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}
    flux, sigma = np.asarray(stellar["light_amp"]), np.asarray(stellar["light_sigma"])
    modulus = np.hypot(stellar["light_e1"], stellar["light_e2"])
    if np.any(flux <= 0) or np.any(sigma <= 0) or np.any(modulus >= 1) or np.any(modulus == 0):
        raise ValueError("Positive flux/width and supported non-circular geometry required")
    shape = tuple(configuration["observation"]["image_shape"])
    scale = configuration["observation"]["pixel_scale"]
    xs = (np.arange(shape[1])-(shape[1]-1)/2)*scale
    ys = (np.arange(shape[0])-(shape[0]-1)/2)*scale
    xx, yy = np.meshgrid(xs, ys)
    x, y = xx.ravel(), yy.ravel()
    backend = StellarMGE()
    actual_a = np.asarray(backend.derivatives(x, y, **stellar))
    actual_h = np.asarray(backend.hessian(x, y, **stellar))
    actual_k = (actual_h[0]+actual_h[1])/2
    expected_a = stellar_alpha(x, y, stellar, order=256)
    expected_k = stellar_density(x, y, stellar)
    expected_h = reference_hessian(lambda X, Y: stellar_alpha(X, Y, stellar), x, y)
    spatial = dict(alpha=errors(actual_a, expected_a, vector=True),
        kappa=errors(actual_k, expected_k), hessian=errors(actual_h, expected_h),
        quadrature_convergence=errors(stellar_alpha(x, y, stellar, 128), expected_a, vector=True),
        hessian_step_convergence=errors(reference_hessian(
            lambda X, Y: stellar_alpha(X, Y, stellar), x, y, 5e-6), expected_h))
    if mask is not None and mask.shape == shape:
        use = mask.astype(bool).ravel()
        spatial["arc_mask"] = dict(n_pixels=int(use.sum()),
            alpha=errors(actual_a[:, use], expected_a[:, use], vector=True),
            kappa=errors(actual_k[use], expected_k[use]))
    light_profile = GaussianEllipse()
    intrinsic_light = sum(np.asarray(light_profile.function(x, y, **p)) for p in light)
    if stellar["ml_gradient"] == 0.0:
        spatial["constant_ml_identity"] = errors(actual_k,
            stellar["upsilon_kappa"]*intrinsic_light/flux.sum())
    amplitudes = stellar_amplitudes(stellar)
    integrated = sum(quad(lambda r: a*r/s**2*np.exp(-r*r/(2*s*s)), 0, 12*s,
                          epsabs=1e-12)[0] for a, s in zip(amplitudes, sigma))
    gradients = {}
    for key in ("upsilon_kappa", "ml_gradient"):
        def objective(value):
            return jnp.sum(jnp.asarray(backend.derivatives(.2, .3, **{**stellar, key: value}))**2)
        value = stellar[key]; step = 1e-5*max(abs(value), .1)
        gradients[key] = errors(jax.grad(objective)(value),
            (objective(value+step)-objective(value-step))/(2*step))
    changed_light = [dict(p) for p in light]
    changed_light[0]["amp"] *= 1.2
    changed_light[0]["center_x"] += .01
    changed = _materialize_dynamic_stellar_mge(
        {k: stellar[k] for k in ("upsilon_kappa", "ml_gradient")}, definition, changed_light)
    for key in ("amp", "sigma", "e1", "e2", "center_x", "center_y"):
        np.testing.assert_allclose(changed["light_"+key], [p[key] for p in changed_light], atol=0, rtol=0)
    model = MassModel(types)
    actual_total = np.asarray(model.alpha(x, y, saved["kwargs_lens"]))
    total_reference = expected_a+shear_alpha(x, y, shear)
    spatial["total_with_shear"] = errors(actual_total, total_reference, vector=True)
    rescaled = {**stellar, "light_amp": flux*17.0}
    spatial["global_light_flux_rescaling"] = errors(backend.derivatives(x, y, **rescaled), actual_a, vector=True)
    # These are saved ELBO samples, not an independent posterior convergence diagnostic.
    loss = np.asarray(json.loads((path/"svi_loss_history.json").read_text())["loss_history"])
    fit_metrics = json.loads((path/"metrics.json").read_text())
    saved_sigma = json.loads((path/"kwargs_sigma.json").read_text())["kwargs_lens"][0]
    prior_offsets = [(np.log(p["amp"])-c["parameters"]["amp"]["prior"][0])/
        c["parameters"]["amp"]["prior"][1] for p,c in zip(light, configuration["profiles"]["lens_light"])]
    gates = dict(saved_mass_light_fields=all(v["max_abs"] == 0 for v in fields.values()),
        constant_ml=spatial.get("constant_ml_identity", {"relative_l2": 0})["relative_l2"] < 1e-10,
        alpha_relative_l2=spatial["alpha"]["relative_l2"] < 1e-6,
        kappa_relative_l2=spatial["kappa"]["relative_l2"] < 1e-10,
        hessian_relative_l2=spatial["hessian"]["relative_l2"] < 1e-5,
        reference_quadrature=spatial["quadrature_convergence"]["relative_l2"] < 1e-7,
        reference_hessian_step=spatial["hessian_step_convergence"]["relative_l2"] < 1e-5,
        total_area=np.isclose(integrated, amplitudes.sum(), rtol=1e-10, atol=1e-12).item(),
        parameter_gradients=all(v["relative_l2"] < 1e-5 for v in gradients.values()),
        dynamic_field_response=True,
        global_light_rescaling=spatial["global_light_flux_rescaling"]["relative_l2"] < 1e-10)
    report = dict(run=path.name, directory=str(path),
        files_sha256={p.name:digest(p) for p in (path/"kwargs_result.json",path/"model_configuration.json")},
        fields=fields, spatial=spatial, gradients=gradients, acceptance_gates=gates,
        implementation_pass=all(gates.values()),
        gaussian_count=len(flux), q_range=[float(((1-modulus)/(1+modulus)).min()),float(((1-modulus)/(1+modulus)).max())],
        upsilon_kappa=stellar["upsilon_kappa"], ml_gradient=stellar["ml_gradient"],
        upsilon_kappa_saved_svi_sigma=saved_sigma["upsilon_kappa"],
        numerical_integrated_kappa_arcsec2=float(integrated), amplitude_sum_arcsec2=float(amplitudes.sum()),
        shear_strength=float(np.hypot(shear["gamma1"],shear["gamma2"])),
        light_total_flux=float(flux.sum()), smallest_sigma_arcsec=float(sigma.min()),
        lognormal_light_flux_prior_z=prior_offsets,
        final_100_loss_mean=float(loss[-100:].mean()), final_100_loss_std=float(loss[-100:].std()),
        saved_fit_metrics=fit_metrics)
    return report, (shape, [xs[0]-scale/2,xs[-1]+scale/2,ys[0]-scale/2,ys[-1]+scale/2],
                    actual_k, expected_k, actual_a, expected_a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"results/stellar_mge_only_validation")
    args = parser.parse_args()
    folder = args.results_dir.resolve()
    if (folder/"pixelated_svi").is_dir():
        folder /= "pixelated_svi"
    runs = [folder] if (folder/"kwargs_result.json").exists() else sorted(folder.glob("run_*"))
    runs = [p for p in runs if (p/"kwargs_result.json").exists()]
    if not runs:
        parser.error("No saved runs found")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42})
    mask_file = folder/"data/mask_1.fits"
    mask = fits.getdata(mask_file) if mask_file.exists() else None
    register_mass_profiles()
    reports, best = [], None
    for path in runs:
        print(f"Auditing {path.name}", flush=True)
        report, fields = audit_run(path, mask)
        reports.append(report)
        if best is None or report["saved_fit_metrics"]["CHI2_MEDIAN"] < best[0]["saved_fit_metrics"]["CHI2_MEDIAN"]:
            best = report, fields
    plot_fields(args.output, best[0]["run"], *best[1])
    sources = [Path(__file__), Path(inspect.getfile(StellarMGE)),
               Path(inspect.getfile(_materialize_dynamic_stellar_mge)), Path(inspect.getfile(stellar_alpha)),
               Path(inspect.getfile(type(StellarMGE()._mge)))]
    summary = dict(results_directory=str(folder), runs=reports,
        all_implementation_checks_pass=all(r["implementation_pass"] for r in reports),
        versions={n:importlib.metadata.version(n) for n in ("jax","numpy","herculens","jax-lensing-profiles")},
        precision="float64", source_sha256={str(p):digest(p) for p in sources},
        limitations=["Saved remote software versions are not reconstructed; fields are reevaluated with the installed local backend.",
                     "No independent image simulation or complete pixelated-source posterior coverage test is performed.",
                     "Mass follows light without a halo can absorb nonstellar mass; upsilon_kappa is not physical M/L.",
                     "The exactly circular Gaussian backend branch is unsupported; all saved components are noncircular."])
    (args.output/"validation.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    lines = ["# Stellar MGE only：实际结果审计", "",
        f"结果目录：`{folder}`", "",
        "固定同一组保存参数，用独立 Gaussian 密度和一维积分验证当前安装 backend，未重新拟合。",
        "当前运行是 STELLAR_MGE + SHEAR，5 个动态 lens-light Gaussian，固定 ml_gradient=0。", "",
        "## 数学定义和单位", "",
        r"$A_i=U\,F_i/\sum F_j\,(\sigma_i/\sigma_{ref})^{-g}$；$\sigma_{ref}$ 为几何平均宽度。",
        r"$g=0$ 时 $\kappa_*=U I_{intrinsic}/\sum F_i$，且 $\int\kappa_*d^2\theta=U$。",
        "amp 是积分通量；sigma 为面积保持椭圆半径的宽度，不能直接当成 major-axis sigma。",
        "upsilon_kappa 的单位是 arcsec²。物理总质量为 Sigma_crit × (D_lens/206265)² × U，",
        "其中 D_lens 和 Sigma_crit 使用匹配的长度单位。没有红移和光度标定不能直接称其为物理 M/L。", "",
        "## 数值验证", "",
        "误差为全图相对 L2 范数；JSON 另保存最大绝对、最大逐点相对误差和弧区误差。",
        "| run | U (arcsec²) | 偏折误差 | 密度误差 | Hessian 误差 | 检查通过 |",
        "|---|---:|---:|---:|---:|---|"]
    for r in reports:
        lines.append(f"| {r['run']} | {r['upsilon_kappa']:.6f} | {r['spatial']['alpha']['relative_l2']:.2e} | "
                     f"{r['spatial']['kappa']['relative_l2']:.2e} | {r['spatial']['hessian']['relative_l2']:.2e} | {r['implementation_pass']} |")
    lines += ["", "## 推断与用法需要分开评估", "",
        "光／质量字段一致、动态响应、面积归一化和参数梯度检查不等于物理假设已被验证。",
        "run_2 的 U 与其余初始化结果差异较大且 saved chi2 更高，应检查优化分支和 posterior convergence。",
        "所有运行固定常数 M/L、仅用 5 个光 Gaussian；应检查 PSF 卷积后的前景光残差、MGE 数目／宽度／几何限制的稳定性。",
        "光 amp 的 prior=[2,0.1] 是 ln(amp) 的 LogNormal 参数，而非 amp=2±0.1。",
        "JSON 保存每个 fitted amp 相对 log-prior 的标准化偏移；不能把初始化值当作修改了 prior。",
        "saved N_PARAMS_FITTED 大于 N_DATA_PIXELS，CHI2_DOF 被限制到 1；REDUCED_CHI2 不能直接作为常规 reduced chi-square。",
        "stellar-only 是光追踪质量的假设；不能仅凭拟合好就认定质量全来自恒星。", "",
        "## 范围", "", *[f"- {x}" for x in summary["limitations"]], "",
        f"![Independent fields]({(args.output/'field_comparison.png').resolve()})", ""]
    (args.output/"report.md").write_text("\n".join(lines))
    print(json.dumps({"all_implementation_checks_pass":summary["all_implementation_checks_pass"],
                      "report":str((args.output/"report.md").resolve()),
                      "failed_checks":{r["run"]:[k for k,v in r["acceptance_gates"].items() if not v] for r in reports}},indent=2))
    if not summary["all_implementation_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
