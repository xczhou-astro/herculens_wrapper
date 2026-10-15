#!/usr/bin/env python3
"""Compare MPPL m=3,4 with JAXtronomy or an independent analytic solution.

Run with /opt/anaconda3/envs/herculens/bin/python. For gamma=2:
JAXtronomy's physical amplitude is b times the wrapper's dimensionless a_m.
For gamma!=2, JAXtronomy is a different-model baseline, not an equality check.
Use --reference analytic to validate the non-isothermal power-law solution.
The curves show perturbations alone, so an EPL cannot hide their residuals.
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from jaxtronomy.LensModel.Profiles.multipole import Multipole
from herculens_wrapper.profiles.multipole import MPPL

BLUE, ORANGE = "#0072B2", "#D55E00"
QUANTITIES = ("potential", "alpha_x", "alpha_y", "kappa")
LABELS = (r"$\psi_m$ (arcsec$^2$)", r"$\alpha_{x,m}$ (arcsec)",
          r"$\alpha_{y,m}$ (arcsec)", r"$\delta\kappa_m$")


def analytic_values(x, y, m, amplitude, b, phase, gamma):
    """Unsoftened polar Poisson solution with explicit Cartesian derivatives."""
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    p = 3.0 - gamma
    c = amplitude * b**(gamma - 1.0) * p / (p*p - m*m)
    co, si = np.cos(m*(theta-phase)), np.sin(m*(theta-phase))
    potential = c * r**p * co
    radial = c * p * r**(p-1.0) * co
    tangential = -c * m * r**(p-1.0) * si
    ax = radial*np.cos(theta) - tangential*np.sin(theta)
    ay = radial*np.sin(theta) + tangential*np.cos(theta)
    kappa = p/2.0 * (b/r)**(gamma-1.0) * amplitude * co
    return np.stack((potential, ax, ay, kappa))


def evaluate(x, y, m, amplitude, b, phase, gamma=2.0, reference_kind="jaxtronomy"):
    common = dict(m=m, phi_m=np.deg2rad(phase), center_x=0.0, center_y=0.0)
    native = dict(**common, a_m=amplitude, b=b, gamma=gamma)
    reference = dict(**common, a_m=b * amplitude)
    values = []
    for profile, kwargs in ((MPPL, native), (Multipole(), reference)):
        potential = np.asarray(profile.function(x, y, **kwargs))
        ax, ay = profile.derivatives(x, y, **kwargs)
        h = profile.hessian(x, y, **kwargs)
        kappa = (h[0] + h[1 if profile is MPPL else 3]) / 2.0
        values.append(np.stack((potential, np.asarray(ax), np.asarray(ay), np.asarray(kappa))))
    if reference_kind == "analytic":
        values[1] = analytic_values(x, y, m, amplitude, b, np.deg2rad(phase), gamma)
    return values


def evaluate_parameters(x, y, m, parameters):
    return evaluate(x, y, m, parameters["amplitude"], parameters["b"],
                    parameters["phase"], parameters["gamma"], parameters["reference"])


def reference_label(parameters):
    return "Independent analytic solution" if parameters["reference"] == "analytic" else r"JAXtronomy ($\gamma=2$)"


def equality_expected(parameters):
    return parameters["reference"] == "analytic" or parameters["gamma"] == 2.0


def normalized_residual(actual, reference):
    scale = np.max(np.abs(reference), axis=-1, keepdims=True)
    return (actual - reference) / np.maximum(scale, 1e-30)


def metrics(actual, reference, check_equality=True):
    residual = normalized_residual(actual, reference)
    result = {}
    for name, a, r, diff in zip(QUANTITIES, actual, reference, residual):
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
            raise ValueError("Non-finite comparison values")
        if check_equality:
            np.testing.assert_allclose(a, r, rtol=0.0,
                                       atol=2e-7 * max(float(np.max(np.abs(r))), 1e-30))
        result[name] = dict(max_absolute_error=float(np.max(np.abs(a-r))),
                            max_normalized_error=float(np.max(np.abs(diff))))
    return result


def residual_multiplier(maximum):
    """Scale tiny numerical residuals without clipping physical differences."""
    return 10.0**max(0, int(np.floor(-np.log10(max(maximum, 1e-16)))))


def export(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def angular_figure(output, parameters):
    theta = np.linspace(0.0, 2.0*np.pi, 721)
    degrees = np.rad2deg(theta)
    x, y = parameters["ring_radius"]*np.cos(theta), parameters["ring_radius"]*np.sin(theta)
    fig = plt.figure(figsize=(13.2, 7.6))
    outer = fig.add_gridspec(2, 4, left=.075, right=.985, bottom=.12, top=.78,
                            hspace=.44, wspace=.32)
    all_metrics = {}
    comparisons = [evaluate_parameters(x, y, m, parameters) for m in (3, 4)]
    residual_max = max(np.max(np.abs(normalized_residual(*pair))) for pair in comparisons)
    multiplier = residual_multiplier(residual_max)
    for row, m in enumerate((3, 4)):
        ours, reference = comparisons[row]
        residual = normalized_residual(ours, reference)
        all_metrics[f"m{m}"] = metrics(ours, reference, equality_expected(parameters))
        for col, label in enumerate(LABELS):
            inner = outer[row, col].subgridspec(2, 1, height_ratios=(3, 1), hspace=.10)
            ax = fig.add_subplot(inner[0])
            err = fig.add_subplot(inner[1], sharex=ax)
            ax.plot(degrees, reference[col], color=ORANGE, lw=2.2, zorder=2)
            ax.plot(degrees, ours[col], color=BLUE, ls="--", lw=1.4, zorder=3)
            ax.plot(degrees[::36], ours[col, ::36], "o", color=BLUE, ms=3,
                    markerfacecolor="white", markeredgewidth=.8, zorder=4)
            ax.axhline(0, color="0.82", lw=.6, zorder=0)
            ax.set_ylabel(label)
            ax.set_title(rf"$m={m}$" + "  |  " + "abcd"[col], loc="left", fontsize=11)
            ax.tick_params(labelbottom=False)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
            # Use a common scale across quantities and orders; this avoids
            # visually exaggerating the difference between tiny residuals.
            err.plot(degrees, residual[col]*multiplier, color=BLUE, lw=1.0)
            err.axhline(0, color="0.65", lw=.65)
            limit = max(1.25, residual_max*multiplier*1.15)
            err.set_ylim(-limit, limit)
            err.set_yticks((-1, 0, 1))
            err.set_ylabel(r"$\epsilon$" if multiplier == 1 else rf"$\epsilon\times10^{{{int(np.log10(multiplier))}}}$", fontsize=9)
            err.set_xlim(0, 360)
            err.set_xticks((0, 90, 180, 270, 360))
            err.tick_params(labelsize=8)
            if row == 1:
                err.set_xlabel(r"Polar angle $\theta$ (deg)")
    title = "MPPL vs independent analytic solution" if parameters["reference"] == "analytic" else "MPPL vs JAXtronomy: circular multipoles"
    fig.suptitle(title, fontsize=17, y=.97)
    fig.text(.5, .916,
             rf"MPPL: $\gamma={parameters['gamma']:g}$, $a_m={parameters['amplitude']:g}$, "
             rf"$b={parameters['b']:g}$ arcsec, $\phi_m={parameters['phase']:g}^\circ$, "
             rf"$r={parameters['ring_radius']:g}$ arcsec", ha="center", fontsize=11)
    handles = [Line2D([], [], color=BLUE, ls="--", marker="o", markerfacecolor="white",
                      markersize=4, label="Our MPPL"),
               Line2D([], [], color=ORANGE, lw=2.2, label=reference_label(parameters))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .90),
               ncol=2, frameon=False, fontsize=11)
    comparison_note = "Equality check" if equality_expected(parameters) else "Different models: JAXtronomy multipoles remain isothermal"
    fig.text(.5, .031, r"$\epsilon=(f_{\rm ours}-f_{\rm ref})/\max_\theta|f_{\rm ref}|$"
             "   |   " + comparison_note + "   |   float64; perturbations only", ha="center", fontsize=10)
    export(fig, output, "01_angular_comparison")
    return all_metrics


def spatial_figure(output, parameters):
    coord = np.linspace(-2.0, 2.0, 260)  # even grid avoids the exact centre
    xx, yy = np.meshgrid(coord, coord)
    valid = np.hypot(xx, yy) >= parameters["minimum_radius"]
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.8))
    fig.subplots_adjust(left=.075, right=.92, bottom=.17, top=.83, hspace=.42, wspace=.27)
    values, all_metrics = [], {}
    for m in (3, 4):
        ours, ref = evaluate_parameters(xx[valid], yy[valid], m, parameters)
        all_metrics[f"m{m}"] = metrics(ours, ref, equality_expected(parameters))
        values.append((ours[3], ref[3]))
    maximum = max(np.max(np.abs(field)) for pair in values for field in pair)
    residual_max = max(np.max(np.abs(ours-ref))/np.max(np.abs(ref)) for ours, ref in values)
    multiplier = residual_multiplier(residual_max)
    for row, ((ours, ref), m) in enumerate(zip(values, (3, 4))):
        fields = (ours, ref, (ours-ref)/np.max(np.abs(ref)))
        for col, field in enumerate(fields):
            array = np.full(xx.shape, np.nan)
            array[valid] = field if col < 2 else field*multiplier
            limit = maximum if col < 2 else max(residual_max*multiplier, 1e-12)
            im = axes[row, col].imshow(array, origin="lower", extent=(-2, 2, -2, 2),
                                      cmap="RdBu_r", vmin=-limit, vmax=limit,
                                      interpolation="nearest")
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xticks((-2, -1, 0, 1, 2))
            axes[row, col].set_yticks((-2, -1, 0, 1, 2))
            axes[row, col].set_xlabel("x (arcsec)")
            if col == 0:
                axes[row, col].set_ylabel(rf"$m={m}$" + "\ny (arcsec)")
            axes[row, col].set_title(("Our MPPL", reference_label(parameters), "Normalized difference")[col],
                                     fontsize=11)
            if col == 1:
                density_image = im
            if col == 2:
                residual_image = im
                axes[row, col].text(.04, .04,
                    rf"max $|\epsilon|$ = {np.max(np.abs(fields[2])):.2e}",
                    transform=axes[row, col].transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=3))
    density_cax = fig.add_axes([.355, .065, .28, .015])
    density_bar = fig.colorbar(density_image, cax=density_cax, orientation="horizontal")
    density_bar.set_label(r"Density perturbation $\delta\kappa_m$", fontsize=10)
    residual_cax = fig.add_axes([.943, .17, .015, .60])
    residual_bar = fig.colorbar(residual_image, cax=residual_cax)
    residual_label = r"$(\kappa_{\rm ours}-\kappa_{\rm ref})/\max|\kappa_{\rm ref}|$"
    if multiplier != 1:
        residual_label += rf"$\times10^{{{int(np.log10(multiplier))}}}$"
    residual_bar.set_label(residual_label, fontsize=10)
    fig.suptitle("Spatial comparison of the density perturbation", fontsize=17, y=.975)
    fig.text(.5, .925, rf"$m=3,4$ only  |  $\gamma={parameters['gamma']:g}$  |  $a_m={parameters['amplitude']:g}$  |  "
             rf"$b={parameters['b']:g}$ arcsec  |  $\phi_m={parameters['phase']:g}^\circ$",
             ha="center", fontsize=11)
    fig.text(.5, .882, rf"White centre: $r<{parameters['minimum_radius']:g}$ arcsec excluded; "
             "density panels share one colour scale; differences use a separate scale.",
             ha="center", fontsize=10, color="0.3")
    export(fig, output, "02_density_comparison")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gamma", type=float, default=2.0, help="MPPL density slope (1 < gamma < 3)")
    parser.add_argument("--reference", choices=("jaxtronomy", "analytic"), default="jaxtronomy")
    args = parser.parse_args()
    if not 1.0 < args.gamma < 3.0:
        parser.error("--gamma must satisfy 1 < gamma < 3")
    if args.output is None:
        suffix = "" if args.gamma == 2.0 and args.reference == "jaxtronomy" else f"_{args.reference}_gamma{args.gamma:g}"
        args.output = ROOT / f"results/mppl_jaxtronomy_comparison{suffix}"
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    parameters = dict(amplitude=.025, b=.63, phase=18.0, gamma=args.gamma,
                      ring_radius=1.0, minimum_radius=.15, reference=args.reference)
    report = dict(parameters=parameters, orders=[3, 4], precision="float64",
        amplitude_mapping="jaxtronomy.a_m = b * MPPL.a_m",
        residual_definition="(ours - reference) / max(abs(reference)) per quantity and order",
        versions={name: importlib.metadata.version(name) for name in ("jax", "jaxtronomy", "numpy", "matplotlib")},
        angular=angular_figure(args.output, parameters),
        spatial=spatial_figure(args.output, parameters),
        equality_expected=equality_expected(parameters),
        limitation="JAXtronomy equality applies to gamma=2; analytic equality applies throughout 1 < gamma < 3, away from the softened centre.")
    source_paths = [Path(inspect.getfile(MPPL)), Path(inspect.getfile(Multipole)), Path(__file__)]
    report["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report["passed"] = True if equality_expected(parameters) else None
    report["status"] = "equality_checks_passed" if equality_expected(parameters) else "different_models_compared"
    (args.output/"comparison_metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "angular": report["angular"],
                      "spatial": report["spatial"]}, indent=2))


if __name__ == "__main__":
    main()
