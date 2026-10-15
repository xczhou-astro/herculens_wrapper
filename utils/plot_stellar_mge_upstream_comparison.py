#!/usr/bin/env python3
"""Plot saved StellarMGE outputs against the pinned official backend."""
from __future__ import annotations

import argparse
import importlib.util
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

from herculens.MassModel.mass_model import MassModel
from herculens_wrapper.profiles.registry import register_mass_profiles
from utils.validate_stellar_mge_upstream import direct_parameters, compare, sha256


def plot_all_run_errors(validation, output):
    quantities = ("potential","deflection","hessian","kappa")
    labels = (r"Potential $\psi$",r"Deflection $\boldsymbol{\alpha}$","Hessian",r"Convergence $\kappa$")
    colors = ("#0072B2","#D55E00","#009E73","#CC79A7")
    markers = ("o","s","^","D")
    fig,ax = plt.subplots(figsize=(8.4,4.4))
    fig.subplots_adjust(left=.12,right=.98,bottom=.22,top=.75)
    positions = np.arange(len(validation["runs"]))
    for j,(key,label,color,marker) in enumerate(zip(quantities,labels,colors,markers)):
        values = [r["checks"][key]["max_normalized_error"]*1e16 for r in validation["runs"]]
        ax.scatter(positions+(j-1.5)*.10,values,color=color,marker=marker,s=42,label=label,zorder=3)
    ax.axhline(np.finfo(np.float64).eps*1e16,color="0.5",ls=":",lw=1)
    ax.text(.99,np.finfo(np.float64).eps*1e16+.08,"float64 machine epsilon",
            transform=ax.get_yaxis_transform(),ha="right",fontsize=9,color="0.4")
    ax.set(xticks=positions,xticklabels=[r["run"] for r in validation["runs"]],
           ylabel=r"Maximum normalized difference $\times10^{16}$",ylim=(-.15,3.75))
    ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("All four runs: agreement with the official backend",fontsize=14,y=.97)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.90),ncol=4,
               frameon=False,fontsize=10)
    fig.text(.5,.08,r"$\max|f_{\rm wrapper}-f_{\rm official}|/\max|f_{\rm official}|$; zeros are shown as zero",
             ha="center",fontsize=10)
    fig.text(.5,.025,"Deflection, Hessian and convergence: 1681 image-grid points per run; potential: 32 points",
             ha="center",fontsize=9,color="0.3")
    for extension in ("png","pdf"):
        fig.savefig(output/f"all_run_errors.{extension}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, default=ROOT/"results/stellar_mge_upstream_validation")
    parser.add_argument("--run", default="run_0")
    args = parser.parse_args()
    output = args.validation_dir.resolve()
    validation = json.loads((output/"validation.json").read_text())
    source = output/"reference/multi_gaussian_ellipse_kappa.py"
    assert sha256(source) == validation["reference_sha256"]
    specification = importlib.util.spec_from_file_location("stellar_mge_plot_official_reference",source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    reference = module.MultiGaussianEllipseKappa()
    root = args.results_dir.resolve()
    if (root/"pixelated_svi").is_dir():
        root /= "pixelated_svi"
    path = root/args.run
    checked_run = next(r for r in validation["runs"] if r["run"] == args.run)
    assert sha256(path/"kwargs_result.json") == checked_run["result_sha256"]
    saved = json.loads((path/"kwargs_result.json").read_text())
    configuration = json.loads((path/"model_configuration.json").read_text())
    stellar_index = configuration["backend_definition"]["types"]["lens_mass_type_list"].index("STELLAR_MGE")
    stellar = saved["kwargs_lens"][stellar_index]
    indices = configuration["backend_definition"]["parameters"]["lens_mass_params_list"][stellar_index]["_stellar_lens_light_indices"]
    expected = direct_parameters([saved["kwargs_lens_light"][i] for i in indices],
                                 stellar["upsilon_kappa"],stellar["ml_gradient"])
    register_mass_profiles(); model = MassModel(["STELLAR_MGE"])
    x = np.linspace(-1.2,1.2,401); y = np.zeros_like(x)
    alpha = model.alpha(x,y,[stellar]); alpha_ref = reference.derivatives(x,y,**expected)
    hessian = model.hessian(x,y,[stellar]); hessian_ref = reference.hessian(x,y,**expected)
    actual = np.array([model.potential(x,y,[stellar]),alpha[0],alpha[1],
                       model.kappa(x,y,[stellar]),hessian[1]])
    ref = np.array([reference.function(x,y,**expected),alpha_ref[0],alpha_ref[1],
                    (hessian_ref[0]+hessian_ref[1])/2,hessian_ref[2]])
    names = ("potential","alpha_x","alpha_y","kappa","hessian_xy")
    metrics = {name:compare(a,b) for name,a,b in zip(names,actual,ref)}
    residual = (actual-ref)/np.maximum(np.max(np.abs(ref),axis=1,keepdims=True),1e-30)
    np.savez_compressed(output/"comparison_curves.npz",x_arcsec=x,y_arcsec=y,
                        wrapper=actual,official=ref,normalized_residual=residual,quantities=names)
    (output/"curve_metrics.json").write_text(json.dumps({"run":args.run,"slice":"y=0 arcsec",
        "points":len(x),"checks":metrics,"reference_commit":validation["reference_commit"],
        "result_sha256":checked_run["result_sha256"],"plot_script_sha256":sha256(__file__)},indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"axes.linewidth":.7})
    fig = plt.figure(figsize=(15.0,6.1))
    grid = fig.add_gridspec(2,5,left=.06,right=.986,bottom=.20,top=.75,
                           height_ratios=(3,1),hspace=.13,wspace=.35)
    labels = (r"$\psi$ (arcsec$^2$)",r"$\alpha_x$ (arcsec)",r"$\alpha_y$ (arcsec)",
              r"$\kappa$",r"$H_{xy}$")
    maximum = max(float(np.max(np.abs(residual))*1e16),.5)
    for col,(a,r,label) in enumerate(zip(actual,ref,labels)):
        ax = fig.add_subplot(grid[0,col]); err = fig.add_subplot(grid[1,col],sharex=ax)
        ax.plot(x,r,color="#D55E00",lw=2.3)
        ax.plot(x,a,color="#0072B2",ls="--",lw=1.35)
        ax.plot(x[::20],a[::20],"o",color="#0072B2",ms=3,markerfacecolor="white",markeredgewidth=.8)
        ax.axhline(0,color="0.85",lw=.6,zorder=0)
        ax.set(ylabel=label,title="abcde"[col],xlim=(-1.2,1.2))
        ax.tick_params(labelbottom=False)
        ax.set_xticks((-1.2,-.6,0,.6,1.2))
        err.plot(x,residual[col]*1e16,color="#0072B2",lw=.8)
        err.axhline(0,color="0.65",lw=.6)
        err.set_ylim(-1.15*maximum,1.15*maximum)
        err.set_ylabel(r"$\epsilon\times10^{16}$",fontsize=9)
        err.set_xlabel("x (arcsec)")
        err.tick_params(labelsize=9)
    fig.suptitle("Stellar MGE: our wrapper vs official Jax-Lensing-Profiles",fontsize=17,y=.98)
    fig.text(.5,.90,rf"Saved {args.run}: $U={stellar['upsilon_kappa']:.6f}$ arcsec$^2$, "
             rf"$g={stellar['ml_gradient']:g}$; 5 Gaussians; slice $y=0$ arcsec; stellar component only",
             ha="center",fontsize=11)
    fig.legend(handles=[Line2D([],[],color="#0072B2",ls="--",marker="o",markerfacecolor="white",
                               markersize=4,label="Our STELLAR_MGE wrapper"),
                        Line2D([],[],color="#D55E00",lw=2.3,label="Official MultiGaussianEllipseKappa")],
               loc="upper center",bbox_to_anchor=(.5,.862),ncol=2,frameon=False,fontsize=11)
    fig.text(.5,.085,r"$\epsilon=(f_{\rm wrapper}-f_{\rm official})/\max_x|f_{\rm official}|$"
             "   |   residuals enlarged by $10^{16}$; float64 rounding only",ha="center",fontsize=10)
    fig.text(.5,.032,"All four runs passed on the image grid; official reference commit "
             +validation["reference_commit"][:12],ha="center",fontsize=10,color="0.3")
    for extension in ("png","pdf"):
        fig.savefig(output/f"official_comparison.{extension}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    plot_all_run_errors(validation,output)
    report = output/"report.md"
    text = report.read_text()
    if "official_comparison.png" not in text:
        report.write_text(text+f"\n## 对比图\n\nrun_0，y=0 截面；残差乘以 10^16 显示。\n\n![Official comparison]({output/'official_comparison.png'})\n")
    if "all_run_errors.png" not in report.read_text():
        report.write_text(report.read_text()+f"\n![All run errors]({output/'all_run_errors.png'})\n")
    print(json.dumps({"figure":str(output/"official_comparison.png"),"checks":metrics},indent=2))


if __name__ == "__main__":
    main()
