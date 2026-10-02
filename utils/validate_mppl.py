#!/usr/bin/env python3
"""Audit the actual MPPL implementation and export explanatory figures.

Run in the modelling environment:
    python utils/validate_mppl.py --output results/mppl_validation

The checks use independent analytic density/derivative formulas, finite
differences, lenstronomy's isothermal circular multipole, and a sampled wrapper
model. Figures evaluate the registered Herculens profiles, not a cartoon.
Known boundary issues are reported separately from the passing checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import exprel

from herculens_wrapper.profiles.multipole import MPPL
from herculens_wrapper.profiles.registry import register_mass_profiles
from herculens.MassModel.mass_model import MassModel


def analytic(x, y, *, m, a_m, phi_m, gamma, b, center_x=0., center_y=0.):
    """Unsoftened Poisson solution; polar tensor rotated into Cartesian axes."""
    x, y = np.asarray(x) - center_x, np.asarray(y) - center_y
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    p = 3. - gamma
    co, si = np.cos(m * (theta - phi_m)), np.sin(m * (theta - phi_m))
    if m == 1:
        c = a_m * b ** (gamma - 1.) * p / (p + 1.)
        integral = np.log(r) * exprel((p - 1.) * np.log(r))
        psi = c * r * integral * co
        ar, at = c * (integral + r ** (p - 1.)) * co, -c * integral * si
        rr, rt, tt = c * p * r ** (p - 2.) * co, -c * r ** (p - 2.) * si, c * r ** (p - 2.) * co
    else:
        c = a_m * b ** (2. - p) * p / (p * p - m * m)
        psi = c * r ** p * co
        ar, at = c * p * r ** (p - 1.) * co, -c * m * r ** (p - 1.) * si
        rr = c * p * (p - 1.) * r ** (p - 2.) * co
        rt = -c * m * (p - 1.) * r ** (p - 2.) * si
        tt = c * (p - m * m) * r ** (p - 2.) * co
    ct, st = np.cos(theta), np.sin(theta)
    alpha = np.stack((ct * ar - st * at, st * ar + ct * at))
    hessian = np.stack((ct*ct*rr - 2.*ct*st*rt + st*st*tt,
                        st*st*rr + 2.*ct*st*rt + ct*ct*tt,
                        ct*st*(rr-tt) + (ct*ct-st*st)*rt))
    envelope = p / 2. * (b / r) ** (gamma - 1.)
    return psi, alpha, hessian, envelope * a_m * co, envelope


def normalized_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - expected)) /
                 max(float(np.max(np.abs(expected))), 1.e-15))


def audit():
    metrics = {}
    rng = np.random.default_rng(73)
    radii, theta = rng.uniform(.15, 2., 160), rng.uniform(-np.pi, np.pi, 160)
    cx, cy = .07, -.03
    x, y = radii*np.cos(theta)+cx, radii*np.sin(theta)+cy
    for m in (1, 3, 4):
        for gamma in (1.5, 1.9, 1.99999999, 2., 2.00000001, 2.1, 2.5):
            kw = dict(m=m, a_m=.037, phi_m=.31, gamma=gamma,
                      b=.63, center_x=cx, center_y=cy)
            psi, alpha, h, kappa, envelope = analytic(x, y, **kw)
            actual_h = np.asarray(MPPL.hessian(x, y, **kw))
            key = f"m{m}_gamma{gamma}"
            metrics[key] = {
                "potential_relative_max_error": normalized_error(MPPL.function(x, y, **kw), psi),
                "deflection_relative_max_error": normalized_error(MPPL.derivatives(x, y, **kw), alpha),
                "hessian_relative_max_error": normalized_error(actual_h, h),
                "kappa_error_over_perturbation_envelope": float(np.max(
                    np.abs((actual_h[0]+actual_h[1])/2.-kappa) / (envelope*kw['a_m']))),
            }
            assert max(metrics[key].values()) < 2.e-7, (key, metrics[key])
    # Finite differences independently check autodiff and the Hessian ordering.
    kw = dict(m=3, a_m=.03, phi_m=.21, gamma=2.12, b=.6)
    x0, y0, step = .53, -.41, 1.e-4
    f = lambda xx, yy: float(MPPL.function(xx, yy, **kw))
    grad_fd = [(f(x0+step,y0)-f(x0-step,y0))/(2*step),
               (f(x0,y0+step)-f(x0,y0-step))/(2*step)]
    h_fd = [(f(x0+step,y0)-2*f(x0,y0)+f(x0-step,y0))/step**2,
            (f(x0,y0+step)-2*f(x0,y0)+f(x0,y0-step))/step**2,
            (f(x0+step,y0+step)-f(x0+step,y0-step)-f(x0-step,y0+step)
             +f(x0-step,y0-step))/(4*step**2)]
    metrics['finite_difference'] = {
        'deflection_relative_max_error': normalized_error(grad_fd, np.asarray(MPPL.derivatives(x0,y0,**kw))),
        'hessian_relative_max_error': normalized_error(h_fd, np.asarray(MPPL.hessian(x0,y0,**kw))),
    }
    assert max(metrics['finite_difference'].values()) < 2.e-6
    from lenstronomy.LensModel.Profiles.multipole import Multipole
    other = Multipole()
    for m in (1,3,4):
        kw = dict(m=m, a_m=.025, phi_m=.24, gamma=2., b=.7)
        ref = dict(m=m, a_m=kw['a_m']*kw['b'], phi_m=kw['phi_m'], r_E=1.)
        hh = other.hessian(x,y,**ref)
        metrics[f'lenstronomy_m{m}'] = {
            'potential_relative_max_error': normalized_error(MPPL.function(x,y,**kw), other.function(x,y,**ref)),
            'deflection_relative_max_error': normalized_error(MPPL.derivatives(x,y,**kw), np.asarray(other.derivatives(x,y,**ref))),
            'hessian_relative_max_error': normalized_error(MPPL.hessian(x,y,**kw), np.stack((hh[0],hh[3],hh[1]))),
        }
        assert max(metrics[f'lenstronomy_m{m}'].values()) < 2.e-7
    # Phase symmetry, rotation covariance, Cartesian amplitude equivalence.
    for m in (1,3,4):
        kw = dict(m=m,a_m=.037,phi_m=.31,gamma=2.1,b=.63)
        e = kw['a_m']/(2.-kw['a_m'])
        exy = {k:v for k,v in kw.items() if k not in ('a_m','phi_m')}
        exy.update(e_x=e*np.cos(m*kw['phi_m']),e_y=e*np.sin(m*kw['phi_m']))
        angle = .48
        rotated_x, rotated_y = x*np.cos(angle)-y*np.sin(angle), x*np.sin(angle)+y*np.cos(angle)
        h = np.asarray(MPPL.hessian(x,y,**kw))
        metrics[f'parameterization_m{m}'] = {
            'cartesian_hessian_error': normalized_error(MPPL.hessian(x,y,**exy),h),
            'periodic_hessian_error': normalized_error(MPPL.hessian(x,y,**{**kw,'phi_m':kw['phi_m']+2*np.pi/m}),h),
            'rotation_potential_error': normalized_error(MPPL.function(rotated_x,rotated_y,**{**kw,'phi_m':kw['phi_m']+angle}),np.asarray(MPPL.function(x,y,**kw))),
        }
        assert max(metrics[f'parameterization_m{m}'].values()) < 1.e-12
    metrics['wrapper_api'] = audit_api()
    # Inspect the repaired resonance and retain the unrelated m>1 centre audit.
    resonance = []
    for gamma in (1.99,1.99999,2.,2.00001,2.01):
        kw = dict(m=1,a_m=.02,phi_m=.2,gamma=gamma,b=.4)
        ax, ay = MPPL.derivatives(.5,.3,**kw)
        h = MPPL.hessian(.5,.3,**kw)
        dg = jax.grad(lambda g: MPPL.function(.5,.3,**{**kw,'gamma':g}))(gamma)
        resonance.append(dict(gamma=gamma,alpha_x=float(ax),alpha_y=float(ay),kappa=float((h[0]+h[1])/2),dpsi_dgamma=float(dg)))
    assert all(np.isfinite(row['dpsi_dgamma']) for row in resonance)
    assert abs(resonance[1]['alpha_x']-resonance[3]['alpha_x']) < 1.e-7
    centre = np.asarray(MPPL.derivatives(0.,0.,m=3,a_m=.02,phi_m=0.,gamma=2.,b=.4))
    return dict(provenance=dict(profile_source=str(ROOT/'herculens_wrapper/profiles/multipole.py'),
                profile_sha256=hashlib.sha256((ROOT/'herculens_wrapper/profiles/multipole.py').read_bytes()).hexdigest(),
                versions={name:importlib.metadata.version(name) for name in ('jax','numpy','lenstronomy','matplotlib')}),
                checks=metrics, checked_domain=dict(orders=[1,3,4],slopes=[1.5,1.9,1.99999999,2.,2.00000001,2.1,2.5],
                radius_arcsec=[.15,2.],precision='float64'),
                boundary_audit=dict(m1_near_gamma2=resonance,exact_centre_deflection_finite=bool(np.all(np.isfinite(centre)))),
                conclusion='Density and derivative checks pass, including the continuous m=1 solution across gamma=2; the exact centre for m>1 remains singular.')


def audit_api():
    from herculens_wrapper.api import (MassProfile, LightProfile, LensProfileCollection,
                                      SingleBandData, SingleBandModel)
    epl = MassProfile('EPL', prior=dict(theta_E=[.5,.8],gamma=[1.2,2.8],q=.75,
                      phi=17.,center_x=[-.07,.07],center_y=[-.07,.07]))
    terms = []
    for m in (1,3,4):
        term = MassProfile('MPPL',value=dict(m=m,a_m=.025,phi_m=36.))
        term.gamma=epl.gamma
        term.b=epl.theta_E
        term.center_x=epl.center_x
        term.center_y=epl.center_y
        terms.append(term)
    source = LightProfile('SERSIC_ELLIPSE',value=dict(amp=1.,R_sersic=.1,n_sersic=1.,
                  e1=0.,e2=0.,center_x=.02,center_y=.01))
    model = SingleBandModel(profiles=LensProfileCollection(lens_mass=[epl,*terms],source_light=source),
            observation=SingleBandData(image=np.zeros((7,7)),noise=np.ones((7,7)),psf=np.ones((1,1)),pixel_scale=.1))
    error = 0.
    for seed in (3,19):
        sample = model.prob_model.get_sample(jax.random.PRNGKey(seed))
        kw = model.prob_model.params2kwargs(sample)['kwargs_lens']
        for term in kw[1:]:
            for dest, src in [('gamma','gamma'),('b','theta_E'),('center_x','center_x'),('center_y','center_y')]:
                assert float(term[dest]) == float(kw[0][src])
            assert np.isclose(float(term['phi_m']),np.deg2rad(36.))
        xx, yy = jnp.asarray([.31,.7]),jnp.asarray([.48,-.37])
        measured = np.asarray(model.lens_image.MassModel.kappa(xx,yy,kw))
        base = np.asarray(MassModel(['EPL']).kappa(xx,yy,[kw[0]]))
        expected = base + sum(analytic(xx,yy,**term)[3] for term in kw[1:])
        error = max(error,normalized_error(measured,expected))
    assert error < 2.e-7
    return dict(sampled_link_checks=24,phase_degrees_to_radians=True,total_kappa_relative_max_error=error)


def model_arrays(q, gamma, amplitude, phase, order, grid):
    xx, yy = grid
    e = (1.-q)/(1.+q)
    base_kw = dict(theta_E=1.,gamma=gamma,e1=e,e2=0.,center_x=0.,center_y=0.)
    term_kw = dict(m=order,a_m=amplitude,phi_m=phase,gamma=gamma,b=1.,center_x=0.,center_y=0.)
    base = MassModel(['EPL'])
    delta_h = np.asarray(MPPL.hessian(xx,yy,**term_kw))
    return base_kw, term_kw, np.asarray(base.kappa(xx,yy,[base_kw])),(delta_h[0]+delta_h[1])/2.


def save_figure(fig, output, name):
    fig.savefig(output/f'{name}.png',dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def figures(output, amplitude):
    coord = np.linspace(-1.9,1.9,360)  # even grid: do not evaluate the singular centre
    xx, yy = np.meshgrid(coord,coord)
    radius = np.hypot(xx,yy)
    grid = (jnp.asarray(xx),jnp.asarray(yy))
    theta = np.linspace(0,2*np.pi,721)
    labels = ['m=1: lopsided','m=3: triangular','m=4: fourfold']
    plus, minus = '#167eab','#d15b35'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for col,m in enumerate((1,3,4)):
        kw = dict(m=m,a_m=amplitude,phi_m=0.,gamma=2.,b=1.)
        hh = np.asarray(MPPL.hessian(np.cos(theta),np.sin(theta),**kw))
        actual = (hh[0]+hh[1])  # kappa_circular(r=1) = 1/2
        ax = axes[0,col]
        ax.plot(np.degrees(theta),amplitude*np.cos(m*theta),color='0.35',lw=3,label='analytic target')
        ax.plot(np.degrees(theta[::12]),actual[::12],'o',ms=3,color=plus,label='actual MPPL Hessian')
        ax.axhline(0.,color='0.7',lw=.7)
        ax.set(xlim=(0,360),xticks=[0,90,180,270,360],xlabel='polar angle (deg)',ylabel=r'$\delta\kappa/\kappa_{\rm circ}$',title=labels[col])
        if col==0: ax.legend(fontsize=8)
        _,_,base,delta = model_arrays(1.,2.,amplitude,0.,m,grid)
        fraction = np.ma.masked_where(radius<.08,delta/base)
        im = axes[1,col].imshow(fraction,origin='lower',extent=[-1.9,1.9,-1.9,1.9],cmap='RdBu_r',vmin=-amplitude,vmax=amplitude)
        axes[1,col].set(title='Signed density perturbation',xlabel='x (arcsec)',ylabel='y (arcsec)')
        fig.colorbar(im,ax=axes[1,col],label=r'$\delta\kappa/\kappa_{\rm circ}$',shrink=.8)
        ax = axes[2,col]
        for total,color,style in [(base,'0.4','--'),(base+delta,plus,'-'),(base-delta,minus,'-')]:
            ax.contour(xx,yy,total,levels=[.5],colors=[color],linestyles=[style],linewidths=2.)
        ax.plot(0.,0.,'+',color='0.2',ms=6)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect='equal',xlabel='x (arcsec)',ylabel='y (arcsec)',title=r'Total density contour: $\kappa=0.5$')
        if col==0:
            ax.legend(handles=[Line2D([],[],ls='--',color='0.4',label='circular EPL'),Line2D([],[],color=plus,label=f'+a = {amplitude:g}'),Line2D([],[],color=minus,label=f'-a = {amplitude:g}')],fontsize=8,loc='lower left')
    fig.suptitle(f'Actual MPPL: angular density, signed perturbation and contour deformation\n'+r'$\gamma=2$, $b=\theta_E=1$ arcsec, $\phi_m=0$; illustrative amplitude',fontsize=13)
    save_figure(fig,output,'01_mppl_meaning')

    fig,axes=plt.subplots(3,3,figsize=(11,10),layout='constrained')
    for row,m in enumerate((1,3,4)):
        for col,q in enumerate((1.,.75,.5)):
            _,_,base,delta=model_arrays(q,2.,amplitude,0.,m,grid)
            ax=axes[row,col]
            for total,color,style in [(base,'0.4','--'),(base+delta,plus,'-'),(base-delta,minus,'-')]:
                ax.contour(xx,yy,total,levels=[.5],colors=[color],linestyles=[style],linewidths=2.)
            ax.plot(0.,0.,'+',color='0.3',ms=5)
            ax.set(xlim=(-1.65,1.65),ylim=(-1.65,1.65),aspect='equal',xlabel='x (arcsec)',ylabel='y (arcsec)',title=f'm={m}, EPL q={q:g}')
    axes[0,0].legend(handles=[Line2D([],[],ls='--',color='0.4',label='EPL'),Line2D([],[],color=plus,label='+a'),Line2D([],[],color=minus,label='-a')],fontsize=8)
    fig.suptitle(f'Circular MPPL added to elliptical EPL: shape depends on q\n'+rf'$\kappa=0.5$, $\gamma=2$, $\phi_m=0$, $|a_m|={amplitude:g}$; fixed centre',fontsize=13)
    save_figure(fig,output,'02_mppl_on_ellipses')

    # Controlled unblurred lensing experiment: fixed source and other mass terms.
    ray_amplitude=.03
    e=(1.-.75)/(1.+.75)
    epl=dict(theta_E=1.,gamma=2.,e1=e,e2=0.,center_x=0.,center_y=0.)
    model=MassModel(['EPL','MPPL'])
    base_alpha=MassModel(['EPL']).alpha(*grid,[epl])
    def source(bx,by):
        return np.exp(-.5*((np.asarray(bx)-.05)**2/.08**2+(np.asarray(by)-.035)**2/.055**2))
    baseline=source(xx-base_alpha[0],yy-base_alpha[1])
    fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    images=[baseline]
    for m in (1,3,4):
        mp=dict(m=m,a_m=ray_amplitude,phi_m=.2,gamma=2.,b=1.,center_x=0.,center_y=0.)
        alpha=model.alpha(*grid,[epl,mp])
        images.append(source(xx-alpha[0],yy-alpha[1]))
    vmax=max(float(np.max(np.abs(im-baseline))) for im in images[1:])
    titles=['EPL only','EPL + m=1','EPL + m=3','EPL + m=4']
    for col,im in enumerate(images):
        artist=axes[0,col].imshow(im,origin='lower',extent=[-1.9,1.9,-1.9,1.9],cmap='inferno',vmin=0.,vmax=1.)
        diff=axes[1,col].imshow(im-baseline,origin='lower',extent=[-1.9,1.9,-1.9,1.9],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes[0,col].set(title=titles[col],xlabel='x (arcsec)',ylabel='y (arcsec)')
        axes[1,col].set(title='Image difference from EPL',xlabel='x (arcsec)',ylabel='y (arcsec)')
    fig.colorbar(artist,ax=list(axes[0]),shrink=.7,label='surface brightness (same source)')
    fig.colorbar(diff,ax=list(axes[1]),shrink=.7,label='signed surface brightness difference')
    fig.suptitle(r'Controlled ray tracing: $q=0.75$, $\gamma=2$, $a_m=0.03$, $\phi_m=0.2$ rad'+ '\nFixed Gaussian source; no PSF, noise, shear or refitting',fontsize=13)
    save_figure(fig,output,'03_mppl_lensed_images')

    # Display the original discontinuity and the repaired HMC slope derivative.
    gammas=np.linspace(1.9999,2.0001,301)
    def alpha_x(gamma):
        return MPPL.derivatives(.5,.3,m=1,a_m=.02,phi_m=.2,gamma=gamma,b=.4)[0]
    repaired=np.asarray(jax.jit(jax.vmap(alpha_x))(jnp.asarray(gammas)))
    slope=np.asarray(jax.jit(jax.vmap(jax.grad(alpha_x)))(jnp.asarray(gammas)))
    old=[]
    r=np.hypot(.5,.3)
    theta=np.arctan2(.3,.5)
    for gamma in gammas:
        p=3.-gamma
        if abs(gamma-2.)<1.e-12:
            old.append(.02*.4/2.*(np.cos(.2)*np.log(r)+np.cos(theta-.2)*np.cos(theta)))
        else:
            c=.02*.4**(gamma-1.)*p/(p*p-1.)
            old.append(c*r**(p-1.)*(p*np.cos(theta-.2)*np.cos(theta)+np.sin(theta-.2)*np.sin(theta)))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    offset=(gammas-2.)*1.e5
    axes[0].plot(offset,old,color=minus,label='previous gauge')
    axes[0].plot(offset,repaired,color=plus,lw=2,label='continuous gauge')
    axes[0].set_yscale('symlog',linthresh=.01)
    axes[0].set(xlabel=r'$(\gamma-2)\times10^5$',ylabel=r'$\alpha_x$ (arcsec)',title='Constant-deflection divergence removed')
    axes[0].legend(fontsize=9)
    axes[1].plot(offset,slope,color=plus,lw=2)
    axes[1].axvline(0.,color='0.6',ls='--',lw=1)
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    axes[1].set(xlabel=r'$(\gamma-2)\times10^5$',ylabel=r'$\partial\alpha_x/\partial\gamma$',title='Finite, continuous slope gradient')
    fig.suptitle(r'Repaired MPPL $m=1$: $a_1=0.02$, $b=0.4$ arcsec, $\phi_1=0.2$ rad; at $(0.5,0.3)$ arcsec',fontsize=11)
    save_figure(fig,output,'04_mppl_free_gamma')


def write_report(output, result, amplitude):
    checks=result['checks']
    rows=[]
    for key,values in checks.items():
        errors=[v for k,v in values.items() if 'error' in k]
        rows.append(f'| {key} | {max(errors):.3e} |')
    boundary=result['boundary_audit']
    near_rows=[f"| {r['gamma']:.5f} | {r['alpha_x']:.9g} | {r['alpha_y']:.9g} | {r['kappa']:.8g} | {r['dpsi_dgamma']:.9g} |" for r in boundary['m1_near_gamma2']]
    report=r'''# MPPL 验证与展示

这次验证读取修复后的项目实际 `MPPL`。m=1 已采用随 gamma 连续的势函数规范；
原拟合脚本与已有拟合结果没有被修改。

## 数学定义

以同中心的普通极坐标 $(r,\theta)$ 定义 $p=3-\gamma$。目标扰动为

$$\delta\kappa_m=\frac p2(b/r)^{\gamma-1}a_m\cos[m(\theta-\phi_m)].$$

这对应 [Enzi et al. (2025), Eq. 7](https://academic.oup.com/mnras/article/540/1/247/8123410)
中的圆形 EPL 密度基底；链接 $b=\theta_E$ 后，`a_m` 为文献的无量纲振幅。
它不是相对于实际椭圆 EPL 每个位置的相同百分比扰动。

当 $p^2\ne m^2$ 时，原始非共振势函数为

$$\psi_m=\frac{p a_m b^{\gamma-1}}{p^2-m^2}r^p\cos[m(\theta-\phi_m)].$$

由 $\nabla^2[r^p\cos(m\theta)]=(p^2-m^2)r^{p-2}\cos(m\theta)$，
立即得到 $\nabla^2\psi_m/2=\delta\kappa_m$。
上式是原始非共振解，仍用于 m=3,4。修复后的 m=1 使用

$$\psi_1=a_1b^{\gamma-1}r\cos(\theta-\phi_1)\frac{p}{p+1}
\log(r)\,\mathrm{exprel}[(p-1)\log(r)],$$

其中 exprel(z)=expm1(z)/z，exprel(0)=1。
这相当于从原始解减去 $a_1b^{\gamma-1}p/(p^2-1)\,r\cos(\theta-\phi_1)$，
这个线性势的拉普拉斯与 Hessian 都为零。
在等温 $m=1$ 情形连续回到 $\psi_1=(a_1b/2)r\log(r)\cos(\theta-\phi_1)$，
空间单位采用 arcsec；对数隐含 1 arcsec 参考尺度。
实现采用 Cartesian 投影 x*cos(phi_1)+y*sin(phi_1)，对数内使用软化半径，
避免把软化的 r*cos(theta) 误当成精确的线性函数。
代码在半径中加入 $10^{-10}$ arcsec²，因此以上无软化解析式只在远离中心时相符。

形状也可以从密度直接推出。圆形 EPL 加单个 MPPL、gamma=2 时，
其 kappa=0.5 的等密度线为

$$r(\theta)=b[1+a_m\cos(m(\theta-\phi_m))].$$

因此一圈出现 m 个鼓出方向与 m 个收缩方向。m=1 的形变可能与中心位置存在退化，
但这个完整的径向密度扰动并不等于把整个质量模型平移。
对于椭圆 EPL，令 $t=\gamma-1$ 与
$f_q(\theta)=\sqrt{q\cos^2\theta+\sin^2\theta/q}$（坐标轴已沿长轴对齐），则

$$r_{\kappa_0}(\theta)=b\left[\frac{3-\gamma}{2\kappa_0}
\{f_q(\theta)^{-t}+a_m\cos[m(\theta-\phi_m)]\}\right]^{1/t}.$$

这要求括号内为正，且 EPL 与扰动共享中心、gamma 与 b=theta_E。
它解释了为什么同一个圆形 MPPL 加到不同 q 的 EPL 上，会得到不同的形状。

`e_x/e_y` 的定义也与文献一致：令 $e=\sqrt{e_x^2+e_y^2}$，
则 $a_m=2e/(1+e)$，$\phi_m=\mathrm{atan2}(e_y,e_x)/m$。
等价地，角向项为 $2[e_x\cos(m\theta)+e_y\sin(m\theta)]/(1+e)$。
这次同时检验了两种参数化、相位周期与整体旋转。

## 数值检查

160 个随机位置，中心有偏移，半径 0.15–2 arcsec；m=1,3,4；
gamma=1.5,1.9,1.99999999,2,2.00000001,2.1,2.5；float64。
势、偏折、Hessian 对比独立极坐标解析公式；收敛对比目标密度。
收敛误差以扰动包络归一化，避免在余弦零点计算无意义的相对误差。
另外使用中心有限差分验证 autodiff，并与外部 lenstronomy 的 gamma=2 解比较。
lenstronomy 的 `a_m` 是有角度单位的振幅，比较时必须换为 `b * MPPL.a_m`。

公共 API 测试用真实 SingleBandModel、两个随机采样：确认 `phi_m=36` 度
转换成弧度，三个扰动的 gamma、b、中心均随 EPL 更新，Herculens 总收敛等于
EPL 加三个解析扰动。可复现实验在 `utils/validate_mppl.py`。

| 检查 | 最大归一化误差 |
|---|---:|
'''+ '\n'.join(rows)+r'''

## gamma=2 附近的修复与兼容性

**m=1 的自由 gamma 穿过 2：修复后偏折与 gamma 梯度连续。**
旧的非共振解的系数包含 $1/(p^2-1)$，在 $p\to1$ 时会出现发散的
$C r\cos(\theta-\phi_1)$。这是线性势项，只产生常量偏折，Hessian 为零。
单透镜平面、源位置可平移时，它可以被源平移吸收。旧代码只在精确 gamma=2
切换为对数解，没有在两侧移除这个项。修复已统一为上面的连续规范，
exprel 的 Taylor 展开同时保留精确 gamma=2 处正确的非零 gamma 梯度。

例：m=1, a=0.02, phi=0.2 rad, b=0.4 arcsec，评价位置 (0.5,0.3) arcsec：

| gamma | alpha_x (arcsec) | alpha_y (arcsec) | delta kappa | dpsi/dgamma |
|---:|---:|---:|---:|---:|
'''+ '\n'.join(near_rows)+r'''

上表是修复后的值；旧实现曾在 gamma=1.99999 时给出 alpha_x≈392 arcsec，
而精确 gamma=2 时为 0.00111853 arcsec。现在两侧都趋近这个有限值。
`tests/test_mppl_free_gamma.py` 覆盖独立数值积分、密度、空间有限差分、gamma 有限差分、
零点的解析 gamma 导数、e_x/e_y 与 offset 接口、m=3,4 兼容性，
以及真实 wrapper 中供 HMC 使用的无约束后验梯度。

旧的非等温 m=1 解与新解的偏折相差常量
$\mathbf C=a_1b^{\gamma-1}p/(p^2-1)(\cos\phi_1,\sin\phi_1)$。
保留旧图像时，源坐标需要同步变为 beta_new=beta_old+C；
固定有限源网格／源位置 prior 时不能直接沿用旧坐标。建议新采样重新初始化。
精确 gamma=2 的旧规范保留，仅有中心软化实现的微小变化；原结果文件没有改写。
你的 `cowls_data/modelling_F277W_EPL_multipoles/modelling_F277W_EPL_multipoles.py`
确实使用共享 gamma 的 m=1，且 gamma prior 为 [1.2,2.8]；这一边界与该用法直接相关。

**精确中心：** m=1 的 Cartesian 表达式使软化中心的偏折有限。
m=3,4 的 `atan2(0,0)` 空间导数仍导致非有限偏折，此次没有修改这些阶数的中心处理。
展示与密度检查均避开精确中心。

## 怎样读图

1. `01_mppl_meaning.png`：顶行为解析余弦与真实 Hessian 的密度响应；
   中行为有正有负的扰动；底行为相同 kappa=0.5 的总等密度线。
   蓝线正振幅、橙线负振幅、虚线基底、十字固定中心。
   m=1 为单侧变化，m=3 为三向变化，m=4 为四向变化。
2. `02_mppl_on_ellipses.png`：改变 EPL 的 q，显示相同圆形 MPPL 在不同椭圆上
   的形状。较扁时不能仅凭 a4 的符号称为 boxy/disky。参见
   [Paugnat & Gilman (2025), Sec. III](https://arxiv.org/html/2502.03530v2)。
3. `03_mppl_lensed_images.png`：只打开一个阶数，其他参数与源保持固定。
   上行为成像，下行为与纯 EPL 的图像差；采用实际 Herculens 偏折。
   振幅 0.03；无 PSF、噪声或重新拟合，这是响应展示，不是对实测数据的约束。
4. `04_mppl_free_gamma.png`：旧偏折的发散与修复后的连续解，以及新解的 gamma 梯度。

第一、二张图振幅为 '''+str(amplitude)+r'''，用于放大形变，不代表你的拟合值。
MPPL 的负密度区域是相对基底的密度减少；物理检查应针对总 kappa。
纯扰动的圆周均值为零，因此只画径向平均会把其结构抵消。

## 对实际拟合还需要验证什么

保持同一 PSF、mask、噪声、源正则化，用最佳拟合样本比较 full model 与逐项关闭 m1/m3/m4。
展示总 kappa 等密度线、signed delta kappa、full-minus-ablated 图像。
如重新拟合，则同时展示 residual 与 posterior，区分直接响应和其他参数吸收后的变化。
最终用注入已知 multipole 的 mock 数据检查恢复与退化；数学实现通过不能单独证明
实际数据能够约束每一阶。

复现命令：

```bash
/opt/anaconda3/envs/herculens/bin/python utils/validate_mppl.py --output results/mppl_validation
```

`validation.json` 同时记录 profile 文件 SHA-256 和依赖版本，便于之后比较不同实现。
'''
    (output/'REPORT.md').write_text(report,encoding='utf-8')
    (output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/mppl_validation')
    parser.add_argument('--amplitude',type=float,default=.12,help='Illustrative shape amplitude; ray-tracing uses 0.03.')
    args=parser.parse_args()
    if not 0.<args.amplitude<.3:
        parser.error('--amplitude must be between 0 and 0.3')
    args.output.mkdir(parents=True,exist_ok=True)
    register_mass_profiles()
    result=audit()
    figures(args.output,args.amplitude)
    write_report(args.output,result,args.amplitude)
    print(json.dumps(dict(output=str(args.output.resolve()),interior_checks='PASS',boundary_audit=result['boundary_audit']),indent=2))


if __name__=='__main__':
    main()
