"""Joint-posterior intervals for the human-readable lens-mass JSON product."""
from collections import Counter
from copy import deepcopy

import numpy as np


def _angle_period(key, original):
    if key in {'PA_deg', 'phi_ref', 'phi_ref_deg'}:
        return 180.
    if key in {'delta_phi_m1', 'delta_phi_m3', 'delta_phi_m4'}:
        return 360. / int(key[-1])
    if key in {'phi_m', 'phi_m_deg', 'varphi_m', 'varphi_m_deg',
               'delta_varphi_m', 'delta_varphi_m_deg'}:
        m = original.get('m')
        return 360. / float(m) if m is not None and float(m) > 0 else None
    return None


def _critical_area(grid, fields, center):
    """Apply the point-summary's closed/tangential/enclosing-curve criteria.

    Use the same determinant-zero contours, without tracing unused caustics.
    Eigenvalues interpolated from the grid distinguish radial critical curves.
    """
    from matplotlib.path import Path
    from scipy.ndimage import map_coordinates
    from skimage.measure import find_contours

    determinant, tangential, radial = fields
    nx, ny = grid.num_pixel_axes
    areas = []
    for contour in find_contours(determinant, 0.):
        if len(contour) < 4 or not np.all(np.isfinite(contour)):
            continue
        y, x = contour.T
        if np.linalg.norm(contour[0]-contour[-1]) > 1.5:
            continue
        if x.min() < 1 or x.max() > nx-2 or y.min() < 1 or y.max() > ny-2:
            continue
        xx, yy = map(np.asarray, grid.map_pix2coord(x, y))
        if not Path(np.column_stack((xx, yy)), closed=True).contains_point(center):
            continue
        sample = np.linspace(0, len(contour)-1, min(16,len(contour)), endpoint=False, dtype=int)
        coordinates = contour[sample].T
        t = map_coordinates(tangential, coordinates, order=1, mode='nearest')
        r = map_coordinates(radial, coordinates, order=1, mode='nearest')
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)):
            continue
        if np.median(np.abs(t)) > np.median(np.abs(r)):
            continue
        area = .5*abs(np.dot(xx,np.roll(yy,-1))-np.dot(yy,np.roll(xx,-1)))
        if np.isfinite(area) and area > 0:
            areas.append(float(area))
    return max(areas) if areas else None


def add_posterior_mass_intervals(
    lens_image, reference_summary, samples, kwargs_lens_from_params, *,
    batch_size=32, probability=0.682689492137,
):
    """Transform every joint draw, then compute equal-tail credible intervals.

    Numeric central values become posterior medians. Their old values remain
    in ``reference_model``. Errors are offsets from each interval's own median,
    including arrays and periodic angles; fixed quantities have zero errors.
    Only one device batch of critical-curve fields is retained at a time.
    """
    import jax
    import jax.numpy as jnp
    from .api.mass_posterior import posterior_interval
    from .visualizations import lens_mass_ellipticity_summary

    if isinstance(batch_size,bool) or not isinstance(batch_size,(int,np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    lengths = {len(value) for value in samples.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError('HMC mass intervals require nonempty, consistently sized posterior arrays.')
    if not 0 < probability < 1:
        raise ValueError('probability must lie between 0 and 1.')
    n = lengths.pop()
    summary = deepcopy(reference_summary)
    summary['reference_model'] = deepcopy(reference_summary)
    types = [profile['profile'] for profile in summary['profiles']]
    records = [dict(original_parameters={},converted_parameters={}) for _ in types]
    einstein = summary['einstein_radius']
    supersampling = einstein.get('supersampling',5)
    failures = Counter()
    areas = []
    try:
        grid = lens_image.Grid.create_model_grid(pixel_scale_factor=1./supersampling)
        x_grid,y_grid = map(jnp.asarray,grid.pixel_coordinates)
        def critical_fields(kwargs):
            fxx,fxy,fyx,fyy = lens_image.MassModel.hessian(x_grid,y_grid,kwargs)
            kappa = .5*(fxx+fyy)
            shear = jnp.hypot(.5*(fxx-fyy),fxy)
            return ((1-fxx)*(1-fyy)-fxy*fyx,1-kappa-shear,1-kappa+shear)
        evaluate_fields = jax.jit(jax.vmap(critical_fields))
        grid_error = None
    except Exception as error:
        grid_error = str(error)
    materialize = jax.jit(jax.vmap(kwargs_lens_from_params))
    halo_index = next((i for i,t in enumerate(types) if t.upper() in {'NFW','NFW_ELLIPSE_KAPPA'}),0)

    for start in range(0,n,batch_size):
        stop = min(start+batch_size,n)
        kwargs_batch = materialize({k:jnp.asarray(v[start:stop]) for k,v in samples.items()})
        host_kwargs = jax.tree_util.tree_map(np.asarray,kwargs_batch)
        try:
            if grid_error is not None:
                raise ValueError(grid_error)
            fields = tuple(np.asarray(value) for value in evaluate_fields(kwargs_batch))
            field_error = None
        except Exception as error:
            fields,field_error = None,str(error)
        for row in range(stop-start):
            kwargs = jax.tree_util.tree_map(lambda value:value[row],host_kwargs)
            converted = lens_mass_ellipticity_summary(
                lens_image,{'kwargs_lens':kwargs},include_einstein_radius=False,
            )
            for index,profile in enumerate(converted['profiles']):
                for section in ('original_parameters','converted_parameters'):
                    for key,value in profile[section].items():
                        if value is None or np.asarray(value).dtype.kind not in 'biuf':
                            continue
                        records[index][section].setdefault(key,[]).append(value)
            if fields is None:
                failures['calculation_failed: '+field_error] += 1
                continue
            reference = kwargs[halo_index]
            center = (float(reference.get('center_x',0.)),float(reference.get('center_y',0.)))
            area = _critical_area(grid,tuple(value[row] for value in fields),center)
            if area is None:
                failures['no_closed_tangential_curve_in_grid'] += 1
            else:
                areas.append(area)
        if start == 0 or stop == n or stop//1000 != start//1000:
            print(f'[lens_mass_parameters] Posterior mass/critical curves: {stop}/{n}')

    for profile,record in zip(summary['profiles'],records):
        profile['uncertainties'] = {}
        for section,quantities in record.items():
            intervals = {}
            for key,values in quantities.items():
                if len(values) != n:
                    raise ValueError(f'Cannot summarize {profile["profile"]}.{key}: missing posterior values.')
                interval = posterior_interval(
                    values,probability=probability,
                    period=_angle_period(key,profile['original_parameters']),
                )
                intervals[key] = interval
                profile[section][key] = interval['median']
            profile['uncertainties'][section] = intervals
    einstein['uncertainties'] = {}
    if areas:
        for key,values in (
            ('area_arcsec2',np.asarray(areas)),
            ('theta_E_eff_arcsec',np.sqrt(np.asarray(areas)/np.pi)),
            ('R_E_eff_arcsec',np.sqrt(np.asarray(areas)/np.pi)),
        ):
            interval = posterior_interval(values,probability=probability)
            einstein[key] = interval['median']
            einstein['uncertainties'][key] = interval
        einstein['status'] = 'ok'
        einstein.pop('reason',None)
    else:
        for key in ('theta_E_eff_arcsec','R_E_eff_arcsec','area_arcsec2'):
            einstein[key] = None
            einstein['uncertainties'][key] = None
        einstein['status'] = 'posterior_unavailable'
    einstein['posterior'] = {
        'valid_draws':len(areas),'failed_draws':n-len(areas),'failure_counts':dict(failures),
        'conditional_on_valid_curve':bool(areas and len(areas)<n),
        'contour_method':'determinant=0; tangential eigenvalue classification; fully enclosed curve',
    }
    summary['posterior'] = {
        'method':'equal_tailed','probability':probability,'used_draws':n,
        'central_values':'posterior_median','error_definition':'median-lower and upper-median',
        'reference_model':'original summary evaluated at supplied kwargs',
    }
    return summary
