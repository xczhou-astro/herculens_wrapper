#!/usr/bin/env python3
import os
import sys
import json
import argparse
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.segmentation import detect_sources, deblend_sources

# Ensure wrapper can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from herculens_wrapper.models import create_lens_image


def deblend_and_ray_trace(run_dir, threshold_frac=0.05, plot_scale='log', n_pixels=5, contrast=0.001, n_levels=32, band=None, thin=1):
    print(f"Loading outputs from run directory: {run_dir}")
    
    run_dir = os.path.abspath(run_dir)
    
    # Check if run_dir points to a band subfolder of a multiband run
    parent_dir = os.path.dirname(run_dir)
    parent_args_path = os.path.join(parent_dir, 'args.json')
    if not os.path.exists(os.path.join(run_dir, 'args.json')) and os.path.exists(parent_args_path):
        with open(parent_args_path, 'r') as f:
            parent_args = json.load(f)
        if parent_args.get('use_multiband', False) or 'band_names' in parent_args:
            band = os.path.basename(run_dir)
            run_dir = parent_dir
            print(f"Detected band subfolder. Rebasing run_dir to: {run_dir} and setting band to: {band}")
            
    # Load JSON settings
    args_path = os.path.join(run_dir, 'args.json')
    config_path = os.path.join(run_dir, 'config.json')
    result_path = os.path.join(run_dir, 'kwargs_result.json')
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Missing args.json in {run_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing kwargs_result.json in {run_dir}")
        
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    # Determine if this is multiband
    is_multiband = args_dict.get('use_multiband', False)
    band_names = args_dict.get('band_names', [])
    
    if is_multiband:
        if not band:
            if len(band_names) == 1:
                band = band_names[0]
            else:
                raise ValueError(
                    f"This is a multiband fit. Please specify which band to process using --band. "
                    f"Available bands: {band_names}"
                )
        if band not in band_names:
            raise ValueError(f"Specified band '{band}' not found in available bands: {band_names}")
        band_idx = band_names.index(band)
        print(f"Processing band: {band}")
    else:
        band_idx = 0

    # Reconstruct parameter and type list
    if is_multiband:
        param_list = config_dict['bands'][band]['param_list']
        type_list = config_dict['bands'][band]['type_list']
    else:
        param_list = config_dict['param_list']
        type_list = config_dict['type_list']
        
    # Helper to retrieve band-specific args
    def get_band_param(key, default_val=None):
        val = args_dict.get(key, default_val)
        if is_multiband and isinstance(val, (list, tuple)):
            return val[band_idx]
        return val

    # Robust path resolution for transferring between environments (e.g. cluster -> local mac)
    def resolve_robust_path(path):
        if not path:
            return path
        if os.path.exists(path):
            return path
        
        # Try resolving relative to CWD, or run_dir parents
        cwd = os.getcwd()
        parts = path.split(os.sep)
        for i in range(len(parts)):
            subpath = os.sep.join(parts[i:])
            for base_dir in [cwd, os.path.dirname(run_dir), os.path.dirname(os.path.dirname(run_dir))]:
                candidate = os.path.join(base_dir, subpath)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
        return path

    # Helper to resolve FITS file path
    def resolve_fits_path(path_key, name_key, default_name):
        base = args_dict.get(path_key)
        if not base:
            return None
        
        # If base is a list/tuple, extract correct element
        if is_multiband and isinstance(base, (list, tuple)):
            base = base[band_idx]
        
        # Apply robust resolution to base path
        base = resolve_robust_path(base)
        
        if os.path.isfile(base):
            return base
            
        name = args_dict.get(name_key, default_name)
        if is_multiband and isinstance(name, (list, tuple)):
            name = name[band_idx]
            
        if is_multiband:
            candidate = os.path.join(base, band, name)
            candidate = resolve_robust_path(candidate)
            if os.path.isfile(candidate):
                return candidate
            candidate = os.path.join(base, name)
            candidate = resolve_robust_path(candidate)
            if os.path.isfile(candidate):
                return candidate
        else:
            candidate = os.path.join(base, name)
            candidate = resolve_robust_path(candidate)
            if os.path.isfile(candidate):
                return candidate
                
        return base

    # Load npy pixelated source
    if is_multiband:
        source_pixels_path = os.path.join(run_dir, band, 'kwargs_source_pixels.npy')
    else:
        source_pixels_path = os.path.join(run_dir, 'kwargs_source_pixels.npy')
        
    if not os.path.exists(source_pixels_path):
        raise FileNotFoundError(f"Missing kwargs_source_pixels.npy in {source_pixels_path}")
    source_pixels = np.load(source_pixels_path)
    
    # Load kwargs_result
    if is_multiband:
        band_result_path = os.path.join(run_dir, band, 'kwargs_result.json')
        if os.path.exists(band_result_path):
            with open(band_result_path, 'r') as f:
                kwargs_result = json.load(f)
        else:
            with open(result_path, 'r') as f:
                top_result = json.load(f)
            kwargs_result = top_result['kwargs_by_band'][band]
    else:
        with open(result_path, 'r') as f:
            kwargs_result = json.load(f)
            
    # Load FITS data cutout, noise map, PSF, mask
    print("Loading data FITS files...")
    data_file = resolve_fits_path('data_path', 'data_name', 'Data_cutout.fits')
    noise_file = resolve_fits_path('noise_path', 'noise_name', 'noise.fits')
    psf_file = resolve_fits_path('psf_path', 'psf_name', 'psf_modelled.fits')
    
    if not data_file or not os.path.isfile(data_file):
        raise FileNotFoundError(f"Could not resolve data FITS path: {data_file}")
    if not noise_file or not os.path.isfile(noise_file):
        raise FileNotFoundError(f"Could not resolve noise FITS path: {noise_file}")
    if not psf_file or not os.path.isfile(psf_file):
        raise FileNotFoundError(f"Could not resolve PSF FITS path: {psf_file}")
        
    print(f"Using FITS files:\n - Data: {data_file}\n - Noise: {noise_file}\n - PSF: {psf_file}")
    image_data = fits.getdata(data_file).astype(np.float64)
    background_offset = float(args_dict.get('background_offset', 0.0))
    if background_offset != 0.0:
        image_data = image_data - background_offset
        print(f"[bkg] Applied stored global background offset: {background_offset:.6f}")
    noise_map = fits.getdata(noise_file).astype(np.float64)
    psf_data = fits.getdata(psf_file).astype(np.float64)
    psf_data = psf_data / np.sum(psf_data) # normalize PSF
    
    source_arc_mask = None
    source_arc_mask_file = resolve_fits_path('source_arc_mask_path', 'source_arc_mask_name', 'mask_1.fits')
    if source_arc_mask_file and os.path.isfile(source_arc_mask_file):
        source_arc_mask = fits.getdata(source_arc_mask_file).astype(bool)
        print(f" - Mask: {source_arc_mask_file}")
        
    # Initialize LensImage Extension
    print("Reconstructing LensImage model...")
    supersampling_factor = get_band_param('supersampling_factor', 2)
    kwargs_numerics = {
        'supersampling_factor': supersampling_factor
    }
    kwargs_lens_equation_solver = {
        'nsolutions': args_dict.get('ps_nsolutions', 5),
        'niter': args_dict.get('ps_niter', 10),
        'scale_factor': args_dict.get('ps_scale_factor', 2),
        'nsubdivisions': args_dict.get('ps_nsubdivisions', 3),
    }
    
    pixel_scale = get_band_param('pixel_scale')
    source_grid_scale = get_band_param('source_grid_scale', 1.0)
    
    lens_image = create_lens_image(
        param_list=param_list,
        type_list=type_list,
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=pixel_scale,
        kwargs_numerics=kwargs_numerics,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver,
        source_arc_mask=source_arc_mask,
        source_grid_scale=source_grid_scale,
    )
    
    # Mask regions of the source plane outside the ray-traced source_arc_mask footprint
    if source_arc_mask is not None:
        try:
            from matplotlib.path import Path
            print("Mapping source_arc_mask to source plane for masking...")
            img_x, img_y = lens_image.Grid.pixel_coordinates
            img_x = np.asarray(img_x)
            img_y = np.asarray(img_y)
            if img_x.ndim == 1 and img_y.ndim == 1:
                img_x, img_y = np.meshgrid(img_x, img_y)
                
            fig_dummy, ax_dummy = plt.subplots()
            cs = ax_dummy.contour(img_x, img_y, source_arc_mask.astype(float), levels=[0.5])
            segments = cs.allsegs[0] if (hasattr(cs, 'allsegs') and len(cs.allsegs) > 0) else []
            plt.close(fig_dummy)
            
            mapped_contours = []
            kwargs_lens = kwargs_result.get('kwargs_lens', None)
            for seg in segments:
                if len(seg) >= 3:
                    x_b_img, y_b_img = seg[:, 0], seg[:, 1]
                    beta_x_b, beta_y_b = lens_image.MassModel.ray_shooting(
                        x_b_img, y_b_img, kwargs_lens
                    )
                    mapped_contours.append((np.asarray(beta_x_b), np.asarray(beta_y_b)))
                    
            if mapped_contours:
                inside_mask = np.zeros(source_pixels.shape, dtype=bool)
                xx_grid, yy_grid = lens_image.SourceModel.pixel_grid.pixel_coordinates
                if xx_grid.ndim == 1 and yy_grid.ndim == 1:
                    xx_grid, yy_grid = np.meshgrid(xx_grid, yy_grid)
                points = np.column_stack((xx_grid.ravel(), yy_grid.ravel()))
                for beta_x_b, beta_y_b in mapped_contours:
                    polygon_vertices = np.column_stack((beta_x_b, beta_y_b))
                    path = Path(polygon_vertices)
                    inside_mask |= path.contains_points(points).reshape(source_pixels.shape)
                    
                # Apply mask to the median source pixels
                source_pixels = np.where(inside_mask, source_pixels, 0.0)
                print("Applied mapped source_arc_mask to source plane pixels.")
        except Exception as e:
            print(f"Warning: could not apply source_arc_mask to source plane: {e}")
            
    # Load HMC sampler chains if HMC is used
    sampler = args_dict.get('sampler', 'svi')
    mcmc_samples = None
    rec_sources = None
    samples_hdf5_path = os.path.join(run_dir, 'hmc_samples.h5')
    
    if sampler == 'hmc' and os.path.exists(samples_hdf5_path):
        try:
            from herculens_wrapper.samplers import _load_hmc_samples_hdf5
            from run_multiband import _band_hmc_samples
            import jax
            import jax.numpy as jnp
            from herculens_wrapper.models import PowerSpectrum
            
            print("Loading HMC samples from hmc_samples.h5...")
            raw_samples, _ = _load_hmc_samples_hdf5(samples_hdf5_path)
            mcmc_samples = _band_hmc_samples(raw_samples, {
                'site_prefix': f'band_{band_idx}_{band}' if is_multiband else '',
                'name': band,
                'prob_model': None
            })
            
            n_samples_total = len(mcmc_samples['pixels_wn_source_grid'])
            print(f"Loaded {n_samples_total} posterior samples.")
            if thin > 1:
                print(f"Thinning samples by a factor of {thin}...")
                for key in list(mcmc_samples.keys()):
                    mcmc_samples[key] = np.asarray(mcmc_samples[key])[::thin]
                n_samples_total = len(mcmc_samples['pixels_wn_source_grid'])
                print(f"Using {n_samples_total} thinned posterior samples.")
                
            # Reconstruct 2D source plane for all samples using JAX PowerSpectrum vmap
            print("Reconstructing physical source planes for samples...")
            p_wn_arr = jnp.asarray(mcmc_samples['pixels_wn_source_grid'], dtype=jnp.float64)
            n_arr = jnp.asarray(np.ravel(mcmc_samples['n_source_grid']), dtype=jnp.float64)
            sigma_arr = jnp.asarray(np.ravel(mcmc_samples['sigma_source_grid']), dtype=jnp.float64)
            rho_arr = jnp.asarray(np.ravel(mcmc_samples['rho_source_grid']), dtype=jnp.float64)
            
            ny, nx = p_wn_arr.shape[1], p_wn_arr.shape[2]
            k_grid = PowerSpectrum.K_grid((ny, nx))
            k_values = jnp.asarray(k_grid.k)
            
            is_positive = True
            if 'pixelated_prior' in config_dict:
                is_positive = bool(config_dict['pixelated_prior'].get('positive', True))
                
            def single_source_pixels(n, sigma, rho, p_wn):
                scale = jnp.sqrt(PowerSpectrum.P_Matern(k_values, n, sigma, rho, k_zero=0.0))
                pixels = jnp.fft.irfft2(PowerSpectrum.pack_fft_values(p_wn * scale), s=scale.shape, norm='ortho')
                if is_positive:
                    return jax.nn.softplus(100.0 * pixels) / 100.0
                return pixels
                
            vmap_fn = jax.jit(jax.vmap(single_source_pixels))
            
            batch_size = 200
            all_rec_sources = []
            for b in range(0, n_samples_total, batch_size):
                b_end = min(b + batch_size, n_samples_total)
                batch_srcs = vmap_fn(n_arr[b:b_end], sigma_arr[b:b_end], rho_arr[b:b_end], p_wn_arr[b:b_end])
                all_rec_sources.append(np.asarray(batch_srcs))
                
            rec_sources = np.concatenate(all_rec_sources, axis=0)
            print(f"Successfully reconstructed {len(rec_sources)} source planes.")
        except Exception as e:
            print(f"Warning: Failed to load and reconstruct HMC samples: {e}. Falling back to single point estimate.")
            sampler = 'svi'
            mcmc_samples = None
            
    # Watershed-based deblending using photutils
    print("Deblending source plane components using photutils...")
    peak_flux = np.max(source_pixels)
    threshold = threshold_frac * peak_flux
    
    segm = detect_sources(source_pixels, threshold, n_pixels=n_pixels)
    if segm is not None:
        try:
            deblended = deblend_sources(source_pixels, segm, n_pixels=n_pixels, contrast=contrast, n_levels=n_levels)
            if deblended is not None:
                labeled_array = deblended.data
                labels = deblended.labels
            else:
                labeled_array = segm.data
                labels = segm.labels
        except Exception as e:
            print(f"Warning: photutils deblend failed: {e}. Falling back to initial detection.")
            labeled_array = segm.data
            labels = segm.labels
    else:
        labeled_array = np.zeros_like(source_pixels, dtype=int)
        labels = []
        
    num_features = len(labels)
    print(f"Photutils found {num_features} deblended source features above threshold of {threshold:.4e} ({threshold_frac*100}% of peak flux {peak_flux:.4e})")
    
    # Compile components sorted by total flux
    components = []
    for lbl in labels:
        mask = (labeled_array == lbl)
        flux = np.sum(source_pixels[mask])
        components.append({
            'id': lbl,
            'flux': flux,
            'mask': mask,
        })
    components.sort(key=lambda x: x['flux'], reverse=True)
    
    num_to_show = len(components)
    if num_to_show == 0:
        print("Warning: No components detected above the threshold.")
        num_to_show = 1
        # Fallback to single dummy component of entire source
        components = [{
            'id': 1,
            'flux': np.sum(source_pixels),
            'mask': np.ones_like(source_pixels, dtype=bool)
        }]
        
    print(f"Evaluating lensed contributions for all {num_to_show} components...")
    
    def _clean_kwargs(kw_dict):
        kw_clean = copy.deepcopy(kw_dict)
        if 'kwargs_source' in kw_clean:
            clean_src = []
            for kw in kw_clean['kwargs_source']:
                if isinstance(kw, dict):
                    kw_c = {
                        key: val for key, val in kw.items()
                        if key not in ['pixels_wn', 'n_source_grid', 'rho_source_grid', 'sigma_source_grid', 'rho_soure_grid']
                        and not isinstance(val, dict)
                    }
                    if 'pixels' in kw:
                        kw_c['pixels'] = kw['pixels']
                    clean_src.append(kw_c)
                else:
                    clean_src.append(kw)
            kw_clean['kwargs_source'] = clean_src
        return kw_clean

    if sampler == 'hmc' and mcmc_samples is not None:
        # Reconstruct sample-averaged lensed contributions
        print(f"Evaluating sample-averaged lensed contributions over {n_samples_total} samples...")
        
        # Helpers to reconstruct kwargs for sample i
        def get_sample_kwargs(i):
            sample_kw = copy.deepcopy(kwargs_result)
            
            # 1. Update lens mass parameters
            if 'kwargs_lens' in sample_kw:
                for j, comp in enumerate(sample_kw['kwargs_lens']):
                    for key in list(comp.keys()):
                        param_name = f"lens_{key}_{j}"
                        if param_name in mcmc_samples:
                            comp[key] = float(np.ravel(mcmc_samples[param_name][i])[0])
                            
            # 2. Update lens light parameters
            if 'kwargs_lens_light' in sample_kw:
                for j, comp in enumerate(sample_kw['kwargs_lens_light']):
                    for key in list(comp.keys()):
                        param_name = f"lens_light_{key}_{j}"
                        if param_name in mcmc_samples:
                            comp[key] = float(np.ravel(mcmc_samples[param_name][i])[0])
                            
            # 3. Update point source parameters
            if 'kwargs_point_source' in sample_kw:
                for j, comp in enumerate(sample_kw['kwargs_point_source']):
                    for key in list(comp.keys()):
                        for prefix in ("point_source_", "ps_"):
                            param_name = f"{prefix}{key}_{j}"
                            if param_name in mcmc_samples:
                                comp[key] = float(np.ravel(mcmc_samples[param_name][i])[0])
                                break
                                
            # 4. Update source parameters with the reconstructed pixels for sample i
            if 'kwargs_source' in sample_kw:
                for j, comp in enumerate(sample_kw['kwargs_source']):
                    comp['pixels'] = rec_sources[i]
                    for key in ('n_source_grid', 'rho_source_grid', 'sigma_source_grid'):
                        for name in (key, f"source_{key}_{j}", f"{key}_{j}"):
                            if name in mcmc_samples:
                                comp[key] = float(np.ravel(mcmc_samples[name][i])[0])
                                break
            return sample_kw

        list_model_combined = []
        list_model_lens_light = []
        list_lensed_components = [[] for _ in range(num_to_show)]
        
        start_time = time.time()
        for i in range(n_samples_total):
            if i % 100 == 0:
                print(f"Processing sample {i}/{n_samples_total}...")
                
            sample_kw = get_sample_kwargs(i)
            sample_kw = _clean_kwargs(sample_kw)
            
            comp_tot = lens_image.model(**sample_kw, source_add=True, lens_light_add=True, point_source_add=True)
            comp_ll = lens_image.model(**sample_kw, source_add=False, lens_light_add=True, point_source_add=False)
            list_model_combined.append(comp_tot)
            list_model_lens_light.append(comp_ll)
            
            for k in range(num_to_show):
                comp = components[k]
                masked_source = np.zeros_like(source_pixels)
                # Mask the sample's reconstructed source plane
                masked_source[comp['mask']] = rec_sources[i][comp['mask']]
                
                kwargs_comp = copy.deepcopy(sample_kw)
                kwargs_comp['kwargs_source'][0]['pixels'] = masked_source
                
                comp_lensed = lens_image.model(
                    **kwargs_comp, source_add=True, lens_light_add=False, point_source_add=False
                )
                list_lensed_components[k].append(comp_lensed)
                
        elapsed = time.time() - start_time
        print(f"Evaluated all samples in {elapsed:.2f} seconds.")
        
        # Compute pixel-by-pixel medians
        model_combined = np.median(np.array(list_model_combined), axis=0)
        model_lens_light = np.median(np.array(list_model_lens_light), axis=0)
        
        lensed_components = []
        for k in range(num_to_show):
            comp_med = np.median(np.array(list_lensed_components[k]), axis=0)
            lensed_components.append(comp_med)
            
    else:
        # Pre-render standard combined model and lens light (Point-estimate approach)
        kwargs_all = copy.deepcopy(kwargs_result)
        kwargs_all['kwargs_source'][0]['pixels'] = source_pixels
        kwargs_all = _clean_kwargs(kwargs_all)
        
        model_combined = lens_image.model(**kwargs_all, source_add=True, lens_light_add=True, point_source_add=True)
        model_lens_light = lens_image.model(**kwargs_all, source_add=False, lens_light_add=True, point_source_add=False)
        
        # Render each component
        lensed_components = []
        for k in range(num_to_show):
            comp = components[k]
            masked_source = np.zeros_like(source_pixels)
            masked_source[comp['mask']] = source_pixels[comp['mask']]
            
            kwargs_comp = copy.deepcopy(kwargs_result)
            kwargs_comp['kwargs_source'][0]['pixels'] = masked_source
            kwargs_comp = _clean_kwargs(kwargs_comp)
            
            comp_lensed = lens_image.model(
                **kwargs_comp, source_add=True, lens_light_add=False, point_source_add=False
            )
            lensed_components.append(comp_lensed)
        
    # Extents for plot axes in arcseconds
    ny_img, nx_img = image_data.shape
    img_half_w_x = nx_img * pixel_scale / 2.0
    img_half_w_y = ny_img * pixel_scale / 2.0
    img_extent = [-img_half_w_x, img_half_w_x, -img_half_w_y, img_half_w_y]
    
    # Calculate coordinate limits of the zoomed source plane footprint
    src_x, src_y = lens_image.SourceModel.pixel_grid.pixel_coordinates
    img_x, img_y = lens_image.Grid.pixel_coordinates
    if source_arc_mask is not None:
        # Find ray-traced coordinate range of the mask footprint
        x_mapped, y_mapped = lens_image.MassModel.ray_shooting(
            img_x[source_arc_mask], img_y[source_arc_mask], kwargs_result['kwargs_lens']
        )
        center_x = 0.5 * (np.min(x_mapped) + np.max(x_mapped))
        center_y = 0.5 * (np.min(y_mapped) + np.max(y_mapped))
        half_range = 0.5 * max(np.max(x_mapped) - np.min(x_mapped), np.max(y_mapped) - np.min(y_mapped))
        src_extent = [center_x - half_range, center_x + half_range, center_y - half_range, center_y + half_range]
    else:
        src_extent = [np.min(src_x), np.max(src_x), np.min(src_y), np.max(src_y)]
        
    # Set up matplotlib figure
    # Row 0 has 4 panels: Cutout, Combined model, Lens light, Segmented source plane.
    # Row 1 has N panels: Lensed Component 1, 2, ..., N.
    n_cols = max(5, num_to_show)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols + 1 * n_cols, 10))
    
    # Helpers for rendering in linear/log
    def render_im(ax, img_raw, extent, title, colormap='twilight', is_log=False):
        img = img_raw
        if is_log:
            vmin = np.percentile(img[img > 0], 10) if np.any(img > 0) else 1e-4
            log_img = np.log10(np.maximum(img, vmin))
            im = ax.imshow(log_img, origin='lower', cmap=colormap, extent=extent)
            plt.colorbar(im, ax=ax, label='log10(pixel flux)')
        else:
            im = ax.imshow(img, origin='lower', cmap=colormap, extent=extent)
            plt.colorbar(im, ax=ax, label='Pixel flux')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        
    is_log = (plot_scale == 'log')
    
    # Panel 1: Image Data cutout (log scale)
    render_im(axes[0, 0], image_data, img_extent, "Original Data Cutout", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 0].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
        
    # Panel 2: Whole Combined Model (log scale)
    render_im(axes[0, 1], model_combined, img_extent, "Combined Model Fit", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 1].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
        
    # Panel 3: Lens Light Model (log scale)
    render_im(axes[0, 2], model_lens_light, img_extent, "Lens Light Model", is_log=is_log)
    
    # Source plane extents for rendering source
    ny_src, nx_src = source_pixels.shape
    src_pixel_scale = (src_extent[1] - src_extent[0]) / nx_src
    src_plot_extent = [src_extent[0], src_extent[1], src_extent[2], src_extent[3]]
    
    # Panel 4: Model - Lens Light (log scale)
    render_im(axes[0, 3], model_combined - model_lens_light, img_extent, "Model - Lens Light", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 3].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)

    # Panel 5: Segmented Source Plane (always linear scale!)
    ax_src = axes[0, 4]
    im_src = ax_src.imshow(source_pixels, origin='lower', cmap='twilight', extent=src_plot_extent)
    plt.colorbar(im_src, ax=ax_src, label='Pixel flux')
        
    # Overlay colors/contours for each component on source plane
    color_cycle = ['cyan', 'magenta', 'orange', 'yellow', 'lime', 'pink', 'purple']
    linestyle_cycle = ['solid', 'dashed', 'dotted', 'dashdot']
    from matplotlib.lines import Line2D
    legend_elements = []
    for idx in range(num_to_show):
        comp = components[idx]
        color = color_cycle[idx % len(color_cycle)]
        linestyle = linestyle_cycle[(idx // len(color_cycle)) % len(linestyle_cycle)]
        # Outline contour around component mask
        ax_src.contour(comp['mask'], levels=[0.5], colors=[color], linestyles=[linestyle], extent=src_plot_extent, linewidths=2.0)
        legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=f"C{idx+1}"))
        
    ax_src.legend(handles=legend_elements, loc='upper right', framealpha=0.8, fontsize=8 if num_to_show > 8 else 10)
    ax_src.set_title("Segmented Source Plane", fontsize=12, pad=10)
    ax_src.set_xlabel('arcsec')
    ax_src.set_ylabel('arcsec')
    
    # Hide any unused subplots in row 0
    for col_idx in range(5, n_cols):
        axes[0, col_idx].axis('off')
    
    # Row 2: Lensed components (log scale)
    for idx in range(num_to_show):
        color = color_cycle[idx % len(color_cycle)]
        linestyle = linestyle_cycle[(idx // len(color_cycle)) % len(linestyle_cycle)]
        ax_comp = axes[1, idx]
        comp_img = lensed_components[idx]
        
        style_tag = f" ({linestyle})" if (idx >= len(color_cycle)) else ""
        render_im(ax_comp, comp_img, img_extent, f"Lensed Component {idx+1}{style_tag}", is_log=is_log)
        if source_arc_mask is not None:
            ax_comp.contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
            
        for spine in ax_comp.spines.values():
            spine.set_color(color)
            spine.set_linestyle(linestyle)
            spine.set_linewidth(2.5)
            
    # Hide any unused subplots in row 1
    for col_idx in range(num_to_show, n_cols):
        axes[1, col_idx].axis('off')
        
    plt.tight_layout()
    if is_multiband:
        output_plot_path = os.path.join(run_dir, band, 'deblended_contributions.png')
    else:
        output_plot_path = os.path.join(run_dir, 'deblended_contributions.png')
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] Generated deblended component ray-tracing diagnostic plot!")
    print(f"Saved to: {output_plot_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblend pixelated source reconstructions and ray-trace individual component contributions.")
    parser.add_argument(
        '--run_dir', type=str, default='/Users/xczhou/Desktop/modelling/modeling_F277W/pixelated',
        help='Path to the pixelated run directory containing kwargs_result.json'
    )
    parser.add_argument(
        '--threshold_frac', type=float, default=0.05,
        help='Threshold fraction of the peak source intensity to define components'
    )
    parser.add_argument(
        '--plot_scale', type=str, default='log', choices=['linear', 'log'],
        help='Scale to plot flux profiles (linear or log)'
    )
    parser.add_argument(
        '--n_pixels', type=int, default=5,
        help='Minimum number of connected pixels to detect a component'
    )
    parser.add_argument(
        '--contrast', type=float, default=0.001,
        help='Fraction of total flux a local peak must have to be deblended (contrast threshold)'
    )
    parser.add_argument(
        '--n_levels', type=int, default=32,
        help='Number of multi-thresholding levels for watershed deblending'
    )
    parser.add_argument(
        '--band', type=str, default=None,
        help='Specific band to process (required for multiband results)'
    )
    parser.add_argument(
        '--thin', type=int, default=1,
        help='Thinning factor for HMC chains (1 means no thinning/use all samples)'
    )
    
    args = parser.parse_args()
    deblend_and_ray_trace(
        args.run_dir, args.threshold_frac, args.plot_scale,
        n_pixels=args.n_pixels, contrast=args.contrast, n_levels=args.n_levels,
        band=args.band, thin=args.thin
    )
