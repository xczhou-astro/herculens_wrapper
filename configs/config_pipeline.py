import numpy as np
import pandas as pd

def lens_mass_config(image_size=None, pixel_scale=None, args=None):

    
    # lens_mass_type_list = ['SIE', 'SHEAR']
    lens_mass_type_list = ['EPL', 'SHEAR']
    # lens_mass_type_list = ['SIE', 'GAUSSIAN', 'SHEAR']
    # lens_mass_type_list = ['SIE'] + ['GAUSSIAN'] * 10 + ['SHEAR']
    if lens_mass_type_list == ['SIE', 'SHEAR']:
        lens_mass_params_list = [
            {
                'theta_E': [0.4, 0.1, 0.3, 0.5],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        ]
    elif lens_mass_type_list == ['EPL', 'SHEAR']:
        lens_mass_params_list = [
            {
                'theta_E': [0.4, 0.1, 0.3, 0.5],
                'gamma': [2.0, 0.1, 1.0, 3.0],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        ]
    elif lens_mass_type_list == ['SIE', 'GAUSSIAN', 'SHEAR']:
        
        lens_mass_params_list = [
            {
                'theta_E': [0.4, 0.1, 0.3, 0.5],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'amp': [1.0, 0.1],
                'sigma_x': [0.2, 0.1, 0, 0.5],
                'sigma_y': [0.2, 0.1, 0, 0.5],
                'center_x': [0.0, 0.1, -0.2, 0.2],
                'center_y': [0.0, 0.1, -0.2, 0.2],
            },
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        ]

    return lens_mass_type_list, lens_mass_params_list

def lens_light_config(image_size=None, pixel_scale=None, args=None):

    num_gaussian_sets = 3
    num_gaussian_per_set = 20
    num_extra_gaussian = 10

    num_total_gaussian = num_gaussian_sets * num_gaussian_per_set + num_extra_gaussian

    lens_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_total_gaussian
    # lens_light_type_list = ['SERSIC_ELLIPSE']
    
    if lens_light_type_list[0] == 'GAUSSIAN_ELLIPSE':

        if pixel_scale is None:
            raise ValueError('Pixel scale is required')
        
        if image_size is None:
            raise ValueError('Image size is required')

        max_sigma = image_size * pixel_scale / 2.0
        # max_sigma = 0.5
        min_sigma = pixel_scale / 5.0

        sigma_list = 10**(np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_gaussian_per_set))

        lens_light_params_list = []
        for i in range(num_gaussian_sets):
            for j in range(num_gaussian_per_set):
                geometry_head = i * num_gaussian_per_set
                if j == 0:
                    lens_light_params_list.append(
                        {
                            'amp': [1.0, 0.1],
                            'sigma': sigma_list[j], 
                            'center_x': [0.0, 0.1, -0.3, 0.3],
                            'center_y': [0.0, 0.1, -0.3, 0.3],
                            'e1': [0.0, 0.1, -0.6, 0.6],
                            'e2': [0.0, 0.1, -0.6, 0.6],
                        }
                    )
                else:
                    lens_light_params_list.append(
                        {
                            'amp': [1.0, 0.1],
                            'sigma': sigma_list[j],
                            'center_x': ['correlated', 'lens_light', geometry_head, 'center_x'],
                            'center_y': ['correlated', 'lens_light', geometry_head, 'center_y'],
                            'e1': ['correlated', 'lens_light', geometry_head, 'e1'],
                            'e2': ['correlated', 'lens_light', geometry_head, 'e2'],
                        }
                    )

        sigma_list_extra = 10**(np.linspace(np.log10(0.01), np.log10(2 * pixel_scale), num_extra_gaussian))
        # Anchor extra Gaussians to the first extra component, which is appended at index
        # num_gaussian_sets * num_gaussian_per_set.
        geometry_head = num_gaussian_sets * num_gaussian_per_set
        for i in range(num_extra_gaussian):
            if i == 0:
                lens_light_params_list.append(
                    {
                        'amp': [1.0, 0.1],
                        'sigma': sigma_list_extra[i],
                        'center_x': [0.0, 0.03, -0.1, 0.1],
                        'center_y': [0.0, 0.03, -0.1, 0.1],
                        'e1': [0.0, 0.03, -0.2, 0.2],
                        'e2': [0.0, 0.03, -0.2, 0.2],
                    }
                )
            else:
                lens_light_params_list.append(
                    {
                        'amp': [1.0, 0.1],
                        'sigma': sigma_list_extra[i],
                        'center_x': ['correlated', 'lens_light', geometry_head, 'center_x'],
                        'center_y': ['correlated', 'lens_light', geometry_head, 'center_y'],
                        'e1': ['correlated', 'lens_light', geometry_head, 'e1'],
                        'e2': ['correlated', 'lens_light', geometry_head, 'e2'],
                    }
                )
    
    elif lens_light_type_list[0] == 'SERSIC_ELLIPSE':
    
        lens_light_params_list = [
            {
                'amp': [3.5, 0.1],
                'e1': [0.0, 0.1, -0.4, 0.4],
                'e2': [0.0, 0.1, -0.4, 0.4],
                'R_sersic': [0.5, 0.2, 0.01, 1.0],
                'n_sersic': [1.5, 0.5, 0.1, 8.0],
                'center_x': [0.0, 0.1, -0.5, 0.5],
                'center_y': [0.0, 0.1, -0.5, 0.5],
            },
        ]

    num_independent_gaussians = 10
    max_sigma = args.crop_size * args.pixel_scale / 2.0
    min_sigma = 0.03
    sigma_bins = 10**(np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_independent_gaussians + 1))

    lens_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_independent_gaussians
    lens_light_params_list = []
    for k in range(num_independent_gaussians):
        sigma_low = sigma_bins[k]
        sigma_high = sigma_bins[k+1]
        init_sigma = 10**(0.5 * (np.log10(sigma_low) + np.log10(sigma_high)))
        sigma_unc = 0.2 * init_sigma
        
        lens_light_params_list.append({
            'amp': [2.0, 0.1],
            'sigma': [init_sigma, sigma_unc, sigma_low, sigma_high],
            'center_x': [0.0, 0.1, -0.2, 0.2],
            'center_y': [0.0, 0.1, -0.2, 0.2],
            'e1': [0.0, 0.1, -0.5, 0.5],
            'e2': [0.0, 0.1, -0.5, 0.5],
        })

    return lens_light_type_list, lens_light_params_list

def source_light_config(image_size=None, pixel_scale=None, args=None, 
                        init_params=None):

    # source_light_type_list = ['SERSIC_ELLIPSE', 'GAUSSIAN']
    source_light_type_list = ['PIXELATED']
    # source_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_source_light * num_mges
    # source_light_type_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
    # source_light_type_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE', 'GAUSSIAN_ELLIPSE']
    # source_light_type_list = ['SERSIC_ELLIPSE']
    if source_light_type_list == ['SERSIC_ELLIPSE', 'GAUSSIAN']:
        source_light_params_list = [
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.1, 0.1],
                'e2': [0.0, 0.1, -0.1, 0.1],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 4.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            },
            {
                'amp': [1.0, 0.1],
                'sigma': [0.05, 0.01, 0.01, 0.1],
                'center_x': [0.0, 0.1, -0.5, 0.5],
                'center_y': [0.0, 0.1, -0.5, 0.5],
            }
            ]
        
    elif source_light_type_list == ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']:
        source_light_params_list = [
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 8.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            },
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 8.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            }
        ]
    elif source_light_type_list == ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE', 'GAUSSIAN_ELLIPSE']:
        source_light_params_list = [
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.1, 0.1],
                'e2': [0.0, 0.1, -0.1, 0.1],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 4.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            },
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.1, 0.1],
                'e2': [0.0, 0.1, -0.1, 0.1],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 4.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            },
            {
                'amp': [1.0, 0.1],
                'sigma': [0.1, 0.05, 0.01, 0.2],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
                'e1': [0.0, 0.03, -0.2, 0.2],
                'e2': [0.0, 0.03, -0.2, 0.2],
            }
        ]

    elif source_light_type_list == ['SERSIC_ELLIPSE']:
        source_light_params_list = [
            {
                'amp': [1.0, 0.1],
                'e1': [0.0, 0.1, -0.1, 0.1],
                'e2': [0.0, 0.1, -0.1, 0.1],
                'R_sersic': [0.2, 0.1, 0.01, 0.5],
                'n_sersic': [2.0, 0.5, 0.1, 4.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            }
        ]

    elif source_light_type_list == ['PIXELATED']:
        source_light_params_list = []
        kwargs_pixelated_source = {
            'pixel_grid': {
                'pixel_adaptive_grid': True,
                'pixel_grid_shape': 150,
                'pixel_interpol': 'fast_bilinear',
                # Fallback settings used when pixel_adaptive_grid is False:
                'pixel_scale_factor': 0.5,
                'grid_center': (0.0, 0.0),
                'grid_shape': (2.0, 2.0),
            },
            'pixelated_prior': {
                'prior_type': 'matern', # matern | wavelet_sparsity | wavelet_penalty
                'regul_strengths': (3.0, 3.0),
                'k_zero': 0.0,
                'n_value_low': 1.0,
                'n_value_high': 100,
                'sigma_low': 1e-5,
                'sigma_high': 10.0,
                'positive': True,
            }
        }

        source_light_params_list.append(
            kwargs_pixelated_source,
        )

    return source_light_type_list, source_light_params_list

def point_source_config(image_size=None, pixel_scale=None, args=None):
    
    point_source_params_list = []
    
    # point_source_type_list = ['IMAGE_POSITIONS'] * args.num_point_sources
    point_source_type_list = ['SOURCE_POSITION'] * args.num_point_sources
    if point_source_type_list[0] == 'SOURCE_POSITION':
        point_source_params_list += [
            {
                'ra': [0.0, 0.1, -0.3, 0.3],
                'dec': [0.0, 0.1, -0.3, 0.3],
                'amp': [2.0, 0.1],
            } 
        ] * args.num_point_sources
        
    elif point_source_type_list[0] == 'IMAGE_POSITIONS':
        ps_sources = pd.read_csv(args.image_positions_catalog)
        
        ras = ps_sources['ra_arcsec'].values
        decs = ps_sources['dec_arcsec'].values
        
        for k in range(args.num_point_sources):
            images_idx = getattr(args, f'images_indices_{k}')
            images_idx = np.asarray(images_idx, dtype=int)
            
            images_ras = ras[images_idx]
            images_decs = decs[images_idx]
            images_num = len(images_idx)
            
            point_source_params_list += [
                {
                    'n_images': images_num,
                    'sigma_image': 3e-3,
                    'sigma_source': 1e-3,
                    'pos_bound': 0.1,
                    'ra': images_ras,
                    'dec': images_decs,
                    'amp': [1.0, 0.1],
                } 
            ]
    
    return point_source_type_list, point_source_params_list
        
def arguments():

    args = {
        'data_path': '../data/F277W/Data_cutout.fits',
        'noise_path': '../data/F277W/noise.fits',
        'psf_path': '../psf/F277W/psf_modelled.fits', 
        'source_arc_mask_path': '../data/F277W/mask_1.fits',
        'save_path': '../modeling_F277W/pipeline_20260718_2',

        'pipeline': True,

        'source_grid_scale': 0.8, 
        'use_source_support_mask': True,
        'use_best_pixel_size': False,
        'manual': True,
        'conjugate_points': None,

        # general settings
        'random_seed': 42,
        'pixel_scale': 0.03,
        'crop_size': 61,
        'background_subtract_corner': 5, 
        'background_subtract_which_corner': 'bottom_left',
        'residual_vis_max': 3.0, 
        'supersampling_factor': 2,
        # Sampler choices: 'svi' | 'hmc'
        'sampler': ['svi', 'hmc'],
        'init_params_path': '../modeling_F277W/parametric/run_0',
        'refine_prior_range': None,
        'refine_prior_min_frac': None,
        'pixel_init_jitter': 0.0,
        'fix_component': [],  # list of components to fix: 'lens_mass' | 'lens_light' | 'source_light'
        'regul_num_samples': 1000,
        'gpus': '7',
        'n_runs': 2,

        # --- svi (Stochastic Variational Inference) ---
        'max_iterations_svi_warmup': 2_000,
        'max_iterations_svi': 30_000,
        'init_learning_rate_svi': 1e-2,
        'init_scale_svi': 0.01,
        'loss_kind_svi': 'trace_meanfield_elbo',  # trace_elbo | trace_meanfield_elbo
        'num_particles_svi': 10,

        'num_warmup_hmc_numpyro': 1000,
        'num_samples_hmc_numpyro': 2000,
        'checkpoint_interval_hmc_numpyro': 500,
        'num_chains_hmc_numpyro': 4,
        'hmc_init_jitter_scale': 1e-3,
        'hmc_init_jitter_sites': 'lens_mass',  # lens_mass | all_non_pixel | all | comma-separated site names | none
        'hmc_init_jitter_max_tries': 200,

        # --- optax (Herculens OptaxOptimizer) ---
        'algorithm_optax': 'adam',  # adam | radam | adabelief
        'max_iterations_optax': 20_000,
        'init_learning_rate_optax': 1e-2,
        'schedule_learning_rate_optax': True,
        'stop_at_loss_increase_optax': False,
        'progress_bar_optax': True,

        # for point sources
        'ps_nsolutions': 5, 
        'ps_niter': 10, 
        'ps_scale_factor': 2,
        'ps_nsubdivisions': 3,

        # for peculiar source
        # 'ps_mask_path': '../F150W/radius_masks/point_source_radius_masks.fits', 
        'ps_mask_path': None,
        'image_positions_catalog': None, 
        'num_point_sources': 1, 
        'relieve_mask_indices': None, 
        'exclude_ps': True, 
    }
    for k in range(args['num_point_sources']):
        args[f'images_indices_{k}'] = None

    return args
