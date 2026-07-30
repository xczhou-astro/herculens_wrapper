import numpy as np
import pandas as pd

def lens_mass_config(image_size=None, pixel_scale=None, args=None):

    
    lens_mass_type_list = ['EPL', 'SHEAR']
    lens_mass_params_list = [            
        {
            'theta_E': [0.0, 0.6], 
            'gamma': [1.2, 2.8], 
            'center_x': [0.0, 0.1, -0.3, 0.3], 
            'center_y': [0.0, 0.1, -0.3, 0.3], 
            'e1': [0.0, 0.25, -0.6, 0.6],
            'e2': [0.0, 0.25, -0.6, 0.6], 
        },
        {
            'ra_0': 0.0, 
            'dec_0': 0.0, 
            'gamma1': [-0.2, 0.2], 
            'gamma2': [-0.2, 0.2], 
        }
    ]

    return lens_mass_type_list, lens_mass_params_list

def lens_light_config(image_size=None, pixel_scale=None, args=None):

    num_independent_gaussians = 10
    lens_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_independent_gaussians

    sigma_range = [args.pixel_scale, 0.5 * args.pixel_scale * args.crop_size]
    sigma_bins = 10**(np.linspace(
        np.log10(sigma_range[0]), 
        np.log10(sigma_range[1]), 
        num_independent_gaussians + 1))

    lens_light_params_list = []
    for i in range(num_independent_gaussians):
        sigma_low = sigma_bins[i]
        sigma_high = sigma_bins[i + 1]

        lens_light_params_list.append({
            # LogNormal(log_loc, log_scale); median=exp(log_loc).
            # Herculens Gaussian amp is total flux.
            'amp': [2.0, 0.1],
            # LogUniform(low, high), bounded to Gaussian scale bin i.
            'sigma': [sigma_low, sigma_high],
            'e1': [0.0, 0.1, -0.5, 0.5], 
            'e2': [0.0, 0.1, -0.5, 0.5], 
            'center_x': [0.0, 0.1, -0.2, 0.2], 
            'center_y': [0.0, 0.1, -0.2, 0.2], 
        })

        # lens_light_params_list.append(
        #     {
        #         'amp': [2.0, 0.1],
        #         'sigma': [sigma_low, sigma_high], 
        #         'e1': [0.0, 0.2, -0.5, 0.5], 
        #         'e2': [0.0, 0.2, -0.5, 0.5], 
        #         'center_x': [0.0, 0.2, -0.3, 0.3], 
        #         'center_y': [0.0, 0.2, -0.3, 0.3], 
        #     }
        # )

    return lens_light_type_list, lens_light_params_list

def source_light_config(image_size=None, pixel_scale=None, args=None, 
                        init_params=None):

    num_independent_gaussians = 4
    # source_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_independent_gaussians
    source_light_type_list = ['SERSIC_ELLIPSE']
    # source_light_type_list = ['PIXELATED']

    if source_light_type_list[0] == 'GAUSSIAN_ELLIPSE':
        
        sigma_range = [0.001, 0.1]
        sigma_bins = np.logspace(
            np.log10(sigma_range[0]), 
            np.log10(sigma_range[1]), 
            num_independent_gaussians + 1,
        )

        source_light_params_list = []
        for i in range(num_independent_gaussians):
            sigma_low = sigma_bins[i]
            sigma_high = sigma_bins[i+1]

            source_light_params_list.append(
                {
                    # LogNormal(log_loc, log_scale); median=exp(log_loc).
                    # Gaussian amp is total flux.
                    'amp': [2.0, 0.1],
                    # LogUniform(low, high), bounded to Gaussian scale bin i.
                    'sigma': [sigma_low, sigma_high],
                    'e1': [0.0, 0.2, -0.5, 0.5], 
                    'e2': [0.0, 0.2, -0.5, 0.5], 
                    'center_x': [0.0, 0.5, -0.3, 0.3], 
                    'center_y': [0.0, 0.5, -0.3, 0.3], 
                }
            )
        

    elif source_light_type_list[0] == 'SERSIC_ELLIPSE':

        source_light_params_list = [
            {
                # LogNormal(log_loc, log_scale); median=exp(log_loc).
                # Sersic amp is I(R_sersic).
                'amp': [2.0, 0.1],
                'e1': [0.0, 0.2, -0.5, 0.5],
                'e2': [0.0, 0.2, -0.5, 0.5],
                'R_sersic': [0.2, 1.0],
                'n_sersic': [1.0, 8.0],
                'center_x': [0.0, 0.1, -0.3, 0.3],
                'center_y': [0.0, 0.1, -0.3, 0.3],
            },
        ]

    elif source_light_type_list[0] == 'PIXELATED':
        source_light_params_list = []
        kwargs_pixelated_source = {
            'pixel_grid': {
                'pixel_adaptive_grid': True,
                'pixel_grid_shape': 60,
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
                'n_value_low': 1e-4,
                'n_value_high': 100,
                'sigma_low': 1e-5,
                'sigma_high': 10.0,
                'rho_low': None, 
                'rho_high': None, 
                'positive': True,
            }
        }

        source_light_params_list.append(
            kwargs_pixelated_source,
        )

    return source_light_type_list, source_light_params_list
        

def arguments():

    args = {
        # data settings
        # 'data_path': '../fit_lens_light/F277W/svi_lens_light_fit_10/lens_light_subtracted.fits',
        'data_path': '../data/F277W/Data_cutout.fits',
        'noise_path': '../data/F277W/noise.fits',
        'psf_path': '../psf/F277W/psf_modelled.fits', 
        'source_arc_mask_path': '../data/F277W/mask_1.fits',
        'save_path': '../modeling_F277W/parametric_updated_params',

        'pipeline': False, 
        'source_grid_scale': 0.8, 
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
        'sampler': 'svi',
        # Path to a prior run directory (e.g. a parametric SVI run folder).
        # Loads kwargs_result.json from a prior run directory to warm-start
        # the next step — use this to run pixelated SVI after parametric SVI.
        # Example pipeline:
        'init_params_path': None,
        # 'init_params_path': 'modeling_sim/svi_parametric_run_2/run_1',
        'refine_prior_range': None,
        'refine_prior_min_frac': None,
        'pixel_init_jitter': 0.0,
        'fix_component': [],  # list of components to fix: 'lens_mass' | 'lens_light' | 'source_light'
        'regul_num_samples': 1000,
        'gpus': '7',
        'n_runs': 4,

        # --- svi (Stochastic Variational Inference) ---
        'pixelated_init_method': 'svi_warmup', 
        'max_iterations_power_init': 2_000, 
        'max_iterations_svi_warmup': 2_000,
        'max_iterations_svi': 10_000,
        'init_learning_rate_svi': 1e-2,
        'init_scale_svi': 0.1,
        'loss_kind_svi': 'trace_elbo',  # trace_elbo | trace_meanfield_elbo
        'num_particles_svi': 10,

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
