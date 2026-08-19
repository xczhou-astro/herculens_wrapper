"""Joint multi-band NumPyro model with one shared lens-mass block."""

import re

import numpyro
from herculens.Inference.ProbModel.numpyro import NumpyroModel

from herculens_wrapper.models import (
    _normalize_link_spec,
    _resolve_link,
    _sample_param_from_prior,
    create_prob_model,
)


_BAND_SPECIFIC_LENS_MASS_KEYS = frozenset({'center_x', 'center_y'})


def band_site_prefix(index, band_name):
    """Return a stable NumPyro-safe namespace for a band."""
    label = re.sub(r'[^0-9A-Za-z_]+', '_', str(band_name)).strip('_') or 'band'
    return f'band_{index}_{label}'


def create_multiband_prob_model(
    bands,
    lens_mass_params_list,
    lens_mass_type_list,
    args,
    fixed_lens_mass=None,
    fixed_lens_mass_by_band=None,
    fixed_lens_light_by_band=None,
):
    """Build a joint likelihood with shared mass shape and per-band mass centres."""
    if not bands:
        raise ValueError('At least one band is required for a multiband model.')

    holders = []
    band_models = []
    for band in bands:
        holder = {}
        holders.append(holder)
        band_models.append(create_prob_model(
            band['param_list'],
            band['type_list'],
            band['lens_image'],
            band['image_data'],
            band['noise_map'],
            args=band.get('args', args),
            fix_lens_mass=True,
            kwargs_lens_fixed=lambda holder=holder: holder['kwargs_lens'],
            fix_lens_light=fixed_lens_light_by_band is not None,
            kwargs_lens_light_fixed=(
                fixed_lens_light_by_band[band['name']]
                if fixed_lens_light_by_band is not None else None
            ),
        ))

    def _site_value(params, band, site):
        joint_site = f"{band['site_prefix']}/{site}"
        if joint_site in params:
            return params[joint_site]
        return params[site]

    def _sample_shared_mass_values():
        values = {}
        for index, mass_model in enumerate(lens_mass_params_list):
            for key, param in mass_model.items():
                if key in _BAND_SPECIFIC_LENS_MASS_KEYS:
                    continue
                if _normalize_link_spec(param) is None and isinstance(param, (list, tuple)):
                    site = f'lens_{key}_{index}'
                    values[(index, key)] = _sample_param_from_prior(site, key, param)
        return values

    def mass_kwargs_from_params(params, band, sample_centers=False, shared_values=None):
        kwargs_lens = []
        bank = {'lens': kwargs_lens}
        band_mass_models = band['param_list']['lens_mass_params_list']
        for index, shared_mass_model in enumerate(lens_mass_params_list):
            band_mass_model = band_mass_models[index]
            kwargs = {}
            for key, shared_param in shared_mass_model.items():
                param = (
                    band_mass_model.get(key, shared_param)
                    if key in _BAND_SPECIFIC_LENS_MASS_KEYS else shared_param
                )
                link_spec = _normalize_link_spec(param)
                if link_spec is not None:
                    kwargs[key] = _resolve_link(bank, link_spec, context=f'multiband lens_mass[{index}].{key}')
                elif isinstance(param, (list, tuple)):
                    site = f'lens_{key}_{index}'
                    if key in _BAND_SPECIFIC_LENS_MASS_KEYS:
                        kwargs[key] = (
                            _sample_param_from_prior(site, key, param)
                            if sample_centers else _site_value(params, band, site)
                        )
                    elif shared_values is not None:
                        kwargs[key] = shared_values[(index, key)]
                    else:
                        kwargs[key] = params[site]
                else:
                    kwargs[key] = param
            kwargs_lens.append(kwargs)
        return kwargs_lens

    def fixed_mass_for_band(band):
        if fixed_lens_mass_by_band is not None:
            return fixed_lens_mass_by_band[band['name']]
        return fixed_lens_mass

    def posterior_site_order():
        """Return joint latent sites in the configured component order."""
        order = []

        for index, mass_model in enumerate(lens_mass_params_list):
            if isinstance(mass_model, dict):
                order.extend(
                    f'lens_{key}_{index}' for key in mass_model
                    if key not in _BAND_SPECIFIC_LENS_MASS_KEYS
                )

        for band in bands:
            prefix = f"{band['site_prefix']}/"
            param_list = band['param_list']
            for index, mass_model in enumerate(param_list.get('lens_mass_params_list', [])):
                if isinstance(mass_model, dict):
                    order.extend(
                        f'{prefix}lens_{key}_{index}'
                        for key, value in mass_model.items()
                        if key in _BAND_SPECIFIC_LENS_MASS_KEYS
                        and _normalize_link_spec(value) is None
                        and isinstance(value, (list, tuple))
                    )
            for index, light_model in enumerate(param_list.get('lens_light_params_list', [])):
                if isinstance(light_model, dict):
                    order.extend(
                        f'{prefix}lens_light_{key}_{index}' for key in light_model
                    )

            source_types = band['type_list'].get('source_light_type_list', [])
            if source_types == ['PIXELATED']:
                order.extend(
                    f'{prefix}{key}' for key in (
                        'n_source_grid', 'rho_source_grid',
                        'sigma_source_grid', 'pixels_wn_source_grid',
                    )
                )
            else:
                for index, source_model in enumerate(param_list.get('source_light_params_list', [])):
                    if isinstance(source_model, dict):
                        order.extend(
                            f'{prefix}source_{key}_{index}' for key in source_model
                        )

            for index, point_source_model in enumerate(param_list.get('point_source_params_list', [])):
                if isinstance(point_source_model, dict):
                    order.extend(f'{prefix}ps_{key}_{index}' for key in point_source_model)
        return order

    class MultiBandProbModel(NumpyroModel):
        def model(self):
            fixed_mass_mode = fixed_lens_mass is not None or fixed_lens_mass_by_band is not None
            shared_values = None if fixed_mass_mode else _sample_shared_mass_values()
            for band, holder, band_model in zip(bands, holders, band_models):
                if fixed_mass_mode:
                    kwargs_lens = fixed_mass_for_band(band)
                else:
                    with numpyro.handlers.scope(prefix=band['site_prefix']):
                        kwargs_lens = mass_kwargs_from_params(
                            {}, band, sample_centers=True, shared_values=shared_values,
                        )
                holder['kwargs_lens'] = kwargs_lens
                numpyro.handlers.scope(band_model.model, prefix=band['site_prefix'])()

        def params2kwargs_by_band(self, params):
            results = {}
            for band, holder, band_model in zip(bands, holders, band_models):
                fixed_mass_mode = fixed_lens_mass is not None or fixed_lens_mass_by_band is not None
                kwargs_lens = (
                    fixed_mass_for_band(band)
                    if fixed_mass_mode else mass_kwargs_from_params(params, band)
                )
                prefix = f"{band['site_prefix']}/"
                band_params = {
                    key[len(prefix):]: value
                    for key, value in params.items()
                    if key.startswith(prefix)
                }
                results[band['name']] = band_model.params2kwargs(
                    band_params, kwargs_lens_override=kwargs_lens,
                )
            return results

        def params2kwargs(self, params):
            kwargs_by_band = self.params2kwargs_by_band(params)
            return {
                'kwargs_lens': kwargs_by_band[bands[0]['name']]['kwargs_lens'],
                'kwargs_by_band': kwargs_by_band,
            }

    model = MultiBandProbModel()
    model.bands = bands
    model.band_models = band_models
    model.lens_mass_params_list = lens_mass_params_list
    model.lens_mass_type_list = lens_mass_type_list
    model.posterior_site_order = posterior_site_order
    model.mass_kwargs_from_params = lambda params: mass_kwargs_from_params(params, bands[0])
    for band, band_model in zip(bands, band_models):
        band_model.mass_kwargs_from_params = (
            lambda params, band=band: mass_kwargs_from_params(params, band)
        )
    model.type_list = {'lens_mass_type_list': lens_mass_type_list}
    model.band_specific_lens_mass_keys = _BAND_SPECIFIC_LENS_MASS_KEYS
    return model


def _create_fully_shared_multidata_prob_model(
    bands, args, fixed_lens_mass=None, fixed_lens_light=None,
):
    """Build one physical model observed through multiple data/PSF/noise realizations.

    Unlike multi-band fitting, all lens mass, lens light, source light, and point
    source parameters are shared. Each entry in ``bands`` contributes only an
    additional image likelihood through its own ``LensImage`` instance.
    """
    if not bands:
        raise ValueError('At least one observation is required for a multi-data model.')

    reference = bands[0]
    reference_param_list = reference['param_list']
    reference_type_list = reference['type_list']
    for band in bands[1:]:
        if band['type_list'] != reference_type_list or band['param_list'] != reference_param_list:
            raise ValueError(
                'Multi-data observations must use identical model types and parameter priors.'
            )

    additional_observations = [
        {
            'lens_image': band['lens_image'],
            'image_data': band['image_data'],
            'likelihood_scale': getattr(band.get('args', args), 'likelihood_scale', 1.0),
        }
        for band in bands[1:]
    ]
    model = create_prob_model(
        reference_param_list,
        reference_type_list,
        reference['lens_image'],
        reference['image_data'],
        reference['noise_map'],
        args=reference.get('args', args),
        fix_lens_mass=fixed_lens_mass is not None,
        kwargs_lens_fixed=fixed_lens_mass,
        fix_lens_light=fixed_lens_light is not None,
        kwargs_lens_light_fixed=fixed_lens_light,
        additional_observations=additional_observations,
    )

    def params2kwargs_by_band(params):
        # Return distinct mappings because downstream result assembly adds
        # derived pixel summaries to each observation's kwargs dictionary.
        return {band['name']: model.params2kwargs(params) for band in bands}

    def posterior_site_order():
        order = []
        for index, mass_model in enumerate(reference_param_list.get('lens_mass_params_list', [])):
            if isinstance(mass_model, dict):
                order.extend(f'lens_{key}_{index}' for key in mass_model)
        for index, light_model in enumerate(reference_param_list.get('lens_light_params_list', [])):
            if isinstance(light_model, dict):
                order.extend(f'lens_light_{key}_{index}' for key in light_model)
        if reference_type_list.get('source_light_type_list') == ['PIXELATED']:
            order.extend(('n_source_grid', 'rho_source_grid', 'sigma_source_grid', 'pixels_wn_source_grid'))
        else:
            for index, source_model in enumerate(reference_param_list.get('source_light_params_list', [])):
                if isinstance(source_model, dict):
                    order.extend(f'source_{key}_{index}' for key in source_model)
        for index, point_source_model in enumerate(reference_param_list.get('point_source_params_list', [])):
            if isinstance(point_source_model, dict):
                order.extend(f'ps_{key}_{index}' for key in point_source_model)
        return order

    model.bands = bands
    model.band_models = [model] * len(bands)
    model.lens_mass_params_list = reference_param_list['lens_mass_params_list']
    model.lens_mass_type_list = reference_type_list['lens_mass_type_list']
    model.params2kwargs_by_band = params2kwargs_by_band
    model.posterior_site_order = posterior_site_order
    model.mass_kwargs_from_params = lambda params: model.params2kwargs(params)['kwargs_lens']
    model.shared_observation_model = True
    model.fully_shared_lens_mass = True
    model.selective_shared_model = False
    return model


_SHARED_COMPONENT_ALIASES = {
    'lens_mass': 'lens_mass',
    'lens mass': 'lens_mass',
    'mass': 'lens_mass',
    'lens_light': 'lens_light',
    'lens light': 'lens_light',
    'source_light': 'source_light',
    'source light': 'source_light',
    'source': 'source_light',
    'point_source': 'point_source',
    'point source': 'point_source',
    'ps': 'point_source',
}

_SHARED_COMPONENT_CONFIG = {
    'lens_mass': ('lens_mass_params_list', 'lens'),
    'lens_light': ('lens_light_params_list', 'lens_light'),
    'source_light': ('source_light_params_list', 'source'),
    'point_source': ('point_source_params_list', 'ps'),
}


def _canonical_shared_component(value):
    normalized = re.sub(r'[_\-\s]+', ' ', str(value).strip().lower())
    return _SHARED_COMPONENT_ALIASES.get(normalized)


def _shared_site_name(component, key, index):
    return f'{_SHARED_COMPONENT_CONFIG[component][1]}_{key}_{index}'


def _shareable_entries(param_list, type_list, component):
    """Return ordinary NumPyro sites which can be tied across observations."""
    list_key, _ = _SHARED_COMPONENT_CONFIG[component]
    if component == 'source_light' and type_list.get('source_light_type_list') == ['PIXELATED']:
        return []
    entries = []
    for index, model in enumerate(param_list.get(list_key, [])):
        if not isinstance(model, dict):
            continue
        for key, prior in model.items():
            if component == 'point_source' and key in ('n_images', 'sigma_image', 'sigma_source'):
                continue
            if _normalize_link_spec(prior) is not None or not isinstance(prior, (list, tuple)):
                continue
            entries.append((component, index, key, prior, _shared_site_name(component, key, index)))
    return entries


def _local_site_order(param_list, type_list):
    order = []
    for component in _SHARED_COMPONENT_CONFIG:
        if component == 'source_light' and type_list.get('source_light_type_list') == ['PIXELATED']:
            order.extend((
                'n_source_grid', 'rho_source_grid', 'sigma_source_grid',
                'pixels_wn_source_grid',
            ))
            continue
        order.extend(entry[-1] for entry in _shareable_entries(param_list, type_list, component))
    return order


def _parse_shared_specs(shared_specs, param_list, type_list):
    """Parse ``args.shared`` and return the selected local site names."""
    if shared_specs is None:
        shared_specs = []
    if not isinstance(shared_specs, (list, tuple)) or not all(
        isinstance(spec, str) for spec in shared_specs
    ):
        raise TypeError('shared must be a list of strings, for example ["lens_mass: theta_E"].')

    entries_by_component = {
        component: _shareable_entries(param_list, type_list, component)
        for component in _SHARED_COMPONENT_CONFIG
    }
    selected = {}
    requested_pixelated_source = False
    for raw_spec in shared_specs:
        component_text, separator, selector_text = raw_spec.partition(':')
        component = _canonical_shared_component(component_text)
        if component is None:
            raise ValueError(
                f'Unknown shared component {component_text!r}. Expected lens_mass, lens_light, '
                'source_light, or point_source.'
            )
        selector = selector_text.strip() if separator else ''
        if component == 'source_light' and type_list.get('source_light_type_list') == ['PIXELATED']:
            requested_pixelated_source = True
            continue
        requested_keys = None if not selector or selector.lower() == 'all' else {
            value.strip() for value in selector.split(',') if value.strip()
        }
        matched = [
            entry for entry in entries_by_component[component]
            if requested_keys is None or entry[2] in requested_keys
        ]
        if requested_keys is not None:
            found_keys = {entry[2] for entry in matched}
            missing = requested_keys - found_keys
            if missing:
                raise ValueError(
                    f"shared spec {raw_spec!r} does not match sampled {component} parameter(s): "
                    f'{sorted(missing)}.'
                )
        for entry in matched:
            selected[entry[-1]] = entry
    return selected, requested_pixelated_source


def _all_existing_components_requested(shared_specs, param_list, type_list):
    """Whether explicit specs request every stochastic physical component."""
    if shared_specs == 'all':
        return True
    parsed = _parse_shared_specs(shared_specs, param_list, type_list)
    selected, pixelated_source = parsed
    all_entries = [
        entry
        for component in _SHARED_COMPONENT_CONFIG
        for entry in _shareable_entries(param_list, type_list, component)
    ]
    all_standard_sites = {entry[-1] for entry in all_entries}
    source_is_pixelated = type_list.get('source_light_type_list') == ['PIXELATED']
    return (
        set(selected) == all_standard_sites
        and (not source_is_pixelated or pixelated_source)
    )


def _create_selectively_shared_multidata_prob_model(bands, args, shared_entries):
    """Joint multi-data model with selected unscoped sites and local remainder."""
    reference = bands[0]
    holders = []
    band_models = []
    for band in bands:
        holder = {}
        holders.append(holder)
        child_model = create_prob_model(
            band['param_list'],
            band['type_list'],
            band['lens_image'],
            band['image_data'],
            band['noise_map'],
            args=band.get('args', args),
            param_overrides=lambda holder=holder: holder['overrides'],
        )
        band_models.append(child_model)

    shared_site_names = set(shared_entries)

    def shared_overrides(params, sample=False):
        overrides = {
            component: [{} for _ in reference['param_list'].get(list_key, [])]
            for component, (list_key, _) in _SHARED_COMPONENT_CONFIG.items()
        }
        for site_name, (component, index, key, prior, _) in shared_entries.items():
            value = _sample_param_from_prior(site_name, key, prior) if sample else params[site_name]
            overrides[component][index][key] = value
        return overrides

    def local_params_from_joint(params, band):
        prefix = f"{band['site_prefix']}/"
        local = {
            key[len(prefix):]: value
            for key, value in params.items()
            if key.startswith(prefix)
        }
        for site_name in shared_site_names:
            if site_name in params:
                local[site_name] = params[site_name]
        return local

    def joint_site_name(band, local_site_name):
        if local_site_name in shared_site_names:
            return local_site_name
        return f"{band['site_prefix']}/{local_site_name}"

    def posterior_site_order():
        order = list(shared_entries)
        for band in bands:
            for site_name in _local_site_order(band['param_list'], band['type_list']):
                if site_name not in shared_site_names:
                    order.append(joint_site_name(band, site_name))
        return order

    class SelectiveMultiDataProbModel(NumpyroModel):
        def model(self):
            overrides = shared_overrides({}, sample=True)
            for holder in holders:
                holder['overrides'] = overrides
            for band, band_model in zip(bands, band_models):
                numpyro.handlers.scope(band_model.model, prefix=band['site_prefix'])()

        def params2kwargs_by_band(self, params):
            return {
                band['name']: band_model.params2kwargs(local_params_from_joint(params, band))
                for band, band_model in zip(bands, band_models)
            }

        def params2kwargs(self, params):
            return {'kwargs_by_band': self.params2kwargs_by_band(params)}

    model = SelectiveMultiDataProbModel()
    for band, band_model in zip(bands, band_models):
        band_model.joint_params_to_local = lambda params, band=band: local_params_from_joint(params, band)
        band_model.mass_kwargs_from_params = lambda params, band_model=band_model: (
            band_model.params2kwargs(params)['kwargs_lens']
        )
    all_mass_sites = {
        entry[-1] for entry in _shareable_entries(
            reference['param_list'], reference['type_list'], 'lens_mass',
        )
    }
    model.bands = bands
    model.band_models = band_models
    model.lens_mass_params_list = reference['param_list']['lens_mass_params_list']
    model.lens_mass_type_list = reference['type_list']['lens_mass_type_list']
    model.posterior_site_order = posterior_site_order
    model.joint_site_name = joint_site_name
    model.shared_site_names = shared_site_names
    model.shared_entries = shared_entries
    model.shared_observation_model = False
    model.selective_shared_model = True
    model.fully_shared_lens_mass = all_mass_sites.issubset(shared_site_names)
    model.type_list = {'lens_mass_type_list': model.lens_mass_type_list}
    return model


def create_multidata_source_warmup_model(
    bands, args, kwargs_lens_by_band, kwargs_lens_light_by_band,
):
    """Optimize independent pixelated sources with each observation's light fixed."""
    band_models = []
    for band in bands:
        band_models.append(create_prob_model(
            band['param_list'],
            band['type_list'],
            band['lens_image'],
            band['image_data'],
            band['noise_map'],
            args=band.get('args', args),
            fix_lens_mass=True,
            kwargs_lens_fixed=kwargs_lens_by_band[band['name']],
            fix_lens_light=True,
            kwargs_lens_light_fixed=kwargs_lens_light_by_band[band['name']],
        ))

    class MultiDataSourceWarmupModel(NumpyroModel):
        def model(self):
            for band, band_model in zip(bands, band_models):
                numpyro.handlers.scope(band_model.model, prefix=band['site_prefix'])()

    model = MultiDataSourceWarmupModel()
    model.band_models = band_models
    return model


def create_multidata_prob_model(bands, args, fixed_lens_mass=None, fixed_lens_light=None):
    """Build a joint same-band model with user-selected shared parameters.

    Omitting ``args.shared`` shares nothing. Use ``shared='all'`` to share
    every physical component, or a list of component/parameter specifications
    to tie only selected sites.
    """
    if not bands:
        raise ValueError('At least one observation is required for a multi-data model.')
    reference = bands[0]
    for band in bands[1:]:
        if band['type_list'] != reference['type_list'] or band['param_list'] != reference['param_list']:
            raise ValueError(
                'Multi-data observations must use identical model types and parameter priors.'
            )

    shared_specs = getattr(args, 'shared', [])
    if _all_existing_components_requested(
        shared_specs, reference['param_list'], reference['type_list'],
    ):
        return _create_fully_shared_multidata_prob_model(
            bands, args, fixed_lens_mass=fixed_lens_mass, fixed_lens_light=fixed_lens_light,
        )

    if fixed_lens_mass is not None or fixed_lens_light is not None:
        raise ValueError(
            'Selective multi-data sharing does not support fixed lens components in this model. '
            'Use the dedicated per-observation source warm-up model instead.'
        )
    shared_entries, requested_pixelated_source = _parse_shared_specs(
        shared_specs, reference['param_list'], reference['type_list'],
    )
    if requested_pixelated_source:
        raise ValueError(
            'Selective sharing of a PIXELATED source is not supported. Share every physical '
            'component (the default when shared is omitted), or leave source_light unshared.'
        )
    return _create_selectively_shared_multidata_prob_model(bands, args, shared_entries)
