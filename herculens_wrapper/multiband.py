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
    fixed_lens_light_by_band=None,
):
    """Build a joint likelihood with shared mass and band-specific light models."""
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
            args=args,
            fix_lens_mass=True,
            kwargs_lens_fixed=lambda holder=holder: holder['kwargs_lens'],
            fix_lens_light=fixed_lens_light_by_band is not None,
            kwargs_lens_light_fixed=(
                fixed_lens_light_by_band[band['name']]
                if fixed_lens_light_by_band is not None else None
            ),
        ))

    def mass_kwargs_from_params(params, sample=False):
        kwargs_lens = []
        bank = {'lens': kwargs_lens}
        for index, mass_model in enumerate(lens_mass_params_list):
            kwargs = {}
            for key, param in mass_model.items():
                link_spec = _normalize_link_spec(param)
                if link_spec is not None:
                    kwargs[key] = _resolve_link(bank, link_spec, context=f'multiband lens_mass[{index}].{key}')
                elif isinstance(param, (list, tuple)):
                    site = f'lens_{key}_{index}'
                    kwargs[key] = _sample_param_from_prior(site, key, param) if sample else params[site]
                else:
                    kwargs[key] = param
            kwargs_lens.append(kwargs)
        return kwargs_lens

    class MultiBandProbModel(NumpyroModel):
        def model(self):
            kwargs_lens = (
                fixed_lens_mass
                if fixed_lens_mass is not None
                else mass_kwargs_from_params({}, sample=True)
            )
            for holder in holders:
                holder['kwargs_lens'] = kwargs_lens
            for band, band_model in zip(bands, band_models):
                numpyro.handlers.scope(band_model.model, prefix=band['site_prefix'])()

        def params2kwargs_by_band(self, params):
            kwargs_lens = (
                fixed_lens_mass
                if fixed_lens_mass is not None
                else mass_kwargs_from_params(params, sample=False)
            )
            results = {}
            for band, holder, band_model in zip(bands, holders, band_models):
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
            return {
                'kwargs_lens': (
                    fixed_lens_mass
                    if fixed_lens_mass is not None
                    else mass_kwargs_from_params(params, sample=False)
                ),
                'kwargs_by_band': self.params2kwargs_by_band(params),
            }

    model = MultiBandProbModel()
    model.bands = bands
    model.band_models = band_models
    model.lens_mass_params_list = lens_mass_params_list
    model.lens_mass_type_list = lens_mass_type_list
    model.mass_kwargs_from_params = lambda params: mass_kwargs_from_params(
        params, sample=False,
    )
    model.type_list = {'lens_mass_type_list': lens_mass_type_list}
    return model
