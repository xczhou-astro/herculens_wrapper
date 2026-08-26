"""Shared public-API types and legacy-wrapper key mappings."""

from typing import Literal

ComponentName = Literal["lens_mass", "lens_light", "source_light", "point_source"]
SamplerName = Literal["optax", "svi", "hmc"]

TYPE_KEYS = {"lens_mass": "lens_mass_type_list", "lens_light": "lens_light_type_list", "source_light": "source_light_type_list", "point_source": "point_source_type_list"}
PARAM_KEYS = {"lens_mass": "lens_mass_params_list", "lens_light": "lens_light_params_list", "source_light": "source_light_params_list", "point_source": "point_source_params_list"}
