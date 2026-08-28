"""Conversion of profile declarations to the legacy wrapper representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .parameters import LightProfile, MassProfile, Parameter, PointSourceProfile, Profile, ProfileCollection
from .types import ComponentName, PARAM_KEYS, TYPE_KEYS


def is_pixelated_latent_site(name: str) -> bool:
    """Whether a NumPyro site is a pixel reconstruction nuisance variable."""
    site = str(name).rsplit("/", 1)[-1]
    return (
        site.startswith("pixels_wn_source_grid")
        or site.startswith("pixels_wn_lens_light_grid_")
        or site in {
            "n_source_grid", "rho_source_grid", "sigma_source_grid",
            "source_scales", "source_coarse", "source_pixels",
        }
        or site.startswith(("n_lens_light_grid_", "rho_lens_light_grid_", "sigma_lens_light_grid_"))
    )


def count_physical_parameters(parameters: Mapping[str, Any]) -> int:
    """Count free non-pixelated parameters for the reported physical BIC.

    Pixel-grid coefficients and their Matérn (or legacy wavelet) nuisance
    variables are deliberately excluded.  Lens mass/light, point-source, and
    analytic source parameters remain included.
    """
    return int(sum(
        np.asarray(value).size
        for name, value in parameters.items()
        if not is_pixelated_latent_site(name)
    ))


def _expected_profile_class(component: ComponentName) -> type[Profile]:
    return MassProfile if component == "lens_mass" else PointSourceProfile if component == "point_source" else LightProfile


@dataclass
class ModelDefinition:
    """Backend-facing representation produced from a profile collection."""

    types: dict[str, list[str]] = field(default_factory=lambda: {key: [] for key in TYPE_KEYS.values()})
    parameters: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {key: [] for key in PARAM_KEYS.values()})
    _locations: dict[int, tuple[ComponentName, int]] = field(default_factory=dict, init=False, repr=False)
    _profiles: dict[tuple[ComponentName, int], Profile] = field(default_factory=dict, init=False, repr=False)

    def add(self, component: ComponentName, profile: str, parameters: Mapping[str, Any]) -> "ModelDefinition":
        self.types[TYPE_KEYS[component]].append(str(profile))
        self.parameters[PARAM_KEYS[component]].append(dict(parameters))
        return self

    def _materialize_profile(self, profile: Profile) -> dict[str, Any]:
        params = {}
        for name, value in profile.parameters.items():
            if isinstance(value, Parameter):
                try:
                    source_component, source_index = self._locations[id(value._profile)]
                except KeyError as error:
                    raise ValueError("A linked profile must be added before its dependent profile.") from error
                value = ["correlated", source_component, source_index, value.name]
            params[name] = value
        return params

    def _refresh_object_profiles(self) -> None:
        for (component, index), profile in self._profiles.items():
            self.parameters[PARAM_KEYS[component]][index] = self._materialize_profile(profile)

    def add_profiles(self, component: ComponentName, profiles: Profile | ProfileCollection | Sequence[Profile]) -> "ModelDefinition":
        collection = profiles if isinstance(profiles, ProfileCollection) else ProfileCollection(
            [profiles] if isinstance(profiles, Profile) else list(profiles)
        )
        expected = _expected_profile_class(component)
        if any(not isinstance(profile, expected) for profile in collection):
            raise TypeError(f"{component} requires {expected.__name__} instances.")
        for profile in collection:
            self.add(component, profile.profile_type, {})
            index = len(self.parameters[PARAM_KEYS[component]]) - 1
            self._locations[id(profile)], self._profiles[(component, index)] = (component, index), profile
            self.parameters[PARAM_KEYS[component]][index] = self._materialize_profile(profile)
        return self

    def as_dicts(self) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
        self._refresh_object_profiles()
        return (
            {key: list(value) for key, value in self.types.items()},
            {key: [dict(item) for item in value] for key, value in self.parameters.items()},
        )

    @property
    def has_free_parameters(self) -> bool:
        self._refresh_object_profiles()
        return any(
            isinstance(value, (list, tuple)) and not (len(value) == 4 and value[0] == "correlated")
            for components in self.parameters.values() for profile in components for value in profile.values()
        )

    def update_values(self, flat_parameters: Mapping[str, Any]) -> None:
        prefixes = {"lens_mass": "lens_", "lens_light": "lens_light_", "source_light": "source_", "point_source": "ps_"}
        for (component, index), profile in self._profiles.items():
            for name in profile._parameters:
                site = f"{prefixes[component]}{name}_{index}"
                if site in flat_parameters:
                    profile.parameter(name).value = flat_parameters[site]

    def profile(self, component: ComponentName, index: int = 0) -> Profile:
        try:
            return self._profiles[(component, index)]
        except KeyError as error:
            raise KeyError(f"No {component!r} profile at index {index}.") from error

    def freeze(self) -> "ModelDefinition":
        result = ModelDefinition()
        for component, index in sorted(self._profiles, key=lambda item: (PARAM_KEYS[item[0]], item[1])):
            profile = self._profiles[(component, index)].freeze()
            result.add(component, profile.profile_type, profile.parameters)
        return result
