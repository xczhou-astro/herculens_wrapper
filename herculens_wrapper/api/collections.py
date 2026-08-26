"""Physical profile collections for lensing models."""

from __future__ import annotations

from copy import deepcopy
from pprint import pformat
from typing import Any, Sequence

from .parameters import LightProfile, MassProfile, PointSourceProfile, Profile, ProfileCollection
from .types import ComponentName


class LensProfileCollection:
    """Profiles grouped by their physical role in a lensing system."""

    _components = ("lens_mass", "lens_light", "source_light", "point_source")

    def __init__(
        self,
        *,
        lens_mass: Profile | ProfileCollection | Sequence[Profile] | None = None,
        lens_light: Profile | ProfileCollection | Sequence[Profile] | None = None,
        source_light: Profile | ProfileCollection | Sequence[Profile] | None = None,
        point_source: Profile | ProfileCollection | Sequence[Profile] | None = None,
    ) -> None:
        for component, profiles in zip(self._components, (lens_mass, lens_light, source_light, point_source)):
            setattr(self, component, self._coerce_profiles(component, profiles))

    @staticmethod
    def _coerce_profiles(
        component: ComponentName, profiles: Profile | ProfileCollection | Sequence[Profile] | None,
    ) -> ProfileCollection | None:
        if profiles is None:
            return None
        collection = profiles if isinstance(profiles, ProfileCollection) else ProfileCollection(
            [profiles] if isinstance(profiles, Profile) else list(profiles)
        )
        expected = MassProfile if component == "lens_mass" else PointSourceProfile if component == "point_source" else LightProfile
        if any(not isinstance(profile, expected) for profile in collection):
            raise TypeError(f"{component} requires {expected.__name__} instances.")
        return collection

    @property
    def configuration(self) -> dict[str, list[dict[str, Any]]]:
        """A notebook-friendly, complete view of the declared profile state."""
        result: dict[str, list[dict[str, Any]]] = {}
        for component in self._components:
            profiles = getattr(self, component)
            if profiles is None:
                continue
            result[component] = []
            for profile in profiles:
                parameters: dict[str, dict[str, Any]] = {}
                for name, parameter in profile._parameters.items():
                    link = profile._specification(name)["link"]
                    entry: dict[str, Any] = {"prior": parameter.prior, "value": parameter.value}
                    if link is not None:
                        entry["linked_to"] = f"{link._profile.profile_type}.{link.name}"
                    parameters[name] = entry
                result[component].append({"profile": profile.profile_type, "parameters": parameters})
        return result

    @property
    def values(self) -> dict[str, list[dict[str, Any]]]:
        """Current parameter values, grouped by physical component."""
        return {
            component: profiles.values
            for component in self._components
            if (profiles := getattr(self, component)) is not None
        }

    @property
    def priors(self) -> dict[str, list[dict[str, Any]]]:
        """Prior definitions, grouped by physical component."""
        return {
            component: profiles.priors
            for component in self._components
            if (profiles := getattr(self, component)) is not None
        }

    def as_definition(self):
        """Convert the declarations to the backend-facing model definition."""
        from .models import ModelDefinition

        definition = ModelDefinition()
        for component in self._components:
            profiles = getattr(self, component)
            if profiles is not None:
                definition.add_profiles(component, profiles)
        return definition

    def freeze(self) -> "LensProfileCollection":
        return LensProfileCollection(**{
            component: ProfileCollection([profile.freeze() for profile in profiles])
            for component in self._components if (profiles := getattr(self, component)) is not None
        })

    def with_fixed(self, **selections: Any) -> "LensProfileCollection":
        """Return a copy with selected parameters converted to fixed values.

        Each keyword names a physical component.  ``True`` fixes every
        parameter in that component; a dictionary selects individual profile
        indices and parameters.  Values are read from ``Parameter.value``.

        Examples
        --------
        ``profiles.with_fixed(lens_light=True)``

        ``profiles.with_fixed(lens_mass={0: ["center_x", "center_y"]})``

        ``profiles.with_fixed(lens_mass={0: {"center_x": 0.0}})``
        """
        unknown = set(selections) - set(self._components)
        if unknown:
            raise KeyError(f"Unknown model component(s): {sorted(unknown)}.")

        fixed: dict[tuple[str, int, str], Any] = {}
        for component, selection in selections.items():
            profiles = getattr(self, component)
            if profiles is None:
                raise ValueError(f"Cannot fix {component}: it is not present.")
            profile_selections = (
                {index: True for index in range(len(profiles))}
                if selection is True else selection
            )
            if not isinstance(profile_selections, dict):
                raise TypeError(
                    f"{component} must be True or a dictionary such as "
                    "{0: ['center_x', 'center_y']}."
                )
            for index, parameter_selection in profile_selections.items():
                if not isinstance(index, int) or not 0 <= index < len(profiles):
                    raise IndexError(f"{component} profile index {index!r} is invalid.")
                profile = profiles[index]
                if parameter_selection is True:
                    parameters = {name: None for name in profile._parameters}
                elif isinstance(parameter_selection, dict):
                    parameters = dict(parameter_selection)
                else:
                    if isinstance(parameter_selection, str):
                        parameter_selection = [parameter_selection]
                    parameters = {name: None for name in parameter_selection}
                for name, explicit_value in parameters.items():
                    if name not in profile._parameters:
                        raise KeyError(f"{component}[{index}] has no parameter {name!r}.")
                    if explicit_value is None:
                        parameter = profile.parameter(name)
                        value = parameter.value
                        if value is None and not isinstance(parameter.prior, (list, tuple)):
                            value = parameter.prior
                        if value is None:
                            raise ValueError(
                                f"Cannot fix {component}[{index}].{name}: its value is unset. "
                                "Call initialize() or set parameter.value first."
                            )
                    else:
                        value = explicit_value
                    fixed[(component, index, name)] = deepcopy(value)

        clones: dict[tuple[str, int], Profile] = {}
        parameter_map: dict[int, Parameter] = {}
        for component in self._components:
            profiles = getattr(self, component)
            if profiles is None:
                continue
            profile_class = MassProfile if component == "lens_mass" else (
                PointSourceProfile if component == "point_source" else LightProfile
            )
            for index, profile in enumerate(profiles):
                prior = {
                    name: deepcopy(profile.parameter(name).prior)
                    for name in profile._parameters
                }
                value = {
                    name: deepcopy(profile._specification(name)["value"])
                    for name in profile._parameters
                    if profile._specification(name)["value"] is not None
                }
                clone = profile_class(profile.profile_type, prior=prior, value=value)
                clones[(component, index)] = clone
                for name, parameter in profile._parameters.items():
                    parameter_map[id(parameter)] = clone.parameter(name)

        # Restore non-fixed relationships after every clone exists.
        for component in self._components:
            profiles = getattr(self, component)
            if profiles is None:
                continue
            for index, profile in enumerate(profiles):
                clone = clones[(component, index)]
                for name, parameter in profile._parameters.items():
                    fixed_value = fixed.get((component, index, name))
                    if (component, index, name) in fixed:
                        clone.parameter(name).prior = fixed_value
                        clone.parameter(name).value = fixed_value
                    else:
                        link = profile._specification(name)["link"]
                        if link is not None:
                            clone.parameter(name).link_to(parameter_map[id(link)])

        return LensProfileCollection(**{
            component: ProfileCollection([
                clones[(component, index)] for index in range(len(profiles))
            ])
            for component in self._components
            if (profiles := getattr(self, component)) is not None
        })

    def __repr__(self) -> str:
        return f"LensProfileCollection(\n{pformat(self.configuration, sort_dicts=False)}\n)"

    __str__ = __repr__
