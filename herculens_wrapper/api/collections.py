"""Physical profile collections for lensing models."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Sequence

from .parameters import (
    LightProfile,
    MassProfile,
    Parameter,
    PointSourceProfile,
    Profile,
    ProfileCollection,
    StellarMassMGE,
)
from .models import ComponentName


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
        if isinstance(profiles, ProfileCollection):
            collection = profiles
        elif isinstance(profiles, Profile):
            collection = ProfileCollection([profiles])
        else:
            # Multi-profile constructors such as
            # ``LightProfile(["GAUSSIAN_ELLIPSE"] * 3, ...)`` return a
            # ProfileCollection.  Allow it to be combined naturally with
            # another profile, e.g. ``[mge_bulge, PixelatedLensLight()]``.
            flattened = []
            for profile in profiles:
                if isinstance(profile, ProfileCollection):
                    flattened.extend(profile)
                else:
                    flattened.append(profile)
            collection = ProfileCollection(flattened)
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
                entry = {"profile": profile.profile_type, "parameters": parameters}
                if isinstance(profile, StellarMassMGE) and profile.follows_lens_light:
                    entry["follow_lens_light"] = True
                # Profile.__getattr__ deliberately creates parameters for the
                # notebook-friendly ``profile.name`` syntax.  Never use
                # hasattr() here: probing an ordinary SIE for ``pixel_grid``
                # would register a phantom, unspecified model parameter.
                pixelated_names = {"pixel_grid", "pixelated_prior"}
                if pixelated_names.issubset(profile._parameters):
                    materialized = profile.parameters
                    entry["pixel_grid"] = materialized["pixel_grid"]
                    entry["pixelated_prior"] = materialized["pixelated_prior"]
                elif all(getattr(type(profile), name, None) is not None
                         for name in pixelated_names):
                    entry["pixel_grid"] = profile.pixel_grid
                    entry["pixelated_prior"] = profile.pixelated_prior
                if profile._initialization is not None:
                    entry["initialize_from"] = {
                        key: value for key, value in profile._initialization.items()
                        if key != "applied"
                    }
                declaration = profile._warm_start or profiles._warm_start
                if declaration is not None:
                    entry["warm_start_from"] = deepcopy(declaration)
                result[component].append(entry)
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

    def apply_initializations(self) -> bool:
        """Resolve profile ``initialize_from`` declarations into fixed values.

        This is intentionally called by a model immediately before it creates
        NumPyro's initialization state.  It returns whether a backend rebuild
        is needed because one or more formerly sampled parameters became
        fixed.  The saved result is indexed by the profile's declared order.
        """
        result_keys = {
            "lens_mass": "kwargs_lens",
            "lens_light": "kwargs_lens_light",
            "source_light": "kwargs_source",
            "point_source": "kwargs_point_source",
        }
        changed = False
        for actual_component in self._components:
            profiles = getattr(self, actual_component)
            if profiles is None:
                continue
            for index, profile in enumerate(profiles):
                declaration = profile._initialization
                if declaration is None or declaration["applied"]:
                    continue
                declared_component = declaration["component"]
                if declared_component != actual_component:
                    raise ValueError(
                        f"{profile.profile_type}[{index}] is in {actual_component!r}, but "
                        f"initialize_from() declared {declared_component!r}."
                    )
                path = Path(declaration["path"])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Initialization result file does not exist: {path}."
                    )
                try:
                    with path.open() as stream:
                        saved = json.load(stream)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Could not read JSON initialization result: {path}.") from error
                entries = saved.get(result_keys[actual_component])
                if not isinstance(entries, list):
                    raise ValueError(
                        f"{path} has no {result_keys[actual_component]!r} list for "
                        f"{actual_component!r}."
                    )
                if index >= len(entries) or not isinstance(entries[index], dict):
                    raise ValueError(
                        f"{path} has no usable {actual_component}[{index}] to initialize "
                        f"{profile.profile_type}."
                    )
                saved_values = entries[index]
                missing = [name for name in profile._parameters if name not in saved_values]
                if missing:
                    raise ValueError(
                        f"{path}: {actual_component}[{index}] is missing declared parameter(s) "
                        f"{missing} for {profile.profile_type}."
                    )
                for name in profile._parameters:
                    # A scalar prior is represented by the backend as fixed;
                    # retain value too for notebook inspection and staging.
                    value = deepcopy(saved_values[name])
                    parameter = profile.parameter(name)
                    parameter.prior = value
                    parameter.value = value
                declaration["applied"] = True
                changed = True
        return changed

    def warm_start_declarations(self) -> dict[str, dict[str, str]]:
        """Return validated non-fixed warm-start declarations by component.

        A declaration belongs to a whole physical component because a saved
        result stores each component as an ordered list.  A one-profile
        component may conveniently declare it on the profile itself; mixed
        collections should declare it on :class:`ProfileCollection`.
        """
        declarations: dict[str, dict[str, str]] = {}
        for actual_component in self._components:
            profiles = getattr(self, actual_component)
            if profiles is None:
                continue
            group_declaration = profiles._warm_start
            member_declarations = [profile._warm_start for profile in profiles if profile._warm_start is not None]
            if group_declaration is not None and member_declarations:
                raise ValueError(
                    f"{actual_component} declares warm_start_from() both on its ProfileCollection "
                    "and on an individual profile; declare it in one place only."
                )
            if group_declaration is not None:
                declaration = group_declaration
            elif not member_declarations:
                continue
            elif len(profiles) == 1:
                declaration = member_declarations[0]
            elif len(member_declarations) == len(profiles) and all(
                item == member_declarations[0] for item in member_declarations[1:]
            ):
                declaration = member_declarations[0]
            else:
                raise ValueError(
                    f"{actual_component} has multiple profile-level warm starts. "
                    "Declare one warm_start_from() on its ProfileCollection instead."
                )
            declared_component = declaration["component"]
            if declared_component != actual_component:
                raise ValueError(
                    f"{actual_component} declares warm_start_from(..., component={declared_component!r}); "
                    f"use component={actual_component!r}."
                )
            declarations[actual_component] = deepcopy(declaration)
        return declarations

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
        # Dynamic stellar mass refers to lens-light profile objects.  Clone
        # those first so a subsequent ``with_fixed(lens_light=True)`` keeps
        # the same dependency while replacing only the light values.
        clone_order = ("lens_light", "lens_mass", "source_light", "point_source")
        original_lens_light_indices = {
            id(profile): index
            for index, profile in enumerate(self.lens_light or [])
        }
        for component in clone_order:
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
                if isinstance(profile, StellarMassMGE) and profile.follows_lens_light:
                    try:
                        linked_light = ProfileCollection([
                            clones[("lens_light", original_lens_light_indices[id(light_profile)])]
                            for light_profile in profile.lens_light_profiles
                        ])
                    except KeyError as error:
                        raise ValueError(
                            "A dynamic StellarMassMGE must reference lens-light profiles "
                            "that are present in this LensProfileCollection."
                        ) from error
                    clone = StellarMassMGE(
                        linked_light,
                        prior=prior,
                        value=value,
                        follow_lens_light=True,
                    )
                else:
                    clone = profile_class(profile.profile_type, prior=prior, value=value)
                clone._initialization = deepcopy(profile._initialization)
                clone._warm_start = deepcopy(profile._warm_start)
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

        copied_components = {}
        for component in self._components:
            profiles = getattr(self, component)
            if profiles is None:
                continue
            collection = ProfileCollection([
                clones[(component, index)] for index in range(len(profiles))
            ])
            collection._warm_start = deepcopy(profiles._warm_start)
            copied_components[component] = collection
        return LensProfileCollection(**copied_components)

    def __repr__(self) -> str:
        return f"LensProfileCollection(\n{pformat(self.configuration, sort_dicts=False)}\n)"

    __str__ = __repr__
