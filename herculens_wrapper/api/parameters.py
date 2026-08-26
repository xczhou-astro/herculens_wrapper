"""Parameter state and profile definitions."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np


class Parameter:
    """A parameter's sampling definition (``prior``) and current state (``value``)."""

    def __init__(self, profile: "Profile", name: str) -> None:
        self._profile, self.name = profile, name

    @property
    def prior(self) -> Any:
        return self._profile._specification(self.name)["prior"]

    @prior.setter
    def prior(self, value: Any) -> None:
        self._profile._specification(self.name)["prior"] = value

    @property
    def value(self) -> Any:
        link = self._profile._specification(self.name)["link"]
        return link.value if link is not None else self._profile._specification(self.name)["value"]

    @value.setter
    def value(self, value: Any) -> None:
        spec = self._profile._specification(self.name)
        spec["link"], spec["value"] = None, value

    def link_to(self, parameter: "Parameter") -> "Parameter":
        if not isinstance(parameter, Parameter):
            raise TypeError("A parameter link must target another Parameter.")
        self._profile._specification(self.name)["link"] = parameter
        return self

    def __repr__(self) -> str:
        """Compact notebook representation of the parameter's definition and state."""
        return repr({"prior": self.prior, "value": self.value})

    __str__ = __repr__


class Profile:
    """Profile with parameter-object access, e.g. ``epl.theta_E.prior = ...``."""

    _internal_names = frozenset({"profile_type", "_specifications", "_parameters"})

    def __init__(self, profile_type: str, *, prior: Mapping[str, Any] | None = None,
                 value: Mapping[str, Any] | None = None, **initial_values: Any) -> None:
        object.__setattr__(self, "profile_type", str(profile_type))
        object.__setattr__(self, "_specifications", {})
        object.__setattr__(self, "_parameters", {})
        for name, item in dict(prior or {}).items(): self.parameter(name).prior = item
        for name, item in {**dict(value or {}), **initial_values}.items(): self.parameter(name).value = item

    def __getattr__(self, name: str) -> Parameter:
        if name.startswith("_"): raise AttributeError(name)
        return self.parameter(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._internal_names or name.startswith("_"): object.__setattr__(self, name, value)
        elif isinstance(value, Parameter): self.parameter(name).link_to(value)
        else: self.parameter(name).value = value

    def _specification(self, name: str) -> dict[str, Any]:
        return self._specifications.setdefault(name, {"prior": None, "value": None, "link": None})

    def parameter(self, name: str) -> Parameter:
        self._specification(name)
        return self._parameters.setdefault(name, Parameter(self, name))

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameter_values(mode="inference")

    @property
    def values(self) -> dict[str, Any]:
        return {name: parameter.value for name, parameter in self._parameters.items()}

    @property
    def priors(self) -> dict[str, Any]:
        """Declared prior (or fixed scalar) for every registered parameter."""
        return {name: parameter.prior for name, parameter in self._parameters.items()}

    def set_prior(self, **priors: Any) -> "Profile":
        for name, item in priors.items(): self.parameter(name).prior = item
        return self

    def set_value(self, **values: Any) -> "Profile":
        for name, item in values.items(): self.parameter(name).value = item
        return self

    @classmethod
    def from_values(cls, profile_type: str, values: Mapping[str, Any]) -> "Profile":
        return cls(profile_type, value=values)

    def _parameter_values(self, *, mode: Literal["inference", "fixed"]) -> dict[str, Any]:
        values = {}
        for name, parameter in self._parameters.items():
            link = self._specification(name)["link"]
            if link is not None and mode == "inference": values[name] = link
            elif link is not None:
                candidate = link.value if link.value is not None else link.prior if not isinstance(link.prior, (list, tuple)) else None
                if candidate is None: raise ValueError(f"Cannot freeze {self.profile_type}.{name}; linked value is unavailable.")
                values[name] = candidate
            elif mode == "inference" and parameter.prior is not None: values[name] = parameter.prior
            elif mode == "fixed" and parameter.value is not None: values[name] = parameter.value
            elif mode == "fixed" and parameter.prior is not None and not isinstance(parameter.prior, (list, tuple)): values[name] = parameter.prior
            elif mode == "inference" and parameter.value is not None: values[name] = parameter.value
            else: raise ValueError(f"{self.profile_type}.{name} has no {mode} specification.")
        return values

    def freeze(self) -> "Profile":
        return type(self)(self.profile_type, prior=self._parameter_values(mode="fixed"))


class ProfileCollection(Sequence[Profile]):
    """An ordered, component-agnostic collection of profile objects."""

    def __init__(self, profiles: Sequence[Profile]) -> None:
        self.profiles = list(profiles)

    def __getitem__(self, index: int) -> Profile:
        return self.profiles[index]

    def __len__(self) -> int:
        return len(self.profiles)

    def __iter__(self):
        return iter(self.profiles)

    @property
    def values(self) -> list[dict[str, Any]]:
        """Current parameter values for every profile in order."""
        return [profile.values for profile in self.profiles]

    @property
    def priors(self) -> list[dict[str, Any]]:
        """Prior definitions for every profile in order."""
        return [profile.priors for profile in self.profiles]


class MassProfile(Profile):
    """Lens-mass profile, or a component-agnostic multi-profile collection."""

    def __new__(
        cls,
        profile_type: str | Sequence[str],
        *,
        prior: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        **initial_values: Any,
    ):
        if isinstance(profile_type, (list, tuple)):
            names = list(profile_type)
            if not names:
                raise ValueError("MassProfile requires at least one profile name.")
            if initial_values:
                raise TypeError("Use prior=[...] or value=[...] when registering multiple mass profiles.")
            priors = [{} for _ in names] if prior is None else list(prior)
            values = [{} for _ in names] if value is None else list(value)
            if len(priors) != len(names) or len(values) != len(names):
                raise ValueError("The prior and value lists must match the number of mass profiles.")
            if any(not isinstance(item, Mapping) for item in priors + values):
                raise TypeError("Each multi-profile prior/value entry must be a dictionary.")
            return ProfileCollection([
                cls(name, prior=item_prior, value=item_value)
                for name, item_prior, item_value in zip(names, priors, values)
            ])
        return super().__new__(cls)

    def __init__(self, profile_type: str, **kwargs: Any) -> None:
        super().__init__(str(profile_type).upper(), **kwargs)


class StellarMassMGE(MassProfile):
    """A fixed lens-light Gaussian MGE scaled into stellar convergence.

    The supplied lens-light profiles must be fixed ``GAUSSIAN_ELLIPSE``
    components.  Their geometry and relative light amplitudes are retained as
    fixed arrays; only ``upsilon_kappa`` and optionally ``ml_gradient`` are
    sampled.  ``upsilon_kappa`` is a dimensionless lensing normalization, not
    a physical mass-to-light ratio.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)

    def __init__(
        self,
        lens_light: "ProfileCollection | Sequence[LightProfile]",
        *,
        prior: Mapping[str, Any] | None = None,
        value: Mapping[str, Any] | None = None,
    ) -> None:
        profiles = list(lens_light)
        if not profiles:
            raise ValueError("StellarMassMGE requires at least one lens-light Gaussian.")
        required = ("amp", "sigma", "e1", "e2", "center_x", "center_y")
        components: dict[str, list[float]] = {name: [] for name in required}
        for index, profile in enumerate(profiles):
            if not isinstance(profile, LightProfile) or profile.profile_type != "GAUSSIAN_ELLIPSE":
                raise TypeError(
                    "StellarMassMGE requires fixed LightProfile('GAUSSIAN_ELLIPSE') components; "
                    f"component {index} is {getattr(profile, 'profile_type', type(profile).__name__)!r}."
                )
            fixed = profile._parameter_values(mode="fixed")
            missing = [name for name in required if name not in fixed]
            if missing:
                raise ValueError(f"Lens-light Gaussian {index} is missing fixed values for {missing}.")
            for name in required:
                components[name].append(float(fixed[name]))
        if any(amplitude <= 0 for amplitude in components["amp"]):
            raise ValueError("StellarMassMGE requires positive fixed lens-light Gaussian amplitudes.")
        if any(sigma <= 0 for sigma in components["sigma"]):
            raise ValueError("StellarMassMGE requires positive fixed lens-light Gaussian sigmas.")

        settings = dict(prior or {})
        values = dict(value or {})
        if "upsilon_kappa" not in settings and "upsilon_kappa" not in values:
            raise ValueError("Specify prior or value for StellarMassMGE.upsilon_kappa.")
        settings.setdefault("ml_gradient", 0.0)
        settings.update({
            "light_amp": np.asarray(components["amp"], dtype=float),
            "light_sigma": np.asarray(components["sigma"], dtype=float),
            "light_e1": np.asarray(components["e1"], dtype=float),
            "light_e2": np.asarray(components["e2"], dtype=float),
            "light_center_x": np.asarray(components["center_x"], dtype=float),
            "light_center_y": np.asarray(components["center_y"], dtype=float),
        })
        super().__init__("STELLAR_MGE", prior=settings, value=values)

    def freeze(self) -> "StellarMassMGE":
        raise TypeError("StellarMassMGE is already based on fixed lens-light components.")


class GNFWHaloMGE(MassProfile):
    """Differentiable elliptical gNFW halo evaluated through a 3-D MGE.

    ``r_s`` is in angular units and ``kappa_s`` is dimensionless.  No redshift
    or cosmology is required until converting the fitted result to physical
    quantities.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)

    def __init__(
        self,
        *,
        prior: Mapping[str, Any] | None = None,
        value: Mapping[str, Any] | None = None,
    ) -> None:
        settings = dict(prior or {})
        values = dict(value or {})
        if "n_outer" not in settings and "n_outer" not in values:
            settings["n_outer"] = 3.0
        super().__init__("GNFW_MGE", prior=settings, value=values)

class LightProfile(Profile):
    """Light profile, or a component-agnostic multi-profile collection."""

    def __new__(
        cls,
        profile_type: str | Sequence[str],
        *,
        prior: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        **initial_values: Any,
    ):
        if isinstance(profile_type, (list, tuple)):
            names = list(profile_type)
            if not names:
                raise ValueError("LightProfile requires at least one profile name.")
            if initial_values:
                raise TypeError("Use prior=[...] or value=[...] when registering multiple light profiles.")
            priors = [{} for _ in names] if prior is None else list(prior)
            values = [{} for _ in names] if value is None else list(value)
            if len(priors) != len(names) or len(values) != len(names):
                raise ValueError("The prior and value lists must match the number of light profiles.")
            if any(not isinstance(item, Mapping) for item in priors + values):
                raise TypeError("Each multi-profile prior/value entry must be a dictionary.")
            return ProfileCollection([
                cls(name, prior=item_prior, value=item_value)
                for name, item_prior, item_value in zip(names, priors, values)
            ])
        return super().__new__(cls)

    def __init__(self, profile_type: str, **kwargs: Any) -> None:
        super().__init__(str(profile_type).upper(), **kwargs)


class PixelatedSource(LightProfile):
    """A pixelated source-light declaration for SVI or HMC reconstruction.

    ``pixel_grid`` controls the source grid, while ``pixelated_prior`` selects
    the source prior (``'matern'``, ``'wavelet_sparsity'``, or
    ``'wavelet_penalty'``).  These are model settings, not scalar sampling
    parameters, so they are kept together in one source profile.
    """

    _grid_defaults = {
        "pixel_adaptive_grid": True,
        "pixel_grid_shape": 80,
        "pixel_interpol": "fast_bilinear",
        "pixel_scale_factor": 0.5,
        "grid_center": (0.0, 0.0),
        "grid_shape": (2.0, 2.0),
    }
    _prior_defaults = {
        "prior_type": "matern",
        "regul_strengths": (3.0, 3.0),
        "k_zero": 0.0,
        "n_value_low": 1e-4,
        "n_value_high": 100.0,
        "sigma_low": 1e-5,
        "sigma_high": 10.0,
        "rho_low": None,
        "rho_high": None,
        "positive": True,
    }

    def __new__(cls, **kwargs: Any):
        return object.__new__(cls)

    def __init__(
        self,
        *,
        pixel_grid: Mapping[str, Any] | None = None,
        pixelated_prior: Mapping[str, Any] | None = None,
    ) -> None:
        grid = {**self._grid_defaults, **dict(pixel_grid or {})}
        prior = {**self._prior_defaults, **dict(pixelated_prior or {})}
        if int(grid["pixel_grid_shape"]) <= 0:
            raise ValueError("pixel_grid['pixel_grid_shape'] must be positive.")
        if prior["prior_type"] not in {"matern", "wavelet_sparsity", "wavelet_penalty"}:
            raise ValueError("pixelated_prior['prior_type'] must be 'matern', 'wavelet_sparsity', or 'wavelet_penalty'.")
        super().__init__("PIXELATED", prior={"pixel_grid": grid, "pixelated_prior": prior})

class PointSourceProfile(Profile): pass
