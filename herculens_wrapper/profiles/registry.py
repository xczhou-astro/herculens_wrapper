"""Register wrapper-local profile names with the installed Herculens package."""

from .multipole import EPLM1M3M4, EPLM3M4, MPPL, MPPLOffset
from .composite import GNFWMGE, InclinedExponentialDiskMGE, StellarMGE


def register_mass_profiles():
    """Make wrapper-local mass profile types available to ``MassModel``.

    This updates the in-memory registry only for the current Python process.
    It does not alter the installed Herculens package and leaves ``MULTIPOLE``
    mapped to Herculens's original implementation.
    """
    from herculens.MassModel import mass_model_base, profile_mapping

    for name, profile in {
        "MPPL": MPPL,
        "MPPL_OFFSET": MPPLOffset,
        "EPL_MULTIPOLE_M1M3M4_ELL": EPLM1M3M4,
        "EPL_MULTIPOLE_M3M4_ELL": EPLM3M4,
    }.items():
        existing = profile_mapping.STRING_MAPPING.get(name)
        if existing is not None and existing is not profile:
            raise RuntimeError(f"Herculens already registered a different profile as {name!r}.")
        profile_mapping.STRING_MAPPING[name] = profile
        # Older Herculens releases cache these objects in mass_model_base.
        mass_model_base.STRING_MAPPING[name] = profile
        if name not in profile_mapping.SUPPORTED_MODELS:
            profile_mapping.SUPPORTED_MODELS.append(name)
        if name not in mass_model_base.SUPPORTED_MODELS:
            mass_model_base.SUPPORTED_MODELS.append(name)

    for name, profile in {
        "STELLAR_MGE": StellarMGE,
        "GNFW_MGE": GNFWMGE,
        "INCLINED_EXPONENTIAL_DISK": InclinedExponentialDiskMGE,
    }.items():
        existing = profile_mapping.STRING_MAPPING.get(name)
        if existing is not None and existing is not profile:
            raise RuntimeError(f"Herculens already registered a different profile as {name!r}.")
        profile_mapping.STRING_MAPPING[name] = profile
        mass_model_base.STRING_MAPPING[name] = profile
        if name not in profile_mapping.SUPPORTED_MODELS:
            profile_mapping.SUPPORTED_MODELS.append(name)
        if name not in mass_model_base.SUPPORTED_MODELS:
            mass_model_base.SUPPORTED_MODELS.append(name)
