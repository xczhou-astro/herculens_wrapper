"""Register wrapper-local profile names with the installed Herculens package."""

from .multipole import MPPL


def register_mass_profiles():
    """Make the wrapper-local ``MPPL`` type available to ``MassModel``.

    This updates the in-memory registry only for the current Python process.
    It does not alter the installed Herculens package and leaves ``MULTIPOLE``
    mapped to Herculens's original implementation.
    """
    from herculens.MassModel import mass_model_base, profile_mapping

    existing = profile_mapping.STRING_MAPPING.get("MPPL")
    if existing is not None and existing is not MPPL:
        raise RuntimeError("Herculens already registered a different profile as 'MPPL'.")

    profile_mapping.STRING_MAPPING["MPPL"] = MPPL
    if "MPPL" not in profile_mapping.SUPPORTED_MODELS:
        profile_mapping.SUPPORTED_MODELS.append("MPPL")

    # Older Herculens releases cache these objects in mass_model_base.
    mass_model_base.STRING_MAPPING["MPPL"] = MPPL
    if "MPPL" not in mass_model_base.SUPPORTED_MODELS:
        mass_model_base.SUPPORTED_MODELS.append("MPPL")
