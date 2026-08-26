"""Wrapper-local lensing profiles and their Herculens registrations."""

from .multipole import MPPL
from .composite import GNFWMGE, StellarMGE
from .registry import register_mass_profiles

__all__ = ["GNFWMGE", "MPPL", "StellarMGE", "register_mass_profiles"]
