"""Wrapper-local lensing profiles and their Herculens registrations."""

from .multipole import MPPL
from .registry import register_mass_profiles

__all__ = ["MPPL", "register_mass_profiles"]
