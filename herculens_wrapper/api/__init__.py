"""Config-free, notebook-oriented public API."""
from .collections import LensProfileCollection
from .config_export import detect_gpus, export_wrapper_config
from .data import SingleBandData
from .multiband import MultiBandData, MultiBandFitResult, MultiBandModel, MultiBandProfileCollection, MultiBandResultsCombination
from .models import ModelDefinition
from .parameters import GNFWHaloMGE, LightProfile, MassProfile, Parameter, PixelatedLensLight, PixelatedSource, PointSourceProfile, Profile, ProfileCollection, StellarMassMGE
from .samplers import (
    FitResult, SingleBandResultsCombination, SamplerConfig, compare_hmc_truth,
    is_completed_svi_run,
)
from .session import SingleBandModel
from .physics import LensGeometry
from .utils import fit_analytic_pixelated_source
from .visualization import plot_single_band_data
__all__ = ["FitResult", "GNFWHaloMGE", "LensGeometry", "LensProfileCollection", "LightProfile", "MassProfile", "ModelDefinition", "MultiBandData", "MultiBandFitResult", "MultiBandModel", "MultiBandProfileCollection", "MultiBandResultsCombination", "Parameter", "PixelatedLensLight", "PixelatedSource", "PointSourceProfile", "Profile", "ProfileCollection", "SingleBandResultsCombination", "SamplerConfig", "SingleBandData", "SingleBandModel", "StellarMassMGE", "compare_hmc_truth", "detect_gpus", "export_wrapper_config", "fit_analytic_pixelated_source", "is_completed_svi_run", "plot_single_band_data"]
