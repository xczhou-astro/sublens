"""
SubLens: Physics-Informed Astrometric Inversion and Subhalo Discovery Toolkit.
"""

__version__ = "1.0.0"
__author__ = "Antigravity Gravitational Lensing Research Team"

# High-Level API
from .pipeline import SubLensPipeline, SubLensResult, detect_subhalo

# Physics & Profiles
from .cosmology import LensingCosmology
from .profiles.macro import (
    epl_deflection,
    external_shear_deflection,
    hernquist_deflection,
    MacroLensModel
)
from .profiles.subhalo import (
    nfw_subhalo_deflection,
    sis_subhalo_deflection,
    Subhalo,
    MultiSubhaloPopulation
)

# Simulation & Benchmarks
from .simulation.simulator import LensingSimulator, build_gaussian_psf
from .simulation.sources import sersic_profile, generate_complex_source
from .simulation.benchmarks import (
    create_j0946_benchmark,
    create_b1938_benchmark,
    create_scale_free_benchmark
)
from .simulation.hst import (
    generate_hst_acs_psf,
    generate_hst_noise_map,
    create_j0946_hst_benchmark
)

# Algorithmic Inversion Stages
from .inversion.macro_fitter import MacroFitter
from .inversion.radar import AstrometricRadar
from .inversion.pinn_optimizer import PINNOptimizer

# Benchmarks & Utilities
from .benchmarks.lenstronomy_bench import LenstronomyBenchmark
from .utils.metrics import compute_astrometric_error, compute_image_metrics
from .utils.visualization import plot_diagnostic_figure

__all__ = [
    "__version__",
    # High-level API
    "SubLensPipeline",
    "SubLensResult",
    "detect_subhalo",
    # Core Classes
    "LensingCosmology",
    "MacroLensModel",
    "Subhalo",
    "MultiSubhaloPopulation",
    "LensingSimulator",
    "build_gaussian_psf",
    "sersic_profile",
    "generate_complex_source",
    "create_j0946_benchmark",
    "create_b1938_benchmark",
    "create_scale_free_benchmark",
    "generate_hst_acs_psf",
    "generate_hst_noise_map",
    "create_j0946_hst_benchmark",
    "MacroFitter",
    "AstrometricRadar",
    "PINNOptimizer",
    "LenstronomyBenchmark",
    "compute_astrometric_error",
    "compute_image_metrics",
    "plot_diagnostic_figure",
    "epl_deflection",
    "external_shear_deflection",
    "hernquist_deflection",
    "nfw_subhalo_deflection",
    "sis_subhalo_deflection"
]
