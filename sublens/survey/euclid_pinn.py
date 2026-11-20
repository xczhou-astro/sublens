"""Compatibility imports for PINN potential reconstruction."""

from ..inversion.potential import (
    ResidualPotential, propose_residual_peaks, _SineLayer, _source_gradient,
    _psf_guard_radius, _arc_masks, _arc_folds, _smooth_convergence,
    _potential_peaks,
)
