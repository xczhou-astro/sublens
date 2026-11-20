"""Compact, fit-improvement-only decisions for fitted PINN halo alternatives.

These are model-based screening decisions, not calibrated physical detections.
No source truth, published source, stability diagnostic or optimizer flag is
used to choose the outcome. Invalid / failed inputs are errors, never negatives.
"""
from collections.abc import Mapping
import math


def _value(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"Candidate is missing {names[0]}")


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _detection_from_candidates(candidates, *, lens_id=None, min_delta_chi2=25.):
    threshold = _finite(min_delta_chi2, "min_delta_chi2")
    if threshold <= 0:
        raise ValueError("min_delta_chi2 must be positive")
    if candidates is None or isinstance(candidates, (str, bytes, Mapping)):
        raise ValueError("candidates must be a sequence of fitted candidate records")
    selected = []
    for index, row in enumerate(candidates, 1):
        if not isinstance(row, Mapping):
            raise ValueError("Each candidate must be a fitted candidate record")
        position = _value(row, "position_arcsec", "position_xy_arcsec", "position")
        if isinstance(position, (str, bytes, Mapping)):
            raise ValueError("position_arcsec must contain x and y")
        try:
            if len(position) != 2:
                raise ValueError("position_arcsec must contain x and y")
            xy = [_finite(v, "position_arcsec") for v in position]
        except TypeError as error:
            raise ValueError("position_arcsec must contain x and y") from error
        strength = _finite(_value(row, "strength_arcsec", "strength"), "strength_arcsec")
        if strength < 0:
            raise ValueError("strength_arcsec must be nonnegative")
        chi = _finite(_value(row, "delta_chi2"), "delta_chi2")
        gain = _finite(_value(row, "delta_objective", "penalized_gain"), "delta_objective")
        if chi >= threshold and gain > 0:
            if strength == 0:
                raise ValueError("Positive halo improvement with zero strength; check the matched null fit")
            selected.append(dict(candidate_rank=row.get("rank", index), position_arcsec=xy,
                                 strength_arcsec=strength, delta_chi2=chi, delta_objective=gain))
    # Filter first: the highest penalized gain can belong to a candidate that
    # fails the chi2 rule. All survivors remain competing single-halo fits.
    best = max(selected, key=lambda row: (row["delta_objective"], row["delta_chi2"]), default=None)
    return dict(
        lens_id=lens_id, detected=best is not None,
        candidate_rank=None if best is None else best["candidate_rank"],
        position_arcsec=None if best is None else best["position_arcsec"],
        strength_arcsec=None if best is None else best["strength_arcsec"],
        delta_chi2=None if best is None else best["delta_chi2"],
        delta_objective=None if best is None else best["delta_objective"],
        rule=dict(name="fit_improvement", min_delta_chi2=threshold,
                  positive_penalized_gain=True, stability_required=False,
                  significance_calibrated=False),
    )


def summarize_subhalo(result, *, min_delta_chi2=None):
    """Return detected, position and strength from already fitted alternatives.

    Accepts a PINN result or its JSON dictionary, including the saved DR1 batch
    and full-SIE follow-up metric schemas. Does not run or modify a fit. A
    positive decision requires delta_chi2 >= min_delta_chi2 and a positive
    penalized gain, and selects the greatest penalized gain among survivors.
    Stability, convergence and bound flags remain diagnostics, not vetoes.
    ``detected=False`` means no fitted alternative met this rule; failed or
    incomplete records raise ValueError instead of being classified as absent.
    """
    data = result if isinstance(result, Mapping) else result.to_dict()
    if min_delta_chi2 is None:
        min_delta_chi2=data.get('config',{}).get('detection_min_delta_chi2',25.)
    if data.get("status", "completed") != "completed":
        raise ValueError("Cannot classify a failed or incomplete fit as a nondetection")
    if "candidates" not in data:
        raise ValueError("A completed fit must include its candidate records")
    return _detection_from_candidates(data["candidates"], lens_id=data.get("lens_id", data.get("identifier")),
                                      min_delta_chi2=min_delta_chi2)
