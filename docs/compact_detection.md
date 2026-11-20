# Compact subhalo results

The default PINN summary reports whether the fit-improvement rule is met, the
estimated image-plane position, and the angular strength. It performs no new
optimization and uses the result's own matched macro-only baseline.

```python
from sublens import summarize_subhalo

decision = result.detection()
print(decision["detected"])
print(decision["position_arcsec"], decision["strength_arcsec"])
result.summary()

# Optional detailed diagnostics and an explicitly different threshold:
result.summary(detailed=True)
decision_at_50 = result.detection(min_delta_chi2=50.)
# A saved JSON dictionary works too, including old files without a detection key:
decision = summarize_subhalo(saved_result, min_delta_chi2=25.)
```

The rule is `delta_chi2 >= 25` **and** positive penalized improvement. Candidates
are filtered before choosing the greatest penalized gain among survivors. Each
is a competing single-halo fit; a list of passing alternatives is not a halo
count. The threshold is configurable but defaults to the previously used value.
The call does not change the stored result or its default threshold. JSON/FITS
exports include the default decision and its rule in their metadata. The FITS
primary header adds `HALODET`, `DCHIMIN`, `DETCAL=False` and, for a positive
decision, `DETRANK`. Legacy `BESTRANK`/`MODEL` still describe the greatest-objective
gain fit; `DETRANK` identifies the selected passing candidate in `CANDIDATES`.

A nondetection returns `None` for position and strength. Failed, missing-input,
malformed or nonfinite fits are not treated as valid negative results. Optimizer
convergence, parameter boundaries and stability variants remain in the detailed
records but are not extra vetoes under the user-selected rule.

This rule is a model-based screening decision, not a calibrated physical
detection probability. Macro/source/regularization degeneracy, PSF and foreground
errors and data reuse can produce improvements. Published sources and injection
truth do not enter the decision. Positions and strengths are point estimates in
arcseconds; `b` is the profile amplitude, not a mass or an uncertainty estimate.

## DR1 full refinement and compact table

The existing frozen candidate cohort can be processed with:

```bash
.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py run --device mps --workers 2
```

Omitting `run` does the same. The command runs or resumes the full seven-parameter
SIE + halo + source refinements and writes the compact catalogue. It does not run
the eight sensitivity variants. The input remains the frozen
`selected_candidates.json` in that experiment's directory; it does not silently
replace the cohort with newly screened systems. `base`, `stability`, `wide`,
`alternate` and `plot` remain explicit stages for the earlier experiment.

To summarize cached full refinements without any new optimization:

```bash
.venv/bin/python sublens_testing/euclid_dr1_candidate_followup.py summary
```

New files go under `candidate_followup/fit_improvement/`:

- `subhalo_results.csv`: one row per system with detection, x/y, b and delta chi2.
- `results.json`: decisions, rule and source-result provenance.
- `README_ZH.md`: a compact Chinese table and rule description.

This table always uses the nominal `full_sie` result. It does not select a result
or impose a veto using sensitivity variants. Old FITS, parameters, stability
results and reports are preserved, so their earlier conclusions remain labelled
with their earlier stability criterion.

The batch scanner additionally writes `subhalo_results.csv`. Where a full-SIE
follow-up exists it is preferred; other rows are explicitly labelled
`refinement_stage=batch_joint`. Missing/failed rows have an empty detection value.
The scanner still performs its existing initial refinements; this presentation
change does not automatically run full-SIE follow-up for every newly scanned
system. Actual candidate fitting and the supplied foreground/PSF assumptions are
unchanged.
