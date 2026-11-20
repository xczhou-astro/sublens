# PyTorch execution backend

The generic `LensingObservation` Delaunay/PINN workflow supports an optional Torch backend. The existing reference backend remains the default. CPU, NVIDIA CUDA and Apple MPS use the same image likelihood, normalized supplied PSF, source regularization and nonnegative source constraint.

## Basic use

```python
from sublens import LensingObservation, torch_backend

obs = LensingObservation.from_fits("observation.fits")

with torch_backend(device="mps"):
    source = obs.reconstruct(
        source_nodes=500, regularization=0.002, mesh_strategy="centroidal"
    )
    result = obs.search_pinn(
        source_nodes=500, regularization=0.002,
        top_k=5, pinn_steps=90, refine_steps=70,
        mesh_strategy="centroidal", cross_fit=True,
    )
    result = result.refine_joint(
        obs.macro_model, maxiter=80, n_starts=2, retry_failed=True
    )
    result.save_fits("result_mps.fits", overwrite=True)
```

The coefficient above is an example, not an observationally calibrated universal value. Keep the existing data-based source selection and candidate validation rules.

The equivalent single-call entry point is:

```python
result = obs.search_pinn(backend="torch", device="mps", source_nodes=500)
# The result remembers its execution settings, including when leaving the context.
result = result.refine_joint(obs.macro_model)
```

To select a different device for verification:

```python
with torch_backend("cpu", dtype="float64"):
    checked = result.refine_joint(obs.macro_model, n_starts=2, retry_failed=True)
```

## Devices and precision

| Device | Selection | Default precision |
|---|---|---|
| CPU | `cpu` | float64 |
| NVIDIA GPU | `gpu`, `cuda`, `cuda:0`, etc. | float64 |
| Apple GPU | `mps` | float32 |
| Available accelerator, otherwise CPU | `auto` | device dependent |

`gpu` means CUDA, not MPS. An unavailable explicitly requested device raises an error. MPS float64 is rejected; no hidden CPU fallback is enabled. Consumer CUDA cards can have limited float64 throughput; float32 can be requested explicitly after validating its accuracy for the dataset. Full CUDA numerical tests are conditional on having CUDA hardware; this Mac validates CPU and MPS, not NVIDIA execution.

Advanced settings:

```python
with torch_backend(
    "cpu", dtype="float64", source_tolerance=1e-7,
    source_max_steps=4000, basis_batch_size=64,
):
    result = obs.search_pinn(source_nodes=500)
```

Larger basis batches can increase memory use. Source fits check projected KKT residuals; nonconverged inner solves are rejected before their envelope gradients are used. Outer fits report gradient or relative-objective convergence separately in optimizer metadata; this is optimization convergence, not posterior convergence or a detection claim.

## Which operations use Torch?

* Dynamic ray-to-triangle location: a precomputed seed grid, neighbor walking and an exact chunked fallback, all using Torch tensors.
* Barycentric interpolation and full supplied PSF convolution, including asymmetric or negative PSF lobes.
* Positive source inversion: a diagonally scaled quadratic solve using block principal pivoting, with a projected-gradient fallback. Warm starts and operator/Gram caches are reused on a fixed geometry and across regularization trials.
* Macro and halo refinement: bounded Torch L-BFGS, with a reprofiled source and envelope gradients. The null and halo hypotheses retain the same source basis, prior and macro bounds.
* PINN training and derivative calculations: Torch on the selected device; the existing cross-fitting, PSF guards and source-coverage rules remain.

FITS I/O, initial adaptive mesh/topology construction, initial lookup generation, cross-fit mask preparation and final candidate summaries/plots remain CPU work. The triangle topology is fixed during each fit; its discrete changes are not differentiated. This is a Torch fitting backend, not a claim that every preprocessing operation is on a GPU. Legacy radar-specific fitting interfaces are not migrated by this option.

The seven-parameter source-profiled macro API is:

```python
fit = obs.fit_macro_torch(
    device="cpu", source_nodes=500, regularization=0.002, maxiter=80
)
macro = fit["macro_model"]
```

The observation supplies the initial macro, which should come from the data or an explicitly labelled initialization. No truth or published source is read by this API. The slope is held fixed; normalization, axis ratio, angle, centre and shear are free. This is a point estimate.

## Grade A DR1 batch interface

```bash
.venv/bin/python sublens_testing/euclid_dr1_grade_a_scan.py prepare --backend torch --device mps
.venv/bin/python sublens_testing/euclid_dr1_grade_a_scan.py run --backend torch --device mps
```

Use `--device cpu`, `--device gpu` or `--device cuda:1` as appropriate. `--dtype float32` or `--dtype float64` must be consistent between preparation and execution. Accelerator runs default to one worker; CPU runs default to three. Torch workers use multiprocessing `spawn`. Outputs default to a separate backend/device directory, preserving reference results; `--output` selects another directory. The batch still selects only the 357 local Grade A systems and records missing inputs, failures and other official inspection statuses.

For a single-system validation before the full scan:

```bash
.venv/bin/python sublens_testing/euclid_dr1_grade_a_scan.py run --backend torch --device mps --ids EUCL-DR1-L020080
```

## Measured validation

See `sublens_testing/euclid_dr1_grade_a_part1/torch_backend_validation/` for identical-data reference/CPU/MPS comparisons. Report actual timing and numerical differences; a Torch CPU rewrite is not assumed to be faster than SciPy. Float32 candidate fits should be checked in CPU/CUDA float64 when precision or decision thresholds matter. These checks do not calibrate subhalo significance or make the published source a ground truth.

The precision setting applies to source inversion and physical fits. The PINN network retains float32 training across devices, matching the reference implementation. `observation.search_pinn(device="mps")` also explicitly selects Torch; `backend="reference"` selects the original CPU implementation for a controlled comparison.

Simultaneous component hypotheses are supported too:

```python
from sublens import MultiHaloProblem
with torch_backend("mps"):
    problem = MultiHaloProblem(obs, source_nodes=500, regularization=0.002)
    fit = problem.fit(initial_parameters, candidate_anchors)
```

Here `initial_parameters` contains three macro offsets followed by `(x, y, b / max_strength)` for each component. These are proposed components, not injected truth. The fit retains the exact supplied configuration, including zero-strength endpoints, as a feasible solution. An explicit component count does not determine whether that many halos exist.

A matched-input DR1 benchmark (500 source nodes, 4×4 pixel integration, five proposals, same search/refinement budgets) measured:

| Implementation | PINN + matched joint fit |
|---|---:|
| Reference CPU | 74.5 s |
| Torch CPU, float64 | 113.2 s |
| Torch MPS, float32 | 41.1 s |

These are single-system measurements, not a survey throughput guarantee. The five proposal positions agreed; the strongest MPS fit differed by about 6.5 mas in position, 1% in strength and 0.017 in delta-chi2 from the reference. Weak alternative minima can differ between optimizers. In a separate full-pipeline run with image-only initialization and adaptive source selection, two flagged MPS candidates were rechecked in CPU float64: their positions differed by about 0.29 and 1.82 mas, and delta-chi2 by about 0.003. They remain unconfirmed screening candidates; macro/source modeling differences matter more than these rounding differences.

A matched-budget one-halo simulation test recovered the injected position to about 2.94 mas and strength to about 4.36% for reference, CPU and MPS. The supplied simulation macro makes this a conditional recovery check. The simulation was smaller and did not show an MPS speedup, illustrating why device selection must be benchmarked for each workload.
