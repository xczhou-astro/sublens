"""Controlled source-capacity/mesh/regularization and matched-SNR diagnosis.

The previous PINN nominations initialize the identical-data ablations. Changed
noise datasets get fresh blind nominations. Truth and independent noise images
are used only after fitting, never to choose candidates or hyperparameters.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from astropy.io import fits

from sublens import LensingObservation
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.adaptive_refinement import _model_rays
from run import ROOT, TRUTH, SEARCH, ADAPT, load_result, save_result, residual_metrics
from strength_scan import write_json, digest
from complex_source_scan import save_source_comparison


OUTPUT = ROOT / "spiral_diagnosis"
ORIGINAL = ROOT / "complex_source"
METHODS = [
    dict(id="initial_joint_reference", dataset="spiral", mode="reference_initial"),
    dict(id="original_adaptive_reference", dataset="spiral", mode="reference_final"),
    dict(id="fixed_initial_extra_optimization", dataset="spiral", mode="repeat_initial"),
    dict(id="fixed_500_lambda2", dataset="spiral", mode="fixed", nodes=500, regularization=2.),
    dict(id="expanded_cv", dataset="spiral", mode="expanded"),
    dict(id="updated_500_lambda2", dataset="spiral", mode="updated", nodes=500, regularization=2.),
    dict(id="fixed_1000_lambda2", dataset="spiral", mode="fixed", nodes=1000, regularization=2.),
    dict(id="updated_1000_lambda2", dataset="spiral", mode="updated", nodes=1000, regularization=2.),
    dict(id="fixed_2000_lambda2", dataset="spiral", mode="fixed", nodes=2000, regularization=2.),
    dict(id="updated_2000_lambda2", dataset="spiral", mode="updated", nodes=2000, regularization=2.),
    dict(id="fixed_1000_lambda0p2", dataset="spiral", mode="fixed", nodes=1000, regularization=.2),
    dict(id="fixed_1000_lambda20", dataset="spiral", mode="fixed", nodes=1000, regularization=20.),
    dict(id="sersic_original_reference", dataset="sersic", mode="reference_final"),
    dict(id="sersic_low_snr_default", dataset="sersic_low_snr", mode="default"),
    dict(id="spiral_high_snr_default", dataset="spiral_high_snr", mode="default"),
    dict(id="spiral_null_reference", dataset="spiral_null", mode="reference_final"),
    dict(id="spiral_null_expanded_cv", dataset="spiral_null", mode="expanded"),
]


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    injected = {c["morphology"]: c for c in manifest["cases"] if not c["is_null"]}
    rho_spiral, rho_sersic = (injected[k]["rho_image"] for k in ("spiral_ism", "sersic"))
    specs = dict(spiral=("spiral_ism", .012, .05), sersic=("sersic", .012, .05),
                 sersic_low_snr=("sersic", .012*rho_sersic/rho_spiral, .05),
                 spiral_high_snr=("spiral_ism", .012*rho_spiral/rho_sersic, .05),
                 spiral_null=("spiral_ism", .012, 0.))
    config = dict(methods=METHODS, mesh_seed=33, fixed_vs_updated_rounds=2,
                  expanded_nodes=[500, 1000, 2000], expanded_regularizations=[.2, 2., 20.],
                  expanded_mesh_seeds=[11, 22, 33, 44], training_noise_seed=314159,
                  independent_noise_seed=271828, independent_image_not_used_for_selection=True,
                  macro_shape_slope_center_fixed_at_simulation_values=True,
                  validation_is_conditional_on_fitted_geometry=True, truth_used_only_for_evaluation=True)
    path = OUTPUT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing diagnosis configuration differs; preserve previous experiment")
    write_json(path, config)
    datasets = {}
    for name, (kind, rms, strength) in specs.items():
        folder = OUTPUT / "datasets" / name; folder.mkdir(parents=True, exist_ok=True)
        original_folder = ORIGINAL / "trials" / f"{kind}_strength_{strength:.6f}_seed_314159"
        base = LensingObservation.from_fits(original_folder / "observation.fits")
        clean = fits.getdata(ORIGINAL / "reference" / f"{kind}_strength_{strength:.6f}_clean.fits")
        clean_null = fits.getdata(ORIGINAL / "reference" / f"{kind}_strength_0.000000_clean.fits")
        for filename, seed in (("observation.fits", 314159), ("independent_observation.fits", 271828)):
            if not (folder / filename).exists():
                z = np.random.default_rng(seed).standard_normal(clean.shape)
                obs = replace(base, image=clean+rms*z, rms=np.full(clean.shape, rms), lens_id=name)
                obs.to_fits(folder / filename)
        if not (folder / "clean.fits").exists():
            fits.PrimaryHDU(clean, header=fits.getheader(ORIGINAL / "reference" / f"{kind}_strength_{strength:.6f}_clean.fits")).writeto(
                folder / "clean.fits", checksum=True)
        datasets[name] = dict(morphology=kind, rms=float(rms), injected_strength_arcsec=strength,
                              rho_image=float(np.linalg.norm((clean-clean_null)[base.mask])/rms),
                              original_folder=str(original_folder), original_rms=.012,
                              reused_clean_source=True)
    write_json(OUTPUT / "datasets.json", datasets)
    return datasets


def evaluate(result, dataset):
    """Post-fit diagnostics; independent noise and truth never feed the optimizer."""
    obs = result.observation
    folder = OUTPUT / "datasets" / dataset
    info = json.loads((OUTPUT / "datasets.json").read_text())[dataset]
    clean = fits.getdata(folder / "clean.fits")
    independent = fits.getdata(folder / "independent_observation.fits", "SCI")
    best = result.best_candidate
    model = result.baseline.image_model if best is None else best.fit.image_model
    mask = obs.mask
    metrics = dict(macro=residual_metrics(obs, result.baseline.image_model),
        with_subhalo=residual_metrics(obs, model),
        independent_prediction_chi2_per_pixel=float(np.mean(((independent-model)[mask]/obs.rms[mask])**2)),
        clean_model_error_per_pixel=float(np.mean(((clean-model)[mask]/obs.rms[mask])**2)),
        macro_converged=result.macro_converged, halo_converged=None, strength_arcsec=None,
        position_arcsec=None, position_error_mas=None, strength_error_fraction=None,
        recovered=None if info["injected_strength_arcsec"] == 0 else False,
        source_nodes=len(result.baseline.grid.points), regularization=result.baseline.regularization,
        delta_objective=None, best_candidate_rank=None)
    if best is not None:
        fit = best.fit
        metrics.update(halo_converged=fit.converged, strength_arcsec=fit.strength,
                       position_arcsec=list(fit.position), delta_objective=fit.delta_objective,
                       best_candidate_rank=best.rank)
        if info["injected_strength_arcsec"] > 0:
            h = TRUTH["halo"]
            error = float(1000*np.hypot(fit.x-h["x"], fit.y-h["y"]))
            amplitude = float(fit.strength/info["injected_strength_arcsec"]-1)
            metrics.update(position_error_mas=error, strength_error_fraction=amplitude,
                           recovered=bool(result.macro_converged and fit.converged
                                          and error <= 50 and abs(amplitude) <= .30))
    return metrics


def rebase(initial, nodes, regularization):
    """Change the basis at the same inferred geometry, without using source truth."""
    obs = initial.observation
    fit = None if initial.best_candidate is None else initial.best_candidate.fit
    geometry = (fit.macro_model if fit is not None else None) or initial.macro_model or obs.macro_model
    bx, by = _model_rays(obs, geometry, fit)
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, bx, by,
                                source_nodes=nodes, regularization=regularization,
                                seed=33, mesh_strategy="centroidal")
    baseline = problem.fit(obs.bx, obs.by)
    candidates = []
    for candidate in initial.candidates:
        previous = candidate.fit
        hx, hy = _model_rays(obs, previous.macro_model or obs.macro_model, previous)
        source = problem.fit(hx, hy)
        candidates.append(replace(candidate, fit=replace(previous, source=source, image_model=source.image_model)))
    return replace(initial, baseline=baseline, candidates=tuple(candidates))


def run_method(method):
    import torch
    torch.set_num_threads(1); matplotlib.use("Agg")
    folder = OUTPUT / "methods" / method["id"]; folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists(): return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    dataset, mode = method["dataset"], method["mode"]
    info = json.loads((OUTPUT / "datasets.json").read_text())[dataset]
    obs = LensingObservation.from_fits(OUTPUT / "datasets" / dataset / "observation.fits")
    original = info["rms"] == .012
    original_folder = Path(info["original_folder"])
    # Completed fits survive a plotting/metadata failure without re-optimization.
    if (folder / "result.fits").exists() and (folder / "source_comparison.fits").exists():
        result = load_result(folder / "result.fits")
        np.testing.assert_array_equal(result.observation.image, obs.image)
        np.testing.assert_array_equal(result.observation.rms, obs.rms)
        origin = (original_folder / ("result.fits" if mode == "reference_final" else "initial_joint.fits")) if original else folder / "initial_joint.fits"
        start = replace(load_result(origin), observation=obs)
        trace = [dict(stage="input_reference" if original else "initial_joint", **evaluate(start, dataset))]
        for iteration in range(2):
            checkpoint = folder / f"round_{iteration+1}.fits"
            if checkpoint.exists():
                stage = load_result(checkpoint)
                trace.append(dict(stage=("fixed" if mode in ("fixed", "repeat_initial") else "updated")+f"_round_{iteration+1}",
                                  **evaluate(stage, dataset)))
        metrics = {**method, **evaluate(result, dataset), "dataset_info": info, "seconds": None,
                   "reused_reference": mode.startswith("reference"), "trace": trace,
                   "input_sha256": digest(OUTPUT / "datasets" / dataset / "observation.fits"),
                   "test_noise_used_only_for_evaluation": True, "recovered_saved_fit": True,
                   "original_fit_runtime_unavailable": True}
        write_json(folder / "metrics.json", metrics)
        print(f"Recovered metadata for {method['id']} from completed FITS", flush=True)
        return metrics
    if mode == "reference_final":
        result = replace(load_result(original_folder / "result.fits"), observation=obs)
    elif original:
        result = replace(load_result(original_folder / "initial_joint.fits"), observation=obs)
    elif (folder / "initial_joint.fits").exists():
        result = load_result(folder / "initial_joint.fits")
    else:
        result = obs.search_pinn(**SEARCH, gain_radius=.22, psf_guard_max_radius=.135)
        result.save_json(folder / "initial_search.json", overwrite=True)
        result = result.refine_joint(obs.macro_model, maxiter=120, n_starts=2, retry_failed=True)
        result.save_fits(folder / "initial_joint.fits", overwrite=True)
        result.save_json(folder / "initial_joint.json", overwrite=True)
    trace = [dict(stage="input_reference" if original else "initial_joint", **evaluate(result, dataset))]
    if mode in ("fixed", "repeat_initial"):
        if mode == "fixed": result = rebase(result, method["nodes"], method["regularization"])
        # Equal joint-optimization budget to the two mesh-updated rounds.
        for iteration in range(2):
            result = result.refine_joint(obs.macro_model, maxiter=120, n_starts=3, retry_failed=True)
            result.save_fits(folder / f"round_{iteration+1}.fits", overwrite=True)
            trace.append(dict(stage=f"fixed_round_{iteration+1}", **evaluate(result, dataset)))
    elif mode in ("updated", "expanded", "default"):
        for iteration in range(2):
            options = dict(ADAPT, mesh_rounds=1)
            if mode == "updated":
                options.update(source_nodes=(method["nodes"],), regularizations=(method["regularization"],), seeds=(33,))
            elif mode == "expanded":
                options.update(source_nodes=(500, 1000, 2000), regularizations=(.2, 2., 20.))
            result = result.refine_adaptive(obs.macro_model, **options)
            result.save_fits(folder / f"round_{iteration+1}.fits", overwrite=True)
            result.save_json(folder / f"round_{iteration+1}.json", overwrite=True)
            trace.append(dict(stage=f"updated_round_{iteration+1}", **evaluate(result, dataset)))
    result = result.merge_candidates(radius=.05)
    save_result(result, folder)
    save_source_comparison(result, folder, info["morphology"])
    metrics = {**method, **evaluate(result, dataset), "dataset_info": info, "seconds": perf_counter()-started,
               "reused_reference": mode.startswith("reference"), "trace": trace,
               "input_sha256": digest(OUTPUT / "datasets" / dataset / "observation.fits"),
               "test_noise_used_only_for_evaluation": True}
    write_json(folder / "metrics.json", metrics)
    print(f"Finished {method['id']}: position={metrics['position_error_mas']}; "
          f"strength={metrics['strength_error_fraction']}; train={metrics['with_subhalo']['chi2_per_pixel']:.3f}; "
          f"test={metrics['independent_prediction_chi2_per_pixel']:.3f}; {metrics['seconds']:.1f}s", flush=True)
    return metrics


def summarize(records):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    order = {m["id"]: i for i, m in enumerate(METHODS)}
    records = sorted(records, key=lambda r: order[r["id"]])
    # Counterfactual prediction diagnostics are post-fit only, for every method.
    for r in records:
        with fits.open(OUTPUT / "methods" / r["id"] / "result.fits") as hdus:
            mask = hdus["MASK"].data.astype(bool); rms = hdus["RMS"].data
            folder = OUTPUT / "datasets" / r["dataset"]
            independent = fits.getdata(folder / "independent_observation.fits", "SCI")
            clean = fits.getdata(folder / "clean.fits")
            macro = hdus["MODEL_MACRO"].data
            r["independent_macro_chi2_per_pixel"] = float(np.mean(((independent-macro)[mask]/rms[mask])**2))
            r["clean_macro_error_per_pixel"] = float(np.mean(((clean-macro)[mask]/rms[mask])**2))
            r["independent_predictive_delta_chi2"] = float(mask.sum()*(
                r["independent_macro_chi2_per_pixel"]-r["independent_prediction_chi2_per_pixel"]))
            r["training_raw_delta_chi2"] = float(mask.sum()*(r["macro"]["chi2_per_pixel"]-r["with_subhalo"]["chi2_per_pixel"]))
        write_json(OUTPUT / "methods" / r["id"] / "metrics.json", r)
    write_json(OUTPUT / "summary.json", dict(completed=len(records), planned=len(METHODS), methods=records,
               independent_noise_never_used_for_selection=True, diagnostic_single_seed=True))
    spiral = [r for r in records if r["dataset"] == "spiral"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    y = np.arange(len(spiral)); labels = [r["id"] for r in spiral]
    for ax in axes: ax.set_yticks(y, labels); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    axes[0].barh(y, [r["position_error_mas"] for r in spiral], color=["#a9d7c4" if r["recovered"] else "#f4b79c" for r in spiral])
    axes[0].axvline(50, linestyle=":", color="gray"); axes[0].set_xlabel("Position error [mas]")
    axes[1].barh(y, [100*r["strength_error_fraction"] for r in spiral], color="#90aacc")
    axes[1].axvline(-30, linestyle=":", color="gray"); axes[1].axvline(30, linestyle=":", color="gray")
    axes[1].set_xlabel("Strength error [%]")
    axes[2].plot([r["with_subhalo"]["chi2_per_pixel"] for r in spiral], y, "o-", label="Training")
    axes[2].plot([r["independent_prediction_chi2_per_pixel"] for r in spiral], y, "o-", label="Independent noise")
    axes[2].set_xlabel("Image χ² / Npix"); axes[2].legend()
    for ax in axes[1:]: ax.set_yticklabels([])
    fig.suptitle("Same spiral observation and PINN proposals; controlled source-model ablations\n"
                 "Green: converged, position ≤50 mas and absolute strength error ≤30%")
    fig.savefig(OUTPUT / "source_ablation.png", dpi=175, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    lookup = {r["id"]: r for r in records}
    for n, color in zip((500, 1000, 2000), ("#4c78a8", "#f58518", "#54a24b")):
        for mode, linestyle in (("fixed", "--"), ("updated", "-")):
            row = lookup.get(f"{mode}_{n}_lambda2")
            if row is None: continue
            trace = row["trace"]
            for ax, key in zip(axes, ("position_error_mas", "train", "independent_prediction_chi2_per_pixel")):
                values = [r["with_subhalo"]["chi2_per_pixel"] if key == "train" else r[key] for r in trace]
                ax.plot(range(len(trace)), values, "o", color=color)
                ax.plot(range(len(trace)), values, linestyle=linestyle, color=color, label=f"{n}: {mode}")
    axes[0].axhline(50, linestyle=":", color="gray")
    for ax, ylabel in zip(axes, ("Position error [mas]", "Training χ² / Npix", "Independent prediction χ² / Npix")):
        ax.set(xticks=(0, 1, 2), xticklabels=("Initial", "Round 1", "Round 2"), ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Paired first-round meshes/predictions match; only updated methods regrid in round 2")
    fig.savefig(OUTPUT / "mesh_update_trajectory.png", dpi=175, bbox_inches="tight"); plt.close(fig)
    controls = [r for r in records if r["dataset"] != "spiral" or r["id"] == "original_adaptive_reference"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    values = np.array([[2 if r["recovered"] is None else int(r["recovered"])] for r in controls])
    if len(values): ax.imshow(values, cmap=ListedColormap(["#f4b79c", "#a9d7c4", "#d0d0d0"]), vmin=0, vmax=2, aspect="auto")
    for i, r in enumerate(controls):
        label = f"rho={r['dataset_info']['rho_image']:.1f}; RMS={r['dataset_info']['rms']:.5f}"
        if r["position_error_mas"] is not None: label += f"; {r['position_error_mas']:.1f} mas, {100*r['strength_error_fraction']:+.1f}%"
        else: label += f"; b_fit={r['strength_arcsec']:.4f}″; gain={r['delta_objective']:.1f}"
        ax.text(0, i, label, ha="center", va="center", fontsize=10)
    ax.set(yticks=range(len(controls)), yticklabels=[r["id"] for r in controls], xticks=[],
           title="Matched fixed-template perturbation SNR controls; same noise field scaled")
    fig.savefig(OUTPUT / "matched_snr_controls.png", dpi=175, bbox_inches="tight"); plt.close(fig)
    nulls = [r for r in records if r["dataset"] == "spiral_null"]
    if nulls:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        x = np.arange(len(nulls))
        ax.bar(x-.18, [r["training_raw_delta_chi2"] for r in nulls], .36, label="Fitted noise realization")
        ax.bar(x+.18, [r["independent_predictive_delta_chi2"] for r in nulls], .36, label="Independent noise realization")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(xticks=x, xticklabels=[r["id"] for r in nulls], ylabel="Raw Δχ²: macro minus halo",
               title="Null injections: positive gain improves that image; independent image was never fitted")
        ax.legend(); ax.grid(axis="y", alpha=.2)
        fig.savefig(OUTPUT / "null_prediction_check.png", dpi=175, bbox_inches="tight"); plt.close(fig)
    text = ["# Spiral source failure diagnosis\n", f"Completed {len(records)}/{len(METHODS)} methods.\n",
        "Identical spiral-data ablations reuse the five original PINN proposals/initial joint fit. "
        "The source coefficients always remain free; fixed means fixed vertices. Fixed and updated "
        "variants share node count, lambda, mesh seed=33, initial geometry and two joint rounds. "
        "The updated modes make two consecutive one-round API calls; their round_1/2 files retain each source selection. "
        "Regularization controls use fixed 1000-node meshes at lambda=0.2/2/20. Expanded CV uses "
        "500/1000/2000 nodes, lambda=0.2/2/20, four mesh seeds and two rounds. No core defaults changed.\n",
        "Matched SNR controls change only RMS: decrease Sersic SNR to the original spiral level; "
        "increase spiral SNR to the original Sersic level. This matches a scalar fixed-template "
        "perturbation norm, not the complete information matrix or illumination pattern. Changed "
        "noise datasets receive fresh blind PINN proposals.\n",
        "An independent Gaussian noise image (seed=271828) is evaluated after fitting and never "
        "used for source selection, starts or candidate ranking. Conditional training-pixel CV still "
        "uses inferred geometry; this is a single-noise scientific diagnosis, not independent recovery-rate calibration. "
        "Clean-image errors and halo truth are also evaluation-only.\n",
        "| Method | rho | Nodes | λ | Pos [mas] | Strength error | Train χ²/N | Independent χ²/N | Clean error/N | Independent Δχ² macro→halo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        pos = "—" if r["position_error_mas"] is None else f"{r['position_error_mas']:.2f}"
        amp = "—" if r["strength_error_fraction"] is None else f"{100*r['strength_error_fraction']:+.2f}%"
        text.append(f"| {r['id']} | {r['dataset_info']['rho_image']:.1f} | {r['source_nodes']} | {r['regularization']:g} | {pos} | {amp} | {r['with_subhalo']['chi2_per_pixel']:.3f} | {r['independent_prediction_chi2_per_pixel']:.3f} | {r['clean_model_error_per_pixel']:.3f} | {r['independent_predictive_delta_chi2']:+.1f} |")
    text += ["\nDatasets include input/independent FITS and clean truth; each method stores final "
             "FITS/JSON, per-round checkpoints, source truth/reconstruction and both residuals. Reference "
             "methods reuse prior results explicitly. Counts in summary include references, not 17 new fits.\n",
             "Run `.venv/bin/python sublens_testing/one_subhalo/spiral_source_diagnosis.py --workers 3`; "
             "cached final results are reused. `--summarize-only` redraws.\n"]
    text += ["\nSome completed fixed-grid fits required metadata recovery after an initial duplicate-key "
             "aggregation error. Their original fit runtimes are unavailable (seconds=null). "
             "Recovered metrics come directly from saved FITS and round checkpoints; optimization was not repeated.\n"]
    text += ["\nnull_prediction_check.png compares raw data χ² gains on fitted versus independent noise, "
             "without source-prior terms. Negative independent gain means the added candidate worsens prediction. "
             "These predictive metrics are evaluation-only and do not change the reported blind fit.\n"]
    (OUTPUT / "README.md").write_text("\n".join(text))
    if len(records) == len(METHODS):
        report = ["# 旋臂源失败诊断结果\n",
            "17项对照包含4项已保存参考和13项新拟合，覆盖5套噪声/SNR输入。"
            "同一输入的消融保留原PINN候选，网格固定时源亮度系数仍然重求。"
            "独立噪声图仅用于拟合后的评估，没有参与超参数或候选选择。\n",
            "## 网格更新对照\n",
            "每对方法使用相同节点数、λ=2、seed=33和两轮优化。第一轮的网格及图像预测完全一致；"
            "第二轮只有updated方法重新布点。以下为最终结果：\n",
            "| 节点数 | 固定网格位置误差 | 更新网格位置误差 | 固定/更新独立预测χ²/Npix |",
            "|---:|---:|---:|---:|"]
        for n in (500, 1000, 2000):
            a, b = (lookup[f"{mode}_{n}_lambda2"] for mode in ("fixed", "updated"))
            report.append(f"| {n} | {a['position_error_mas']:.1f} mas | {b['position_error_mas']:.1f} mas | {a['independent_prediction_chi2_per_pixel']:.3f} / {b['independent_prediction_chi2_per_pixel']:.3f} |")
        expanded = lookup["expanded_cv"]
        low, high = lookup["sersic_low_snr_default"], lookup["spiral_high_snr_default"]
        null = lookup["spiral_null_expanded_cv"]
        report += ["\n500/1000节点的重布点推动了参数漂移，而独立图像预测变化很小；"
                   "2000节点没有同样大的位置漂移。换网格同时改变基函数和离散正则，"
                   "这些结果支持源基底/先验与透镜参数存在耦合，不代表所有更新方式都会失败。\n",
            f"扩大CV后位置误差 {expanded['position_error_mas']:.1f} mas，强度误差 "
            f"{100*expanded['strength_error_fraction']:+.1f}%，独立预测χ²/Npix="
            f"{expanded['independent_prediction_chi2_per_pixel']:.3f}。定位改善，但强度偏差仍明显。\n",
            "## SNR和零注入对照\n",
            f"Sérsic降至rho={low['dataset_info']['rho_image']:.1f}后，位置误差仍为 "
            f"{low['position_error_mas']:.1f} mas；旋臂提高至rho={high['dataset_info']['rho_image']:.1f}后，"
            f"位置误差 {high['position_error_mas']:.1f} mas，强度误差 {100*high['strength_error_fraction']:+.1f}%。"
            "所以SNR和源形态/重构能力都影响结果；匹配标量rho不等于匹配完整局部信息。\n",
            f"零注入扩大CV后仍拟合出 b={null['strength_arcsec']:.5f} arcsec，惩罚增益 "
            f"{null['delta_objective']:.1f}。独立图像上macro-only χ²/Npix="
            f"{null['independent_macro_chi2_per_pixel']:.3f}，加入候选后="
            f"{null['independent_prediction_chi2_per_pixel']:.3f}，独立raw Δχ²="
            f"{null['independent_predictive_delta_chi2']:+.1f}，预测变差。扩大网格并未消除假候选。\n",
            "## 下一步\n",
            "保留固定高分辨率源网格作为对照，检验整个macro+halo+source联合拟合的留出预测，"
            "并检查跨网格参数稳定性。当前条件源CV不足以独立判断子晕参数是否稳定；"
            "不据此修改全局默认值或建立检出阈值。\n",
            "本轮仅一个源实现、一个主要训练噪声场，宏模型形状/斜率/中心仍固定在输入真值，"
            "PSF完全匹配。独立噪声预测检验并非新的独立子晕恢复试验，不能据此估计完备率或误检率。\n"]
        (OUTPUT / "REPORT.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    matplotlib.use("Agg")
    if args.workers < 1: parser.error("workers must be positive")
    if args.summarize_only:
        records = [json.loads((OUTPUT / "methods" / m["id"] / "metrics.json").read_text()) for m in METHODS
                   if (OUTPUT / "methods" / m["id"] / "metrics.json").exists()]
    else:
        prepare(); records = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_method, m) for m in METHODS]): records.append(future.result())
    summarize(records)


if __name__ == "__main__": main()
