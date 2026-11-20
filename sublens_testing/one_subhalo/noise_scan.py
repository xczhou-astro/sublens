"""Paired noise scan for the one-subhalo PINN pipeline.

All fits start from observations, never injected halo coordinates. The same
standard-normal noise field is scaled across noise levels within each seed;
different seeds are independent. Recovery is evaluated after candidate selection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import traceback

import matplotlib
import numpy as np
import torch
from astropy.io import fits

from sublens import LensingObservation, LensingSimulator, MacroLensModel, sersic_profile, truncated_subhalo_deflection
from run import ROOT as BASE_ROOT, TRUTH, SEARCH, ADAPT, residual_metrics


OUTPUT = BASE_ROOT / "noise_scan"
FACTORS = (1., 2., 4., 8., 16., 32.)
SEEDS = (314159, 271828, 161803, 141421)
RECOVERY = dict(position_error_mas=50., absolute_strength_error_fraction=.30,
                both_optimizers_converged=True, detection_threshold_calibrated=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(factors, seeds):
    OUTPUT.mkdir(exist_ok=True)
    base = LensingObservation.from_fits(BASE_ROOT / "data" / "observation.fits")
    reference = OUTPUT / "reference"
    reference.mkdir(exist_ok=True)
    p = TRUTH
    sim = LensingSimulator(N_img=p["image_size"], pixel_scale=p["pixel_scale"],
                          N_src=p["source_size"], L_src=p["source_half_width"],
                          psf_fwhm=p["psf_fwhm"], noise_std=p["noise_rms"], device="cpu")
    with torch.no_grad():
        y, x = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
        source = sersic_profile(x, y, **p["source"])
        macro, _ = MacroLensModel(**p["macro"])(sim.theta_grid)
        x, y = sim.theta_grid.unbind(dim=1)
        clean0, _ = sim.simulate(source[None, None], macro, add_noise=False)
        clean0 = clean0.numpy().squeeze().astype(float)
        strength_norms = []
        for strength in (.00625, .0125, .025, .05, .10):
            hx, hy = truncated_subhalo_deflection(x, y, p["halo"]["x"], p["halo"]["y"], strength)
            image, _ = sim.simulate(source[None, None], macro+torch.stack((hx, hy), dim=-1), add_noise=False)
            signal_norm = float(np.linalg.norm((image.numpy().squeeze()-clean0)[base.mask]))
            strength_norms.append(dict(strength_arcsec=strength, signal_l2=signal_norm,
                                       rho_at_baseline=signal_norm/p["noise_rms"]))
    clean1 = fits.getdata(BASE_ROOT / "data" / "clean_image.fits").astype(float)
    delta = clean1-clean0
    header = fits.getheader(BASE_ROOT / "data" / "clean_image.fits")
    for name, image in (("clean_macro", clean0), ("clean_with_subhalo", clean1), ("delta_image", delta)):
        path = reference / f"{name}.fits"
        if not path.exists():
            fits.PrimaryHDU(image, header=header).writeto(path, checksum=True)
    config = dict(factors=list(factors), noise_seeds=list(seeds), base_rms=p["noise_rms"],
                  search=SEARCH, adaptive=ADAPT, recovery=RECOVERY, threads_per_worker=1,
                  common_random_numbers_across_noise_levels=True,
                  halo_truth_not_used_by_inference=True,
                  macro_shape_slope_center_fixed_at_simulation_values=True,
                  noise_model="independent Gaussian pixels; uniform RMS in image units",
                  noise_rng="numpy.random.default_rng(seed).standard_normal(image_shape)")
    config_path = OUTPUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Existing noise_scan has another configuration; preserve it in a separate folder")
    write_json(config_path, config)
    signal_norm = float(np.linalg.norm(delta[base.mask]))
    write_json(reference / "signal_metrics.json", dict(
        definition="rho_image = sqrt(sum_mask((clean_halo-clean_macro)^2 / RMS^2))",
        known_source_and_macro=True, nuisance_parameters_not_profiled=True,
        signal_l2=signal_norm, rho_at_baseline=signal_norm/p["noise_rms"],
        strengths=strength_norms,
    ))
    cases = []
    for seed in seeds:
        noise = np.random.default_rng(seed).standard_normal(base.image.shape)
        for factor in factors:
            sigma = p["noise_rms"]*factor
            case = dict(id=f"noise_{factor:g}x_seed_{seed}", factor=factor, noise_rms=sigma,
                        seed=seed, rho_image=signal_norm/sigma)
            folder = OUTPUT / "trials" / case["id"]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "observation.fits"
            if not path.exists():
                observation = replace(base, image=clean1+sigma*noise, rms=np.full(base.image.shape, sigma),
                                      lens_id=case["id"])
                observation.to_fits(path)
            cases.append(case)
    write_json(OUTPUT / "manifest.json", dict(config=config, cases=cases,
                                              truth_for_evaluation_only=p["halo"]))
    return cases


def run_trial(task):
    case, force = task
    torch.set_num_threads(1)
    matplotlib.use("Agg")
    folder = OUTPUT / "trials" / case["id"]
    observation_path = folder / "observation.fits"
    if (folder / "inference.json").exists() and not force:
        return json.loads((folder / "inference.json").read_text())
    started = perf_counter()
    observation = LensingObservation.from_fits(observation_path)
    # Preserve the historical scan when library defaults evolve. These are the
    # original 0.44 arcsec seed exclusion and 0.22 arcsec gain aperture.
    result = observation.search_pinn(**SEARCH, separation=.44, gain_radius=.22)
    result.save_json(folder / "initial_search.json", overwrite=True)
    result = result.refine_joint(observation.macro_model, maxiter=120, n_starts=2, retry_failed=True)
    result.save_json(folder / "initial_joint.json", overwrite=True)
    result = result.refine_adaptive(observation.macro_model, **ADAPT)
    result.save_fits(folder / "result.fits", overwrite=True)
    result.save_json(folder / "result.json", overwrite=True)
    import matplotlib.pyplot as plt

    for figure in (result.plot_residuals(save_path=folder / "residuals.png"),
                   result.plot_fit(save_path=folder / "source_reconstruction.png"),
                   result.plot(zoom=max(.28, result.refinement_radius+.1), save_path=folder / "pinn_candidates.png")):
        plt.close(figure)
    best = result.best_candidate
    record = dict(case_id=case["id"], seconds=perf_counter()-started,
                  observation_sha256=digest(observation_path), candidate_count=len(result.candidates),
                  refinement_radius_arcsec=result.refinement_radius, macro_converged=result.macro_converged,
                  macro=residual_metrics(observation, result.baseline.image_model),
                  best_rank=None if best is None else best.rank,
                  with_subhalo=None if best is None else residual_metrics(observation, best.fit.image_model))
    write_json(folder / "inference.json", record)
    return record


def wilson(k, n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def evaluate(case, truth):
    folder = OUTPUT / "trials" / case["id"]
    if not (folder / "inference.json").exists():
        failed = (folder / "error.json").exists()
        return dict(**case, completed=failed, inference_failed=failed, recovered=False)
    inference = json.loads((folder / "inference.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    best = next((r for r in result["candidates"] if r["is_best"]), None)
    row = dict(**case, completed=True, recovered=False, inference=inference)
    if best is not None:
        # Candidate selection has already happened without consulting truth.
        position_error = float(1000*np.linalg.norm(np.asarray(best["position_arcsec"])-[truth["x"], truth["y"]]))
        strength_error = float(best["strength_arcsec"]/truth["strength"]-1)
        converged = bool(best["converged"] and result["macro_converged"])
        radius = result["refinement_radius_arcsec"]
        seeds = np.array([r["seed_position_arcsec"] for r in result["candidates"]])
        covered = bool((np.max(np.abs(seeds-[truth["x"], truth["y"]]), axis=1) <= radius).any())
        row.update(position_error_mas=position_error, strength_error_fraction=strength_error,
                   position_arcsec=best["position_arcsec"], strength_arcsec=best["strength_arcsec"],
                   converged=converged, delta_chi2=best["delta_chi2"], gain=best["delta_objective"],
                   true_position_in_any_final_proposal_box=covered,
                   recovered=bool(converged and position_error <= RECOVERY["position_error_mas"]
                                  and abs(strength_error) <= RECOVERY["absolute_strength_error_fraction"]))
    write_json(folder / "evaluation.json", row)
    return row


def summarize(cases):
    config = json.loads((OUTPUT / "config.json").read_text())
    extension = OUTPUT / "extension_manifest.json"
    if extension.exists():
        extra = json.loads(extension.read_text())
        known = {c["id"] for c in cases}
        cases = list(cases)+[c for c in extra["cases"] if c["id"] not in known]
        config["initial_noise_factors"] = list(config["factors"])
        config["factors"] = sorted(set(config["factors"]+extra["factors"]))
        config["exploratory_extension_reason"] = extra["reason"]
    rows = [evaluate(case, TRUTH["halo"]) for case in cases]
    grouped = []
    for factor in config["factors"]:
        trials = [r for r in rows if r["factor"] == factor]
        complete = [r for r in trials if r["completed"]]
        scored = [r for r in complete if "position_error_mas" in r]
        n, k = len(complete), sum(r["recovered"] for r in complete)
        grouped.append(dict(factor=factor, noise_rms=config["base_rms"]*factor,
                            rho_image=trials[0]["rho_image"], completed=n, planned=len(trials),
                            recovered=k, recovery_rate=None if not n else k/n,
                            recovery_wilson95=None if not n else wilson(k, n),
                            converged=sum(r.get("converged", False) for r in complete),
                            inference_failed=sum(r.get("inference_failed", False) for r in complete),
                            localized=sum(r.get("position_error_mas", float("inf")) <= RECOVERY["position_error_mas"]
                                          for r in complete),
                            strength_accurate=sum(abs(r.get("strength_error_fraction", float("inf")))
                                                 <= RECOVERY["absolute_strength_error_fraction"] for r in complete),
                            proposal_boxes_cover_truth=sum(r.get("true_position_in_any_final_proposal_box", False)
                                                           for r in complete),
                            median_position_error_mas=None if not scored else float(np.median([r["position_error_mas"] for r in scored])),
                            median_absolute_strength_error_fraction=None if not scored else float(np.median([abs(r["strength_error_fraction"]) for r in scored]))))
    summary = dict(config=config, completed=sum(r["completed"] for r in rows), planned=len(rows),
                   groups=grouped, trials=rows, limitations=[
                       "Exploratory recovery scan, not a calibrated detection false-alarm threshold",
                       "Four noise fields per level; wide binomial uncertainty, paired across levels",
                       "Known input macro; only theta_E and shear free; one source/profile/location/PSF",
                       "Image-template SNR knows the generating source; macro/source absorption not removed",
                       "Fixed pipeline settings; nonmonotonic recovery can reflect proposal/optimizer failures",
                   ])
    ready = [g for g in grouped if g["completed"] == g["planned"]]
    summary["empirical_transition"] = dict(
        interpretation="Observed downward crossings at high noise; nonmonotonic rates do not define one population sigma_50",
        all_trials_recovered_factors=[g["factor"] for g in ready if g["recovered"] == g["planned"]],
        observed_rates_nonmonotonic=any(a["recovery_rate"] < b["recovery_rate"] for a, b in zip(ready, ready[1:])),
        above_half_to_at_most_half_brackets=[
            dict(noise_rms=[a["noise_rms"], b["noise_rms"]],
                 rho_image=[b["rho_image"], a["rho_image"]],
                 observed_recovery_rates=[a["recovery_rate"], b["recovery_rate"]])
            for a, b in zip(ready, ready[1:]) if a["recovery_rate"] > .5 and b["recovery_rate"] <= .5],
    )
    baseline_failures = [r for r in rows if r["factor"] == 1 and r["completed"] and not r["recovered"]]
    summary["baseline_recovery_failures"] = [dict(seed=r["seed"],
        position_error_mas=r.get("position_error_mas"), strength_error_fraction=r.get("strength_error_fraction"),
        converged=r.get("converged"), inference_failed=r.get("inference_failed", False)) for r in baseline_failures]
    diagnostic_path = OUTPUT / "diagnostic_retry_low_noise_seed161803" / "metrics.json"
    diagnostic = None if not diagnostic_path.exists() else json.loads(diagnostic_path.read_text())
    if diagnostic is not None:
        summary["posthoc_diagnostic_excluded_from_recovery_counts"] = diagnostic
    write_json(OUTPUT / "summary.json", summary)
    plot(summary)
    table = ["| Noise factor | RMS | rho_image | Accurate / completed | Median position [mas] | Median abs. strength error |",
             "|---:|---:|---:|---:|---:|---:|"]
    for g in grouped:
        pos = "—" if g["median_position_error_mas"] is None else f"{g['median_position_error_mas']:.1f}"
        amp = "—" if g["median_absolute_strength_error_fraction"] is None else f"{100*g['median_absolute_strength_error_fraction']:.1f}%"
        table.append(f"| {g['factor']:g} | {g['noise_rms']:g} | {g['rho_image']:.2f} | {g['recovered']}/{g['completed']} | {pos} | {amp} |")
    report = (
        "# Single-subhalo noise recovery scan\n\n"
        "固定强度 0.05 arcsec、位置 (−1.28, 0.61) arcsec、源、PSF 与 mask。改变每像素 Gaussian RMS；"
        "每档使用 4 个独立噪声场，同一个 seed 的噪声场在各档按比例放大。候选选择完成后才用真值评估。\n\n"
        "恢复标准：macro 与 halo 优化均收敛、位置误差 ≤50 mas、强度相对误差绝对值 ≤30%。"
        "此处评估参数恢复能力，没有校准子晕存在检验的假阳性率。\n\n"
        "定义 `delta_image = clean_macro_plus_halo − clean_macro`，两者使用相同真源和 macro 并经过 PSF；"
        "`rho_image = sqrt(sum_mask((delta_image/RMS)**2))`。相关噪声推广为 `sqrt(delta_image.T @ C^-1 @ delta_image)`。"
        "这是无量纲的注入扰动信噪比；源和 macro 可重新拟合吸收部分扰动，它不是发现显著性。"
        "强度[arcsec]/噪声[图像通量]并非无量纲，不能作为通用阈值；弱扰动时 delta I≈−grad(S)·delta alpha，"
        "固定位置、源、PSF 下 rho 才近似随强度/噪声缩放。reference/signal_metrics.json 给出不同强度的精确前向计算。\n\n"
        + "\n".join(table)+"\n\n"
        "位置和强度误差的中位数包含所有有候选的试验，并非只统计成功样本。"
        "图中恢复率误差条为逐档 Wilson 95% 区间。4 次试验不足以可靠测量 90%/95% 完备率阈值；"
        "这里应报告恢复率下降的区间和条件，不能称为普适质量或噪声阈值。源验证同样是条件于拟合几何的验证。\n\n"
        "实际观测使用相同拟合自由度和源先验比较 raw Δchi2 与含正则的 gain；"
        "全盲搜索、边界、候选筛选和源网格选择都会影响其零假设分布，不能将 sqrt(Δchi2) 或 sqrt(gain) 直接称为 N sigma。"
        "存在性阈值需用相同噪声水平的零注入完整流程校准并用独立零注入验证，本扫描不自动继承旧流程的阈值。\n\n"
        "参考：[Vegetti et al. 2014](https://arxiv.org/abs/1405.3666) 讨论位置依赖和 macro/source 吸收；"
        "[O'Riordan & Vegetti 2024](https://arxiv.org/abs/2310.10714) 研究 macro 角向复杂度对灵敏度的影响。\n\n"
        "复现：`.venv/bin/python sublens_testing/one_subhalo/noise_scan.py --workers 2`；"
        "高噪声扩展：`.venv/bin/python sublens_testing/one_subhalo/extend_noise_scan.py`；"
        "使用 `--summarize-only` 更新汇总，`--force` 重新拟合。trials/ 每个条件保存输入 FITS、"
        "完整结果 FITS/JSON、残差、源重构、PINN 候选及真值评估。\n"
    )
    if extension.exists():
        report += ("\n初始方案为 1–32 倍噪声；因首个噪声场在 32 倍仍恢复成功，"
                   "另行记录扩展方案至 64、128 倍，以观察失败端。恢复标准和拟合设置未改变。\n")
    if baseline_failures:
        report += (f"\n重要：原噪声档也有 {len(baseline_failures)} 个恢复失败。"
                   "高信噪比不保证当前提名、局部优化和源网格更新流程找到正确解；"
                   "曲线可能非单调，不能把所有失效归于噪声。具体 seed 和误差列在 summary.json。\n")
    for interval in summary["empirical_transition"]["above_half_to_at_most_half_brackets"]:
        lower, upper = interval["noise_rms"]
        rho_low, rho_high = interval["rho_image"]
        report += (f"\n高噪声段观察到的恢复率下降区间：RMS={lower:g}–{upper:g}（当前图像单位），"
                   f"对应 rho_image={rho_high:.2f}–{rho_low:.2f}。这是离散小样本结果，不是已测准的 sigma_50。\n")
    if diagnostic is not None:
        report += ("\n另存事后诊断：seed=161803、原噪声，初始精修半径从 0.35 增至 0.56 arcsec，"
                   f"其余设置不变。位置误差 {diagnostic['position_error_mas']:.1f} mas，"
                   f"强度误差 {100*diagnostic['strength_error_fraction']:+.1f}%，"
                   f"χ²/Npix={diagnostic['with_subhalo']['chi2_per_pixel']:.3f}。"
                   "该重试不替换主扫描结果，单独保存于 diagnostic_retry_low_noise_seed161803/。\n")
    report += ("\n图表：noise_recovery.png 为噪声、误差与拟合分数；paired_recovery.png 列出每次恢复；"
               "recovery_vs_snr.png 使用无量纲 rho 横轴；noise_gallery.png 展示预先选定首个 seed 的观测与两种残差。"
               "strength_noise_snr.csv 是不同强度和噪声下的精确前向扰动 SNR，属于已知真源诊断，"
               "不代表已完成二维参数恢复扫描。运行 verify_noise_scan.py 检查保存的输入、噪声场、结果和 FITS 校验和。\n")
    (OUTPUT / "README.md").write_text(report)
    return summary


def plot(summary):
    import matplotlib.pyplot as plt

    groups, trials = summary["groups"], summary["trials"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for g in groups:
        if g["completed"]:
            rate = g["recovery_rate"]
            lo, hi = g["recovery_wilson95"]
            axes[0, 0].errorbar(g["factor"], rate, yerr=[[rate-lo], [hi-rate]], fmt="o", color="#2374ab", capsize=3)
    ready = [g for g in groups if g["completed"]]
    axes[0, 0].plot([g["factor"] for g in ready], [g["recovery_rate"] for g in ready], color="#2374ab", alpha=.5)
    axes[0, 0].set(ylabel="Accurate recovery fraction", ylim=(-.03, 1.03), title="Position <=50 mas; strength error <=30%")
    for r in trials:
        if "position_error_mas" not in r:
            continue
        style = dict(marker="o" if r["converged"] else "x", color="#238443" if r["recovered"] else "#d95f0e", alpha=.75)
        axes[0, 1].scatter(r["factor"], max(r["position_error_mas"], .01), **style)
        axes[1, 0].scatter(r["factor"], 100*r["strength_error_fraction"], **style)
        axes[1, 1].scatter(r["factor"], r["gain"], **style)
    axes[0, 1].axhline(50, color="gray", linestyle="--")
    axes[0, 1].set(ylabel="Position error [mas]", yscale="log", title="All selected candidates, including failures")
    axes[1, 0].axhspan(-30, 30, color="green", alpha=.1)
    axes[1, 0].set(ylabel="Strength error [%]", title="Green: accurate recovery; orange: otherwise")
    axes[1, 1].set(ylabel="Penalized fit gain", yscale="symlog", title="Gain is not a sigma significance")
    for ax in axes.flat:
        ax.set(xscale="log", xlabel="Noise RMS / 0.012", xticks=summary["config"]["factors"])
        ax.set_xticklabels([f"{f:g}" for f in summary["config"]["factors"]])
        ax.minorticks_off()
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT / "noise_recovery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    matplotlib.use("Agg")
    torch.set_num_threads(1)
    if args.summarize_only:
        cases = json.loads((OUTPUT / "manifest.json").read_text())["cases"]
    else:
        cases = generate(FACTORS, SEEDS)
        started = perf_counter()
        # Interleave seeds at each level so partial results cover the noise range.
        ordered = sorted(cases, key=lambda c: (SEEDS.index(c["seed"]), FACTORS.index(c["factor"])))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_trial, (case, args.force)): case for case in ordered}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    result = future.result()
                    print(f"[{index}/{len(cases)}] {case['id']}: chi2/pixel="
                          f"{None if result['with_subhalo'] is None else result['with_subhalo']['chi2_per_pixel']}; "
                          f"{result['seconds']:.1f}s", flush=True)
                except Exception:
                    message = traceback.format_exc()
                    write_json(OUTPUT / "trials" / case["id"] / "error.json", dict(error=message))
                    print(f"[{index}/{len(cases)}] FAILED {case['id']}: {message}", flush=True)
        write_json(OUTPUT / "runtime.json", dict(seconds=perf_counter()-started, workers=args.workers))
    summary = summarize(cases)
    print(json.dumps(summary["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
