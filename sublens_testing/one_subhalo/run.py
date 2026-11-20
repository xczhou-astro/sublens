"""Reproducible one-subhalo PINN recovery, source adaptation and residual checks.

Run with the workspace Python, e.g. .venv/bin/python sublens_testing/one_subhalo/run.py.
Inference uses the observation bundle, including an input macro geometry. Halo
truth is used for simulation and evaluation, never as candidate positions or starts.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from time import perf_counter

import matplotlib
import numpy as np
import torch
from astropy.io import fits
from scipy.spatial import Delaunay

from sublens import LensingObservation, LensingSimulator, MacroLensModel, sersic_profile, truncated_subhalo_deflection
from sublens.inversion.halo_search import HaloFit, PinnHaloCandidate, PinnHaloSearchResult
from sublens.inversion.potential import ResidualPotential
from sublens.inversion.source_plane import PixelSource, PixelSourceProblem
from sublens.inversion.adaptive_refinement import _model_rays


ROOT = Path(__file__).resolve().parent
TRUTH = dict(
    macro=dict(theta_E=1.35, q=.80, phi=.35, gamma=2.05, gamma_ext=.045, phi_ext=.20, center=[0., 0.]),
    halo=dict(x=-1.28, y=.61, strength=.05, core=.035, truncation=.22),
    source=dict(center=[.04, -.03], i_eff=.30, r_eff=.22, n_sersic=1.25, q=.78, phi=.30),
    pixel_scale=.05, image_size=100, source_size=257, source_half_width=1.6,
    psf_fwhm=.09, noise_rms=.012, original_noise_seed=42, repeat_noise_seed=314159,
)
SEARCH = dict(source_nodes=300, regularization=20., mesh_strategy="centroidal",
              top_k=5, pinn_steps=90, refine_steps=70)
ADAPT = dict(source_nodes=(200, 350, 500), regularizations=(2., 20., 200.), seeds=(11, 22, 33, 44),
             mesh_rounds=2, maxiter=120, n_starts=3, retry_failed=True, expand_boundary=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare_data():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    reference = ROOT / "reference"
    reference.mkdir(exist_ok=True)
    old = ROOT.parent / "pinn_subhalo_results"
    for name in ("residuals.npz", "residual_comparison.png", "pinn_convergence.png", "subhalo_fit.png"):
        if (old / name).exists() and not (reference / name).exists():
            shutil.copy2(old / name, reference / name)
    if not (data / "observation.fits").exists():
        shutil.copy2(old / "simulation_injected.fits", data / "observation.fits")
    base = LensingObservation.from_fits(data / "observation.fits")
    sim = LensingSimulator(N_img=TRUTH["image_size"], pixel_scale=TRUTH["pixel_scale"],
                          N_src=TRUTH["source_size"], L_src=TRUTH["source_half_width"],
                          psf_fwhm=TRUTH["psf_fwhm"], noise_std=TRUTH["noise_rms"], device="cpu")
    with torch.no_grad():
        y, x = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
        source = sersic_profile(x, y, **TRUTH["source"])
        alpha, _ = MacroLensModel(**TRUTH["macro"])(sim.theta_grid)
        tx, ty = sim.theta_grid.unbind(dim=1)
        halo = TRUTH["halo"]
        hx, hy = truncated_subhalo_deflection(tx, ty, halo["x"], halo["y"], halo["strength"])
        total = alpha + torch.stack((hx, hy), dim=-1)
        _, clean = sim.simulate(source[None, None], total, add_noise=False)
        for name, deflection, seed in (("repeat_observation.fits", total, TRUTH["repeat_noise_seed"]),
                                        ("null_observation.fits", alpha, TRUTH["original_noise_seed"])):
            if not (data / name).exists():
                image, _ = sim.simulate(source[None, None], deflection, seed=seed)
                obs = LensingObservation.from_simulation(sim, image, alpha, macro_model=TRUTH["macro"],
                                                          mask=base.mask, lens_id=name.removesuffix(".fits"))
                obs.to_fits(data / name)
    null_path = data / "null_observation.fits"
    null = LensingObservation.from_fits(null_path)
    if not np.array_equal(null.mask, base.mask):
        replace(null, mask=base.mask).to_fits(null_path, overwrite=True)
    # Store truth separately from the observation bundle used for inference.
    header = fits.Header()
    for axis in (1, 2):
        header[f"CTYPE{axis}"] = "LINEAR"
        header[f"CUNIT{axis}"] = "arcsec"
        header[f"CRPIX{axis}"] = (TRUTH["source_size"] + 1) / 2
        header[f"CRVAL{axis}"] = 0.
        header[f"CDELT{axis}"] = 2 * TRUTH["source_half_width"] / (TRUTH["source_size"] - 1)
    fits.PrimaryHDU(source.numpy(), header).writeto(data / "source_truth.fits", overwrite=True, checksum=True)
    image_header = fits.Header()
    for axis in (1, 2):
        image_header[f"CTYPE{axis}"] = "LINEAR"
        image_header[f"CUNIT{axis}"] = "arcsec"
        image_header[f"CRPIX{axis}"] = (TRUTH["image_size"] + 1) / 2
        image_header[f"CRVAL{axis}"] = 0.
        image_header[f"CDELT{axis}"] = TRUTH["pixel_scale"]
    fits.PrimaryHDU(clean.numpy().squeeze(), image_header).writeto(data / "clean_image.fits", overwrite=True, checksum=True)
    write_json(data / "truth.json", TRUTH)
    write_json(ROOT / "settings.json", dict(
        search=SEARCH, adaptive=ADAPT, halo_truth_not_used_by_inference=True,
        macro_input="Simulation macro supplied; theta_E and shear refined; shape, slope and center fixed",
    ))


def load_result(path):
    """Restore saved coefficients and parameters without another optimization."""
    obs = LensingObservation.from_fits(path)
    with fits.open(path) as hdus:
        metadata = json.loads(hdus["METADATA"].data["JSON"][0])
        points = np.column_stack((hdus["SOURCE_NODES"].data["X"], hdus["SOURCE_NODES"].data["Y"]))
        grid = Delaunay(points)
        coefficients = {name: hdus["SOURCE_NODES"].data[name].copy()
                        for name in hdus["SOURCE_NODES"].columns.names if "FLUX" in name}
        baseline_image = hdus["MODEL_MACRO"].data.copy()
        best_image = hdus["MODEL"].data.copy()
        potential = ResidualPotential(hdus["PINN_MASK"].data.astype(bool),
                                     hdus["PINN_BASE_RES"].data.copy(), hdus["PINN_FIT_RES"].data.copy(),
                                     hdus["PINN_KAPPA"].data.copy(), [])
    problem = PixelSourceProblem(obs.image, obs.rms, obs.psf, obs.mask, obs.bx, obs.by,
                                 source_grid=grid, regularization=metadata["baseline"]["regularization"],
                                 mesh_strategy=metadata["baseline"].get("mesh_strategy") or "random")
    def restore_source(flux, prediction, objective):
        return PixelSource(flux, prediction, "delaunay", grid, problem.regularization,
                           objective, mesh_strategy=problem.mesh_strategy)

    baseline = restore_source(coefficients["BASE_FLUX"], baseline_image,
                              metadata["baseline"]["source_objective"])
    candidates = []
    for row in metadata["candidates"]:
        position = row["position_arcsec"]
        shell = HaloFit(*position, row["strength_arcsec"], row["delta_chi2"], row["delta_objective"],
                        row["converged"], baseline, baseline.image_model, row["macro_model"],
                        optimizer_message=row.get("optimizer_message"), iterations=row.get("iterations"))
        hx, hy = _model_rays(obs, row["macro_model"] or obs.macro_model, shell)
        flux = coefficients[f"FLUX_{row['rank']}"]
        prediction = best_image if row["is_best"] else problem.render(hx, hy, flux)
        source = restore_source(flux, prediction, row["source_objective"])
        fit = replace(shell, source=source, image_model=source.image_model)
        candidates.append(PinnHaloCandidate(row["rank"], tuple(row["seed_position_arcsec"]), row["proposal_score"], fit))
        potential.peaks.append((*row["seed_position_arcsec"], row["proposal_score"]))
    config = metadata["config"]
    radius = config.get("search_radius_arcsec", [.7 * obs.einstein_radius, 1.3 * obs.einstein_radius])
    return PinnHaloSearchResult(obs, baseline, potential, tuple(candidates), tuple(radius),
                               macro_model=metadata["macro_model_baseline"] if metadata["mode"] == "joint_refinement" else None,
                               macro_converged=metadata["macro_converged"],
                               max_strength=metadata.get("max_strength_arcsec", .12),
                               refinement_radius=metadata.get("refinement_radius_arcsec", config.get("refinement_radius_arcsec", .35)),
                               config=config)


def residual_metrics(obs, model):
    residual = (obs.image - model) / obs.rms
    mask = obs.mask
    correlations = []
    for axis in (0, 1):
        a, b = (slice(None, -1), slice(1, None))
        first, second = [slice(None)] * 2, [slice(None)] * 2
        first[axis], second[axis] = a, b
        first, second = tuple(first), tuple(second)
        valid = mask[first] & mask[second]
        correlations.append(float(np.corrcoef(residual[first][valid], residual[second][valid])[0, 1]))
    return dict(chi2_per_pixel=float(np.mean(residual[mask]**2)),
                rms_normalized=float(np.std(residual[mask])),
                lag1_correlation_y=correlations[0], lag1_correlation_x=correlations[1])


def save_result(result, output):
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    result.save_json(output / "result.json", overwrite=True)
    result.save_fits(output / "result.fits", overwrite=True)
    figures = [result.plot_residuals(save_path=output / "residual_comparison.png"),
               result.plot(zoom=max(.28, result.refinement_radius+.1), save_path=output / "pinn_candidates.png"),
               result.plot_fit(save_path=output / "source_reconstruction.png")]
    for figure in figures:
        plt.close(figure)


def run_case(case, force=False):
    filename = dict(base="observation.fits", repeat="repeat_observation.fits", null="null_observation.fits")[case]
    obs = LensingObservation.from_fits(ROOT / "data" / filename)
    output = ROOT / ("improved" if case == "base" else case)
    output.mkdir(exist_ok=True)
    started = perf_counter()
    if (output / "result.fits").exists() and not force:
        return load_result(output / "result.fits")
    resumed = (output / "initial_joint.fits").exists() and not force
    if resumed:
        result = load_result(output / "initial_joint.fits")
    else:
        result = obs.search_pinn(**SEARCH)
        result.save_json(output / "initial_search.json", overwrite=True)
        result = result.refine_joint(obs.macro_model, maxiter=120, n_starts=2, retry_failed=True)
        result.save_fits(output / "initial_joint.fits", overwrite=True)
        result.save_json(output / "initial_joint.json", overwrite=True)
    result = result.refine_adaptive(obs.macro_model, **ADAPT, verbose=True)
    save_result(result, output)
    write_json(output / "runtime.json", dict(seconds=perf_counter()-started, case=case,
                                             resumed_from_initial_joint=resumed,
                                             includes_search_and_initial_joint=not resumed))
    result.summary()
    return result


def summarize(result, case):
    fit = result.best_candidate.fit if result.best_candidate is not None else None
    output = ROOT / ("improved" if case == "base" else case)
    metrics = dict(case=case, macro=residual_metrics(result.observation, result.baseline.image_model))
    if fit is not None:
        metrics.update(with_subhalo=residual_metrics(result.observation, fit.image_model),
                       position_arcsec=list(fit.position), strength_arcsec=fit.strength,
                       halo_converged=fit.converged, macro_converged=result.macro_converged)
        if case != "null":
            halo = TRUTH["halo"]
            metrics.update(position_error_mas=float(1000*np.hypot(fit.x-halo["x"], fit.y-halo["y"])),
                           strength_error_fraction=float(fit.strength/halo["strength"]-1))
    history = result.config.get("adaptive_source", {}).get("history", [])
    if history:
        metrics["conditional_source_validation"] = history[-1]["selected"]
    write_json(output / "metrics.json", metrics)
    return metrics


def compare_base(result, metrics):
    import matplotlib.pyplot as plt

    old = np.load(ROOT / "reference" / "residuals.npz")
    obs = result.observation
    current = result.residual_with_subhalo
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    half = obs.image.shape[0]*obs.pixel_scale/2
    for ax, residual, title in zip(axes, [result.residual_macro, old["residual_with_subhalo"], current],
                                   ["Macro only / updated source", "Previous macro + halo", "Improved macro + halo"]):
        shown = np.where(obs.mask, residual/obs.rms, np.nan)
        artist = ax.imshow(shown, origin="lower", extent=(-half, half, -half, half), cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set(title=f"{title}\nχ²/Npix={np.nanmean(shown**2):.3f}", xlabel="θx [arcsec]", ylabel="θy [arcsec]")
    fig.colorbar(artist, ax=axes, shrink=.8, label="Observed − model / RMS")
    fig.savefig(ROOT / "before_after_residuals.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    halo = TRUTH["halo"]
    figure = result.plot(truth_position=(halo["x"], halo["y"]),
                         zoom=max(.28, result.refinement_radius+.1),
                         save_path=ROOT / "improved" / "pinn_candidates.png")
    plt.close(figure)
    previous = residual_metrics(obs, old["macro_subhalo_model"])
    checks = {}
    for case in ("repeat", "null"):
        path = ROOT / case / "metrics.json"
        if path.exists():
            checks[case] = json.loads(path.read_text())
    report = (
        "# 单子晕基础测试\n\n"
        "只注入一个子晕，位置和强度不传入提名或精修。使用原始观测、RMS 和 PSF，保持与旧结果相同的有效像素。\n\n"
        "改进：亮度加权 centroidal Delaunay 布点；留出预测选择节点数、随机种子和正则；"
        "触边时扩展候选搜索范围；多起点联合精修；每次重新求源；外层更新网格。"
        "两种质量假设在每一轮使用相同的网格、正则和 macro 自由度。\n\n"
        f"旧 macro+halo χ²/Npix={previous['chi2_per_pixel']:.4f}；"
        f"新 macro+halo χ²/Npix={metrics['with_subhalo']['chi2_per_pixel']:.4f}。\n\n"
        f"位置误差={metrics['position_error_mas']:.2f} mas；"
        f"强度相对误差={100*metrics['strength_error_fraction']:+.2f}%；"
        f"halo 收敛={metrics['halo_converged']}；macro 收敛={metrics['macro_converged']}。\n\n"
        "χ²/Npix 按相同的 3744 个有效像素计算，没有扣除有效自由度；它不是 reduced χ²。"
        "源布点和正则改变后 macro-only χ²/Npix 也会改变，不能用 Δχ² 的增大直接宣称探测显著性提高。\n\n"
        "源验证指标是在当前拟合透镜几何下的条件预测，不能视作整套发现流程的独立盲测成绩。"
        "macro 的归一化和两个剪切分量可调整，形状、斜率和中心固定在模拟输入值。"
        "新流程的子晕存在阈值尚未经过大样本零注入校准。\n\n"
        "运行：`.venv/bin/python sublens_testing/one_subhalo/run.py --case base`；"
        "独立噪声复测：`--case repeat`；零注入检查：`--case null`。缓存保存于各结果目录，"
        "使用 `--force` 可重新计算。子晕和源的真值只用于生成与结果评估。"
        "每个候选分别比较单子晕假设，5 个提名不代表同时拟合或注入 5 个子晕。"
        "强度是截断子晕模型的角尺度参数，单位 arcsec，不是物理质量。\n\n"
    )
    if "repeat" in checks:
        check = checks["repeat"]
        report += (f"独立噪声复测（seed=314159）：χ²/Npix={check['with_subhalo']['chi2_per_pixel']:.4f}，"
                   f"位置误差={check['position_error_mas']:.2f} mas，"
                   f"强度误差={100*check['strength_error_fraction']:+.2f}%。\n\n")
    if "null" in checks:
        check = checks["null"]
        report += (f"零注入检查：macro χ²/Npix={check['macro']['chi2_per_pixel']:.4f}，"
                   f"加扰动后={check['with_subhalo']['chi2_per_pixel']:.4f}，"
                   f"最佳强度={check['strength_arcsec']:.6f} arcsec。该弱扰动不能自动解释为检出。\n\n")
    report += (
        "文件：data/ 保存 image、RMS、PSF、mask、输入 macro 的 FITS，以及独立的真值；"
        "improved/、repeat/、null/ 各保存结果 FITS/JSON、macro 和 macro+halo 残差图、源重构图、候选图及优化日志；"
        "reference/ 是改进前的模型和残差；development/ 仅保存探索过程，不作为最终结果。"
        "根目录 before_after_residuals.png 用相同 ±4 RMS 色标比较旧、新残差。"
        "FITS 的 MODEL_MACRO/RESID_MACRO 和 MODEL/RESIDUAL 分别是两种假设的图像及 observed−model；"
        "SOURCE_NODES 包含源节点和系数，CANDIDATES/METADATA 包含拟合参数和配置。\n\n"
        "基础调用（已读入 observation）：\n\n"
        "```python\n"
        "result = observation.search_pinn(mesh_strategy='centroidal', source_nodes=300, regularization=20.)\n"
        "result = result.refine_joint(n_starts=2, retry_failed=True, maxiter=120)\n"
        "result = result.refine_adaptive(source_nodes=(200, 350, 500), regularizations=(2., 20., 200.),\n"
        "                                seeds=(11, 22, 33, 44), n_starts=3, maxiter=120)\n"
        "result.plot_residuals()\n"
        "```\n"
    )
    (ROOT / "README.md").write_text(report)
    write_json(ROOT / "results_summary.json", dict(previous=previous, improved=metrics, checks=checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("base", "repeat", "null", "all"), default="base")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    matplotlib.use("Agg")
    torch.set_num_threads(1)
    prepare_data()
    cases = ("base", "repeat", "null") if args.case == "all" else (args.case,)
    for case in cases:
        print(f"\nRunning one-subhalo case: {case}", flush=True)
        result = run_case(case, force=args.force)
        metrics = summarize(result, case)
        if case == "base":
            compare_base(result, metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    if args.case != "base" and (ROOT / "improved" / "result.fits").exists():
        base = load_result(ROOT / "improved" / "result.fits")
        compare_base(base, summarize(base, "base"))


if __name__ == "__main__":
    main()
