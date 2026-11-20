"""Paired angular-strength scan using the current blind PINN/adaptive pipeline.

Run with .venv/bin/python sublens_testing/one_subhalo/strength_scan.py --workers 4.
The simulator and post-fit evaluator use halo truth; fit_observation does not.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch
from astropy.io import fits
from scipy.signal import fftconvolve

from sublens import (LensingObservation, LensingSimulator, MacroLensModel,
                     build_psf_kernel, sersic_profile, truncated_subhalo_deflection)
from run import ROOT, TRUTH, SEARCH, ADAPT, load_result, residual_metrics, save_result


OUTPUT = ROOT / "strength_scan"
STRENGTHS = (0., .002, .004, .007, .01, .015, .02, .03, .05, .08)
SEEDS = (314159,)
FWHM = .09
RMS = .012


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    OUTPUT.mkdir(exist_ok=True)
    reference = OUTPUT / "reference"
    reference.mkdir(exist_ok=True)
    config = dict(strengths_arcsec=list(STRENGTHS), noise_seeds=list(SEEDS), noise_rms=RMS,
                  psf_profile="gaussian", psf_fwhm_arcsec=FWHM, pixel_scale_arcsec=.05,
                  psf_pixel_quadrature=7, psf_known_exactly=True, paired_noise=True,
                  search=SEARCH, adaptive=ADAPT, gain_radius_arcsec=.22,
                  psf_guard_max_radius_arcsec=1.5*FWHM,
                  minimum_seed_separation_pixels=2, max_strength_arcsec=.12,
                  recovery=dict(position_error_mas=50., absolute_strength_error_fraction=.30,
                                macro_and_halo_converged=True),
                  null_is_diagnostic_only=True, existence_threshold_calibrated=False,
                  macro_shape_slope_center_fixed_at_simulation_values=True)
    old = OUTPUT / "config.json"
    if old.exists() and json.loads(old.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Existing strength scan has different settings; preserve that experiment")
    write_json(old, config)
    kernel, metadata = build_psf_kernel(FWHM, .05, profile="gaussian", return_metadata=True)
    psf_path = reference / "psf.fits"
    if not psf_path.exists():
        header = fits.Header(dict(PROFILE="gaussian", FWHM=FWHM, PIXSCALE=.05))
        fits.PrimaryHDU(kernel, header=header).writeto(psf_path, checksum=True)
    np.testing.assert_allclose(fits.getdata(psf_path), kernel, rtol=0, atol=1e-15)
    write_json(reference / "psf.json", metadata)
    write_json(reference / "truth.json", TRUTH)
    base = LensingObservation.from_fits(ROOT / "data" / "observation.fits")
    sim = LensingSimulator(N_img=TRUTH["image_size"], pixel_scale=TRUTH["pixel_scale"],
                          N_src=TRUTH["source_size"], L_src=TRUTH["source_half_width"],
                          psf_kernel=np.ones((1, 1)), device="cpu")
    header = fits.getheader(ROOT / "data" / "clean_image.fits")
    cases = []
    with torch.no_grad():
        sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
        source = sersic_profile(sx, sy, **TRUTH["source"])
        macro, _ = MacroLensModel(**TRUTH["macro"])(sim.theta_grid)
        tx, ty = sim.theta_grid.unbind(dim=1)
        for strength in STRENGTHS:
            halo = TRUTH["halo"]
            ax, ay = truncated_subhalo_deflection(tx, ty, halo["x"], halo["y"], strength,
                                                  halo["core"], halo["truncation"])
            native = sim.render(source[None, None], sim.theta_grid-macro-torch.stack((ax, ay), dim=-1))
            native = native.numpy().squeeze().astype(float)
            clean = fftconvolve(native, kernel, mode="same")
            key = f"strength_{strength:.6f}"
            for suffix, image in (("native", native), ("clean", clean)):
                path = reference / f"{key}_{suffix}.fits"
                if not path.exists():
                    fits.PrimaryHDU(image, header=header).writeto(path, checksum=True)
            if strength == 0:
                null_clean = clean
            rho = float(np.linalg.norm((clean-null_clean)[base.mask])/RMS)
            for seed in SEEDS:
                case = dict(id=f"{key}_seed_{seed}", injected_strength_arcsec=strength,
                            noise_seed=seed, is_null=strength == 0, rho_image=rho)
                folder = OUTPUT / "trials" / case["id"]
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "truth.json", {**TRUTH, "halo": {**halo, "strength": strength}})
                if not (folder / "observation.fits").exists():
                    noise = np.random.default_rng(seed).standard_normal(clean.shape)
                    obs = replace(base, image=clean+RMS*noise, rms=np.full(clean.shape, RMS),
                                  psf=kernel, lens_id=case["id"])
                    obs.to_fits(folder / "observation.fits")
                cases.append(case)
    write_json(OUTPUT / "manifest.json", dict(config=config, cases=cases))
    return cases


def fit_observation(obs, folder, *, resume=True):
    """Blind inference: no injected strength or halo position enters this function."""
    checkpoint = folder / "initial_joint.fits"
    if resume and checkpoint.exists():
        result = load_result(checkpoint)
    else:
        result = obs.search_pinn(**SEARCH, gain_radius=.22, psf_guard_max_radius=1.5*FWHM)
        result.save_json(folder / "initial_search.json", overwrite=True)
        result = result.refine_joint(obs.macro_model, maxiter=120, n_starts=2, retry_failed=True)
        result.save_json(folder / "initial_joint.json", overwrite=True)
        result.save_fits(checkpoint, overwrite=True)
    if result.candidates:
        result = result.refine_adaptive(obs.macro_model, **ADAPT)
    return result.merge_candidates(radius=obs.pixel_scale)


def run_trial(task):
    case, force = task
    torch.set_num_threads(1)
    matplotlib.use("Agg")
    folder = OUTPUT / "trials" / case["id"]
    if (folder / "metrics.json").exists() and not force:
        return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    obs = LensingObservation.from_fits(folder / "observation.fits")
    result = fit_observation(obs, folder, resume=not force)
    save_result(result, folder)
    metrics = dict(**case, seconds=perf_counter()-started,
                   proposals=len(result.residual_potential.peaks), unique_hypotheses=len(result.candidates),
                   macro=residual_metrics(obs, result.baseline.image_model),
                   macro_converged=result.macro_converged, halo_converged=None,
                   recovered=None if case["is_null"] else False,
                   position_arcsec=None, strength_arcsec=None, position_error_mas=None,
                   strength_error_fraction=None, delta_chi2=None, delta_objective=None,
                   observation_sha256=digest(folder / "observation.fits"))
    # Truth is consulted only after the pipeline has selected its best hypothesis.
    best = result.best_candidate
    if best is not None:
        fit = best.fit
        metrics.update(position_arcsec=list(fit.position), strength_arcsec=fit.strength,
                       halo_converged=fit.converged, delta_chi2=fit.delta_chi2,
                       delta_objective=fit.delta_objective,
                       with_subhalo=residual_metrics(obs, fit.image_model))
        if not case["is_null"]:
            halo = TRUTH["halo"]
            error = float(1000*np.linalg.norm(np.asarray(fit.position)-[halo["x"], halo["y"]]))
            strength_error = float(fit.strength/case["injected_strength_arcsec"]-1)
            metrics.update(position_error_mas=error, strength_error_fraction=strength_error,
                           recovered=bool(fit.converged and result.macro_converged
                                          and error <= 50 and abs(strength_error) <= .30))
    write_json(folder / "metrics.json", metrics)
    return metrics


def summarize(records):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import NullFormatter

    records = sorted(records, key=lambda r: (SEEDS.index(r["noise_seed"]), r["injected_strength_arcsec"]))
    planned = len(SEEDS)*len(STRENGTHS)
    write_json(OUTPUT / "summary.json", dict(completed=len(records), planned=planned,
               noise_seeds=list(SEEDS), strengths_arcsec=list(STRENGTHS), trials=records,
               recovery_is_not_existence_significance=True))
    lookup = {(r["noise_seed"], r["injected_strength_arcsec"]): r for r in records}
    fig, ax = plt.subplots(figsize=(14.5, 3.6+max(0, len(SEEDS)-1)), layout="constrained")
    matrix = np.full((len(SEEDS), len(STRENGTHS)), np.nan)
    for iy, seed in enumerate(SEEDS):
        for ix, strength in enumerate(STRENGTHS):
            row = lookup.get((seed, strength))
            if row is None:
                continue
            matrix[iy, ix] = 2 if row["is_null"] else int(row["recovered"])
            if row["is_null"]:
                label = "Null control" if row["strength_arcsec"] is None else f"Null control\nb_fit={row['strength_arcsec']:.4f}″"
            else:
                label = "No proposal" if row["position_error_mas"] is None else f"{row['position_error_mas']:.1f} mas\n{100*row['strength_error_fraction']:+.1f}%"
            ax.text(ix, iy, label, ha="center", va="center", fontsize=12)
    ax.imshow(matrix, cmap=ListedColormap(["#f4b79c", "#a9d7c4", "#d0d0d0"]), vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(len(STRENGTHS)), xticklabels=[f"{b:g}″" for b in STRENGTHS],
           yticks=range(len(SEEDS)), yticklabels=SEEDS, xlabel="Injected angular strength b [arcsec]", ylabel="Noise seed",
           title="Gaussian PSF FWHM=0.09″; RMS=0.012; pixel=0.05″\n"
                 "Green: converged, position ≤50 mas and strength error ≤30%; gray: null control")
    fig.savefig(OUTPUT / "strength_recovery_matrix.png", dpi=175, bbox_inches="tight")
    fig.savefig(OUTPUT / "strength_recovery_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    injected = [r for r in records if not r["is_null"] and r["position_error_mas"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for seed in SEEDS:
        rows = [r for r in injected if r["noise_seed"] == seed]
        x = [r["injected_strength_arcsec"] for r in rows]
        axes[0].plot(x, [r["position_error_mas"] for r in rows], "o-", label=str(seed))
        axes[1].plot(x, [100*r["strength_error_fraction"] for r in rows], "o-")
        axes[2].plot(x, [r["macro"]["chi2_per_pixel"] for r in rows], "o--", label="Macro")
        axes[2].plot(x, [r["with_subhalo"]["chi2_per_pixel"] for r in rows], "o-", label="Macro + halo")
    axes[0].axhline(50, color="gray", linestyle=":")
    for y in (-30, 30):
        axes[1].axhline(y, color="gray", linestyle=":")
    axes[0].set(ylabel="Position error [mas]", yscale="log")
    axes[1].set(ylabel="Strength error [%]")
    axes[1].set_yscale("symlog", linthresh=30)
    axes[1].axhspan(-30, 30, color="#a9d7c4", alpha=.3)
    axes[2].set(ylabel="χ² / Npix")
    for ax in axes:
        ax.set(xlabel="Injected angular strength [arcsec]", xscale="log")
        ax.set_xticks([.002, .004, .01, .03, .08], ["0.002", "0.004", "0.01", "0.03", "0.08"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
    axes[0].legend(); axes[2].legend()
    fig.savefig(OUTPUT / "strength_recovery_curves.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Latest-pipeline single-halo angular-strength scan\n",
             f"Completed {len(records)}/{planned} trials, noise seeds {list(SEEDS)}.\n",
             "一个子晕，注入位置 (−1.28,0.61) arcsec，core=0.035、truncation=0.22 arcsec；"
             "Gaussian detector-response PSF FWHM=0.09 arcsec，pixel=0.05 arcsec，RMS=0.012。"
             "仅改变 b，所有强度使用同一噪声场；注入源、macro、PSF、mask 和总源通量保持相同。\n",
             "流程：PINN 五个提名；候选去重间距 0.10 arcsec 与增益孔径 0.22 arcsec 分开；"
             "先独立物理精修，再多起点联合精修；两轮条件留出选择源节点数、网格种子、正则，"
             "每次透镜变化重新求源；最终按精修位置合并重复假设。"
             "theta_E 和两个 shear 分量可变，macro 形状、斜率、中心固定在模拟值。"
             "子晕位置/强度真值只用于生成和最终评估；全部档位使用相同初始化和上界 b=0.12 arcsec。\n",
             "恢复标准：macro 和 halo 收敛，位置误差 ≤50 mas，强度误差 ≤30%。"
             "最佳候选按拟合目标选出，再与真值比较，不能用最近真值的候选替换。"
             "未加入未经校准的存在性阈值；零注入只作流程诊断，单个零注入不能校准假阳性率。\n",
             "PSF 使用 build_psf_kernel 的 7×7 像素积分，与最近 PSF 扫描的 Gaussian 0.09 arcsec 完全相同。"
             "交叉拟合隔离半径上限为 1.5×FWHM，此窄 Gaussian 的默认 98% 半径小于上限。"
             "模拟与拟合使用相同已知 PSF。\n",
             "rho_image 为固定真值源/macro 时的图像扰动范数除以 RMS，未扣除源/macro 重拟合可吸收的信号，"
             "delta_objective 包含源正则，两者均非校准的 sigma 显著性。"
             "b 是该固定轮廓的角尺度强度，不是物理质量；单 seed 的首次成功档位不是普适阈值。\n",
             "| b [arcsec] | Position [mas] | Strength error | Recovered | χ²/Npix macro → halo | Penalized gain |",
             "|---:|---:|---:|---|---:|---:|"]
    for r in records:
        position = "—" if r["position_error_mas"] is None else f"{r['position_error_mas']:.2f}"
        error = "—" if r["strength_error_fraction"] is None else f"{100*r['strength_error_fraction']:+.2f}%"
        status = "Null control" if r["is_null"] else str(r["recovered"])
        chi = f"{r['macro']['chi2_per_pixel']:.4f}"
        if "with_subhalo" in r:
            chi += f" → {r['with_subhalo']['chi2_per_pixel']:.4f}"
        gain = "—" if r["delta_objective"] is None else f"{r['delta_objective']:.2f}"
        lines.append(f"| {r['injected_strength_arcsec']:g} | {position} | {error} | {status} | {chi} | {gain} |")
    lines += ["\nreference/：PSF、各档 native/clean FITS、注入真值；trials/：输入 observation.fits、"
              "初始提名/联合精修、最终 result.json/result.fits、macro 与 macro+halo 残差、源重构和候选图。\n",
              "复现：`.venv/bin/python sublens_testing/one_subhalo/strength_scan.py --workers 4`；"
              "已有结果会复用，`--force` 重拟合，`--summarize-only` 仅汇总重画。"
              "中断后可从 initial_joint.fits 继续；最终候选的选择使用相同模型比较目标。\n"]
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    matplotlib.use("Agg"); torch.set_num_threads(1)
    if args.summarize_only:
        cases = json.loads((OUTPUT / "manifest.json").read_text())["cases"]
        records = [json.loads((OUTPUT / "trials" / c["id"] / "metrics.json").read_text())
                   for c in cases if (OUTPUT / "trials" / c["id"] / "metrics.json").exists()]
    else:
        cases = generate()
        records = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_trial, (c, args.force)) for c in cases]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                records.append(row)
                print(f"[{index}/{len(cases)}] {row['id']}: recovered={row['recovered']}; "
                      f"position={row['position_error_mas']}; {row['seconds']:.1f}s", flush=True)
    summarize(records)


if __name__ == "__main__":
    main()
