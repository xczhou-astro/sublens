"""Paired null/halo tests of complex sources with the latest unchanged pipeline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from time import perf_counter

import matplotlib
import numpy as np
import torch
from astropy.io import fits
from scipy.signal import fftconvolve

from sublens import (LensingObservation, LensingSimulator, MacroLensModel,
                     build_psf_kernel, sersic_profile, truncated_subhalo_deflection)
from sublens.simulation.sources import generate_complex_source
from run import ROOT, TRUTH, SEARCH, ADAPT, residual_metrics, save_result, load_result
from strength_scan import fit_observation, write_json, digest


OUTPUT = ROOT / "complex_source"
MORPHOLOGIES = ("sersic", "clumpy", "merger", "spiral_ism")
STRENGTHS = (0., .05)
NOISE_SEED = 314159
SOURCE_SEED = 101
RMS = .012
FWHM = .09


def make_sources(sim):
    """Generate source truth only; inference consumes the observation bundle."""
    sy, sx = torch.meshgrid(sim.src_coords_1d, sim.src_coords_1d, indexing="ij")
    smooth = sersic_profile(sx, sy, **TRUTH["source"])
    disk = sersic_profile(sx, sy, center=(.04, -.03), i_eff=.45, r_eff=.22,
                          n_sersic=1., q=.8, phi=.3)
    clumps = ((.04, -.03, .035, 1.20), (.12, .02, .026, .95),
              (-.05, -.10, .028, .90), (.08, -.14, .024, 1.15), (-.02, .06, .030, .70))
    clumpy = disk.clone()
    for x, y, sigma, amplitude in clumps:
        clumpy += amplitude*torch.exp(-.5*((sx-x)**2+(sy-y)**2)/sigma**2)
    merger = sersic_profile(sx, sy, center=(.01, -.06), i_eff=.75, r_eff=.14,
                            n_sersic=1.3, q=.70, phi=.3)
    merger += sersic_profile(sx, sy, center=(.11, .05), i_eff=.65, r_eff=.09,
                             n_sersic=1.1, q=.60, phi=-.6)
    for x, y, sigma, amplitude in ((.06, 0., .025, .90), (.15, -.05, .028, .85),
                                    (-.08, -.15, .027, .80)):
        merger += amplitude*torch.exp(-.5*((sx-x)**2+(sy-y)**2)/sigma**2)
    # Isolate the public generator's global RNG side effects from the caller.
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            spiral, _ = generate_complex_source(N_src=sim.N_src, L_src=sim.L_src,
                                                 seed=SOURCE_SEED, device="cpu")
        finally:
            np.random.set_state(numpy_state)
    raw = dict(sersic=smooth, clumpy=clumpy, merger=merger, spiral_ism=spiral.squeeze())
    return {kind: source*(smooth.sum()/source.sum()) for kind, source in raw.items()}


def source_header():
    header = fits.Header()
    for axis in (1, 2):
        header[f"CTYPE{axis}"] = "LINEAR"
        header[f"CUNIT{axis}"] = "arcsec"
        header[f"CRPIX{axis}"] = (TRUTH["source_size"]+1)/2
        header[f"CRVAL{axis}"] = 0.
        header[f"CDELT{axis}"] = 2*TRUTH["source_half_width"]/(TRUTH["source_size"]-1)
    return header


def generate():
    OUTPUT.mkdir(exist_ok=True)
    reference = OUTPUT / "reference"
    reference.mkdir(exist_ok=True)
    config = dict(morphologies=list(MORPHOLOGIES), strengths_arcsec=list(STRENGTHS),
                  noise_seed=NOISE_SEED, source_seed=SOURCE_SEED, source_flux_matched=True,
                  image_peak_not_renormalized=True, noise_rms=RMS, pixel_scale_arcsec=.05,
                  psf_profile="gaussian", psf_fwhm_arcsec=FWHM, search=SEARCH, adaptive=ADAPT,
                  fit_pipeline="strength_scan.fit_observation", paired_noise=True,
                  macro_shape_slope_center_fixed_at_simulation_values=True,
                  recovery=dict(position_error_mas=50., absolute_strength_error_fraction=.30,
                                macro_and_halo_converged=True),
                  nulls_diagnostic_only=True, false_positive_rate_calibrated=False)
    old = OUTPUT / "config.json"
    if old.exists() and json.loads(old.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Existing complex-source experiment differs; preserve its results")
    write_json(old, config)
    kernel, psf_info = build_psf_kernel(FWHM, .05, return_metadata=True)
    if not (reference / "psf.fits").exists():
        fits.PrimaryHDU(kernel, header=fits.Header(dict(FWHM=FWHM, PIXSCALE=.05))).writeto(
            reference / "psf.fits", checksum=True)
    write_json(reference / "psf.json", psf_info)
    base = LensingObservation.from_fits(ROOT / "data" / "observation.fits")
    sim = LensingSimulator(N_img=TRUTH["image_size"], pixel_scale=.05, N_src=TRUTH["source_size"],
                          L_src=TRUTH["source_half_width"], psf_kernel=np.ones((1, 1)), device="cpu")
    image_header = fits.getheader(ROOT / "data" / "clean_image.fits")
    cases, statistics = [], {}
    with torch.no_grad():
        sources = make_sources(sim)
        macro, _ = MacroLensModel(**TRUTH["macro"])(sim.theta_grid)
        tx, ty = sim.theta_grid.unbind(dim=1)
        coords = sim.src_coords_1d.numpy().astype(float)
        sx, sy = np.meshgrid(coords, coords)
        for kind, tensor in sources.items():
            source = tensor.numpy().astype(float)
            source_path = reference / f"{kind}_source.fits"
            if not source_path.exists():
                fits.PrimaryHDU(source, header=source_header()).writeto(source_path, checksum=True)
            total = source.sum()
            cx, cy = float((source*sx).sum()/total), float((source*sy).sum()/total)
            radii = np.hypot(sx-cx, sy-cy).ravel()
            order = np.argsort(radii)
            half_light = radii[order[np.searchsorted(np.cumsum(source.ravel()[order]), total/2)]]
            statistics[kind] = dict(source_integral=float(total*(coords[1]-coords[0])**2),
                                    centroid_arcsec=[cx, cy], circular_half_light_radius_arcsec=float(half_light),
                                    peak_brightness=float(source.max()))
            for strength in STRENGTHS:
                h = TRUTH["halo"]
                ax, ay = truncated_subhalo_deflection(tx, ty, h["x"], h["y"], strength,
                                                      h["core"], h["truncation"])
                native = sim.render(tensor[None, None], sim.theta_grid-macro-torch.stack((ax, ay), dim=-1))
                native = native.numpy().squeeze().astype(float)
                clean = fftconvolve(native, kernel, mode="same")
                key = f"{kind}_strength_{strength:.6f}"
                for suffix, image in (("native", native), ("clean", clean)):
                    path = reference / f"{key}_{suffix}.fits"
                    if not path.exists():
                        fits.PrimaryHDU(image, header=image_header).writeto(path, checksum=True)
                if strength == 0:
                    null_clean = clean
                case = dict(id=f"{key}_seed_{NOISE_SEED}", morphology=kind,
                            injected_strength_arcsec=strength, is_null=strength == 0,
                            noise_seed=NOISE_SEED, rho_image=float(np.linalg.norm((clean-null_clean)[base.mask])/RMS),
                            clean_lensed_flux_in_mask=float(clean[base.mask].sum()), source_statistics=statistics[kind])
                folder = OUTPUT / "trials" / case["id"]
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "truth.json", dict(common_parameters=TRUTH, morphology=kind,
                           source_file=f"reference/{kind}_source.fits", injected_strength_arcsec=strength))
                if not (folder / "observation.fits").exists():
                    z = np.random.default_rng(NOISE_SEED).standard_normal(clean.shape)
                    obs = replace(base, image=clean+RMS*z, rms=np.full(clean.shape, RMS), psf=kernel,
                                  lens_id=case["id"])
                    obs.to_fits(folder / "observation.fits")
                cases.append(case)
    write_json(reference / "source_statistics.json", statistics)
    write_json(OUTPUT / "manifest.json", dict(config=config, cases=cases))
    plot_truth_gallery()
    return cases


def plot_truth_gallery():
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    arrays = [fits.getdata(OUTPUT / "reference" / f"{k}_source.fits") for k in MORPHOLOGIES]
    maximum = max(a.max() for a in arrays)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
    for ax, kind, array in zip(axes, MORPHOLOGIES, arrays):
        shown = ax.imshow(array, origin="lower", extent=(-1.60625, 1.60625, -1.60625, 1.60625),
                          cmap="magma", norm=AsinhNorm(linear_width=.03*maximum, vmin=0, vmax=maximum))
        ax.set(title=kind, xlim=(-.65, .65), ylim=(-.65, .65), xlabel="βx [arcsec]", ylabel="βy [arcsec]")
    fig.colorbar(shown, ax=axes, shrink=.75, label="Source surface brightness (asinh)",
                 ticks=[0, .1*maximum, maximum], format="%.2g")
    fig.suptitle("Injected source morphologies; matched intrinsic integral")
    fig.savefig(OUTPUT / "source_truth_gallery.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_source_comparison(result, folder, kind):
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from scipy.ndimage import map_coordinates

    axis = np.linspace(-.65, .65, 201)
    sx, sy = np.meshgrid(axis, axis)
    truth = fits.getdata(OUTPUT / "reference" / f"{kind}_source.fits")
    scale = (TRUTH["source_size"]-1)/(2*TRUTH["source_half_width"])
    truth = map_coordinates(truth, [(sy+1.6)*scale, (sx+1.6)*scale], order=1, mode="constant")
    best = result.best_candidate
    source = result.baseline if best is None else best.fit.source
    arrays = (truth, result.baseline.sample(sx, sy), source.sample(sx, sy))
    header = fits.Header()
    for n in (1, 2):
        header[f"CTYPE{n}"] = "LINEAR"; header[f"CUNIT{n}"] = "arcsec"
        header[f"CRPIX{n}"] = 101.; header[f"CRVAL{n}"] = 0.; header[f"CDELT{n}"] = .0065
    hdus = [fits.PrimaryHDU()]
    for name, array in zip(("TRUTH", "MACRO_SOURCE", "HALO_SOURCE"), arrays):
        hdus.append(fits.ImageHDU(array, header=header, name=name))
    fits.HDUList(hdus).writeto(folder / "source_comparison.fits", overwrite=True, checksum=True)
    maximum = max(np.nanmax(a) for a in arrays)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, array, title in zip(axes, arrays, ("Truth", "Macro source", "Macro + halo source")):
        shown = ax.imshow(array, origin="lower", extent=(-.65325, .65325, -.65325, .65325),
                          cmap="magma", norm=AsinhNorm(linear_width=.03*maximum, vmin=0, vmax=maximum))
        ax.set(title=title, xlabel="βx [arcsec]", ylabel="βy [arcsec]")
    fig.colorbar(shown, ax=axes, shrink=.75, label="Surface brightness",
                 ticks=[0, .1*maximum, maximum], format="%.2g")
    fig.suptitle(f"{kind}: common source coordinates; qualitative comparison")
    fig.savefig(folder / "source_truth_reconstruction.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def run_trial(task):
    case, force = task
    torch.set_num_threads(1); matplotlib.use("Agg")
    folder = OUTPUT / "trials" / case["id"]
    if (folder / "metrics.json").exists() and not force:
        return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    obs = LensingObservation.from_fits(folder / "observation.fits")
    result = fit_observation(obs, folder, resume=not force)
    save_result(result, folder)
    save_source_comparison(result, folder, case["morphology"])
    metrics = dict(**case, seconds=perf_counter()-started, macro=residual_metrics(obs, result.baseline.image_model),
                   macro_converged=result.macro_converged, halo_converged=None, position_arcsec=None,
                   strength_arcsec=None, position_error_mas=None, strength_error_fraction=None,
                   recovered=None if case["is_null"] else False, delta_objective=None, delta_chi2=None,
                   unique_hypotheses=len(result.candidates), proposals=len(result.residual_potential.peaks),
                   observation_sha256=digest(folder / "observation.fits"))
    best = result.best_candidate
    if best is not None:
        fit = best.fit
        metrics.update(position_arcsec=list(fit.position), strength_arcsec=fit.strength,
                       halo_converged=fit.converged, delta_objective=fit.delta_objective,
                       delta_chi2=fit.delta_chi2, with_subhalo=residual_metrics(obs, fit.image_model))
        if not case["is_null"]:
            h = TRUTH["halo"]
            position_error = float(1000*np.linalg.norm(np.array(fit.position)-[h["x"], h["y"]]))
            strength_error = float(fit.strength/case["injected_strength_arcsec"]-1)
            metrics.update(position_error_mas=position_error, strength_error_fraction=strength_error,
                           recovered=bool(fit.converged and result.macro_converged
                                          and position_error <= 50 and abs(strength_error) <= .30))
    write_json(folder / "metrics.json", metrics)
    return metrics


def summarize(records):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, AsinhNorm

    records = sorted(records, key=lambda r: (MORPHOLOGIES.index(r["morphology"]), r["injected_strength_arcsec"]))
    lookup = {(r["morphology"], r["injected_strength_arcsec"]): r for r in records}
    write_json(OUTPUT / "summary.json", dict(completed=len(records), planned=8, trials=records,
               paired_nulls_not_false_positive_rate_calibration=True))
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    matrix = np.full((4, 2), np.nan)
    for iy, kind in enumerate(MORPHOLOGIES):
        for ix, strength in enumerate(STRENGTHS):
            r = lookup.get((kind, strength))
            if r is None: continue
            matrix[iy, ix] = 2 if r["is_null"] else int(r["recovered"])
            if r["strength_arcsec"] is None:
                label = "No proposal"
            elif r["is_null"]:
                label = f"b_fit={r['strength_arcsec']:.4f}″\nPenalized gain={r['delta_objective']:.1f}"
            else:
                label = f"{r['position_error_mas']:.1f} mas\n{100*r['strength_error_fraction']:+.1f}%"
            ax.text(ix, iy, label, ha="center", va="center", fontsize=12)
    ax.imshow(matrix, cmap=ListedColormap(["#f4b79c", "#a9d7c4", "#d0d0d0"]), vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=(0, 1), xticklabels=("Null: b=0", "Injected: b=0.05″"), yticks=range(4), yticklabels=MORPHOLOGIES,
           ylabel="Source morphology", title="Fixed source integral; Gaussian FWHM=0.09″; RMS=0.012; seed=314159\n"
           "Green: converged, position ≤50 mas and strength error ≤30%; gray: diagnostic null")
    fig.savefig(OUTPUT / "source_recovery_matrix.png", dpi=175, bbox_inches="tight")
    fig.savefig(OUTPUT / "source_recovery_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    # Full image-plane residuals on a shared noise-unit scale, for injected cases.
    fig, axes = plt.subplots(4, 2, figsize=(8, 13), layout="constrained")
    shown = None
    for row, kind in enumerate(MORPHOLOGIES):
        r = lookup.get((kind, .05))
        for ax in axes[row]: ax.set_axis_off()
        if r is None: continue
        with fits.open(OUTPUT / "trials" / r["id"] / "result.fits") as hdus:
            mask, rms = hdus["MASK"].data.astype(bool), hdus["RMS"].data
            for col, (name, key, title) in enumerate((("RESID_MACRO", "macro", "Macro"),
                                                      ("RESIDUAL", "with_subhalo", "Macro + halo"))):
                ax = axes[row, col]; ax.set_axis_on()
                image = np.where(mask, hdus[name].data/rms, np.nan)
                shown = ax.imshow(image, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5),
                                  cmap="RdBu_r", vmin=-4, vmax=4)
                ax.set(title=f"{kind}: {title}; χ²/Npix={r[key]['chi2_per_pixel']:.3f}",
                       xlabel="θx [arcsec]", ylabel="θy [arcsec]")
    if shown is not None:
        fig.colorbar(shown, ax=axes, shrink=.45, label="Observed − model / RMS")
    fig.savefig(OUTPUT / "residual_overview.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), layout="constrained")
    for row, kind in enumerate(MORPHOLOGIES):
        r = lookup.get((kind, .05))
        for ax in axes[row]: ax.set_axis_off()
        if r is None: continue
        with fits.open(OUTPUT / "trials" / r["id"] / "source_comparison.fits") as rasters:
            arrays = [rasters[name].data for name in ("TRUTH", "MACRO_SOURCE", "HALO_SOURCE")]
            maximum = max(np.nanmax(a) for a in arrays)
            for ax, array, title in zip(axes[row], arrays, ("Truth", "Macro source", "Macro + halo source")):
                ax.set_axis_on()
                shown = ax.imshow(array, origin="lower", extent=(-.65325, .65325, -.65325, .65325),
                                  cmap="magma", norm=AsinhNorm(linear_width=.03*maximum, vmin=0, vmax=maximum))
                ax.set(title=f"{kind}: {title}", xlabel="βx [arcsec]", ylabel="βy [arcsec]")
            fig.colorbar(shown, ax=axes[row], shrink=.7, ticks=[0, .1*maximum, maximum], format="%.2g")
    fig.suptitle("Qualitative source comparison; common coordinates and shared brightness scale within each row")
    fig.savefig(OUTPUT / "source_reconstruction_overview.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    spiral = lookup.get(("spiral_ism", .05))
    if spiral is not None:
        stages = []
        folder = OUTPUT / "trials" / spiral["id"]
        for stage in ("initial_search", "initial_joint", "result"):
            metadata = json.loads((folder / f"{stage}.json").read_text())
            best = next((c for c in metadata["candidates"] if c["is_best"]), None)
            if best is None: continue
            h = TRUTH["halo"]
            stages.append(dict(stage=stage, position_arcsec=best["position_arcsec"],
                               position_error_mas=float(1000*np.linalg.norm(np.array(best["position_arcsec"])-[h["x"], h["y"]])),
                               strength_arcsec=best["strength_arcsec"],
                               strength_error_fraction=best["strength_arcsec"]/.05-1,
                               chi2_per_pixel=best["chi2_per_pixel"],
                               halo_converged=best["converged"], macro_converged=metadata["macro_converged"]))
        write_json(OUTPUT / "spiral_stage_diagnosis.json", dict(
            stages=stages, source_selection=metadata["config"].get("adaptive_source", {}).get("history", []),
            evaluation_only=True, objectives_change_with_source_basis=True,
            interpretation="Records where the position shifts; does not isolate a causal failure mechanism"))
    lines = ["# Complex source recovery with paired nulls\n", f"Completed {len(records)}/8 trials.\n",
             "四种源：原始 Sérsic、diffuse disk + 五亮结、双 Sérsic 核 + 三亮结、"
             "sublens.generate_complex_source 的旋臂/六亮结/湍流/尘埃模型（seed=101）。"
             "源的栅格积分通量归一到 Sérsic 基准；不按图像峰值归一。源的质心、半光半径、"
             "局部照明和透镜放大率仍会改变，不能把差异全部归因于抽象的复杂度。\n",
             "每种源各注入 b=0 和 b=0.05 arcsec；非零位置固定 (−1.28,0.61) arcsec，core=0.035、"
             "truncation=0.22 arcsec。Gaussian 像素积分 PSF FWHM=0.09 arcsec，pixel=0.05 arcsec，"
             "RMS=0.012，8 组共用同一 seed=314159 噪声场。模拟和拟合的 PSF 完全匹配。\n",
             "严格复用 strength_scan.fit_observation：五候选，独立评分孔径与去重距离，"
             "多起点联合精修、两轮条件留出选择源网格/正则/种子，透镜变化后重求源，最后合并重复解。"
             "所有形态使用相同计算预算；源节点候选为200/350/500，正则2/20/200。"
             "theta_E 和两个剪切分量可调；macro形状、斜率和中心固定在模拟值。"
             "源真值、子晕位置与强度不进入拟合，只作模拟/最终评估。\n",
             "非零恢复标准：macro/halo收敛、位置误差≤50 mas、强度误差≤30%。"
             "最佳解按拟合目标选取；零注入里的候选仅是拟合假设，未按未经校准的增益判为检出。"
             "每种形态只有一个噪声实现；四个配对零注入不足以标定假阳性率。"
             "rho_image 是固定真值源和macro的图像扰动SNR，不是检测显著性；惩罚增益包含源正则。\n",
             "| Source | b injected | b fitted | Position [mas] | Strength error | Recovered | χ²/Npix macro → halo | Penalized gain | rho_image |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in records:
        pos = "—" if r["position_error_mas"] is None else f"{r['position_error_mas']:.2f}"
        error = "—" if r["strength_error_fraction"] is None else f"{100*r['strength_error_fraction']:+.2f}%"
        fitted = "—" if r["strength_arcsec"] is None else f"{r['strength_arcsec']:.6f}"
        gain = "—" if r["delta_objective"] is None else f"{r['delta_objective']:.2f}"
        status = "Null control" if r["is_null"] else str(r["recovered"])
        chi = f"{r['macro']['chi2_per_pixel']:.3f}"
        if "with_subhalo" in r: chi += f" → {r['with_subhalo']['chi2_per_pixel']:.3f}"
        lines.append(f"| {r['morphology']} | {r['injected_strength_arcsec']:g} | {fitted} | {pos} | {error} | {status} | {chi} | {gain} | {r['rho_image']:.1f} |")
    lines += ["\nsource_truth_gallery.png：共同绝对色标下的注入源；source_recovery_matrix.png：配对结果。"
              "residual_overview.png：四种注入源的 macro/halo 残差，统一±4 RMS色标；"
              "source_reconstruction_overview.png：每行同一色标下的源真值、macro源、macro+halo源。"
              "reference/ 保存源真值及WCS、PSF、native/clean图和源统计；trials/ 保存输入、"
              "拟合FITS/JSON、PINN候选图、重构及两类残差。source_truth_reconstruction.png "
              "在共同源坐标画真值、macro源、macro+halo源；未将未约束区域的源差异当作定量恢复精度。\n",
              "spiral_stage_diagnosis.json 保存旋臂源初始提名、初始联合精修和最终解的参数漂移；"
              "属于事后评估，不进入候选选择，也不能单独隔离失败的因果机制。\n",
              "复现：`.venv/bin/python sublens_testing/one_subhalo/complex_source_scan.py --workers 4`。"
              "`--summarize-only` 重画汇总，`--force` 重拟合；每个案例可从 initial_joint.fits 断点继续。\n"]
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    matplotlib.use("Agg"); torch.set_num_threads(1)
    if args.summarize_only:
        cases = json.loads((OUTPUT / "manifest.json").read_text())["cases"]
        records = [json.loads((OUTPUT / "trials" / c["id"] / "metrics.json").read_text()) for c in cases
                   if (OUTPUT / "trials" / c["id"] / "metrics.json").exists()]
    else:
        cases = generate(); records = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, future in enumerate(as_completed([pool.submit(run_trial, (c, args.force)) for c in cases]), 1):
                r = future.result(); records.append(r)
                print(f"[{index}/{len(cases)}] {r['id']}: recovered={r['recovered']}; "
                      f"position={r['position_error_mas']}; {r['seconds']:.1f}s", flush=True)
    summarize(records)


if __name__ == "__main__":
    main()
