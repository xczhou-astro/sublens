"""Paired validation of independent gain aperture and physical-fit merging.

Reuse saved noise-scan observations. Truth is consulted only after the public
PINN, joint and adaptive methods have selected their final best candidate.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter

import matplotlib
import numpy as np
import torch
from astropy.io import fits

from sublens import LensingObservation
from run import ROOT, SEARCH, ADAPT, TRUTH, residual_metrics, save_result


OUTPUT = ROOT / "candidate_refinement"
CASES = ("noise_1x_seed_161803", "noise_1x_seed_141421",
         "noise_1x_seed_314159", "noise_1x_seed_271828",
         "noise_32x_seed_161803", "noise_32x_seed_141421")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def run_case(task):
    name, force = task
    matplotlib.use("Agg")
    torch.set_num_threads(1)
    old = ROOT / "noise_scan" / "trials" / name
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists() and not force:
        return json.loads((folder / "metrics.json").read_text())
    started = perf_counter()
    shutil.copy2(old / "observation.fits", folder / "observation.fits")
    observation = LensingObservation.from_fits(folder / "observation.fits")
    search = {**SEARCH, "separation": 2*observation.pixel_scale, "gain_radius": .22}
    initial = observation.search_pinn(**search)
    initial.save_json(folder / "initial_search.json", overwrite=True)
    # The neural field itself is unchanged: only its peak selection differs.
    with fits.open(old / "result.fits") as hdus:
        np.testing.assert_array_equal(initial.residual_potential.convergence, hdus["PINN_KAPPA"].data)
    joint = initial.refine_joint(observation.macro_model, maxiter=120, n_starts=2, retry_failed=True)
    joint.save_json(folder / "initial_joint.json", overwrite=True)
    refined = joint.refine_adaptive(observation.macro_model, **ADAPT, verbose=True)
    refined.save_json(folder / "unmerged_result.json", overwrite=True)
    refined.save_fits(folder / "unmerged_result.fits", overwrite=True)
    result = refined.merge_candidates(radius=observation.pixel_scale)
    assert result.best_candidate is refined.best_candidate
    save_result(result, folder)
    best = result.best_candidate.fit
    # Evaluation starts here; neither the truth nor the old candidate is a fit start.
    truth = TRUTH["halo"]
    position_error = float(1000*np.linalg.norm(np.asarray(best.position)-[truth["x"], truth["y"]]))
    strength_error = float(best.strength/truth["strength"]-1)
    previous = json.loads((old / "evaluation.json").read_text())
    record = dict(case=name, previous=dict(
        recovered=previous["recovered"], position_error_mas=previous["position_error_mas"],
        strength_error_fraction=previous["strength_error_fraction"],
        chi2_per_pixel=previous["inference"]["with_subhalo"]["chi2_per_pixel"]),
        position_arcsec=list(best.position), strength_arcsec=best.strength,
        position_error_mas=position_error, strength_error_fraction=strength_error,
        converged=bool(best.converged and result.macro_converged),
        recovered=bool(best.converged and result.macro_converged and position_error <= 50
                       and abs(strength_error) <= .30),
        macro=residual_metrics(observation, result.baseline.image_model),
        with_subhalo=residual_metrics(observation, best.image_model),
        proposed_candidates=len(initial.candidates), merged_candidates=len(result.candidates),
        groups=result.config["candidate_merge"]["groups"],
        same_pinn_field_as_original=True, search=search,
        seconds=perf_counter()-started,
        input_sha256=hashlib.sha256((folder / "observation.fits").read_bytes()).hexdigest())
    write_json(folder / "metrics.json", record)
    return record


def summarize(records):
    import matplotlib.pyplot as plt

    previous_scan = json.loads((ROOT / "noise_scan" / "summary.json").read_text())
    factors = previous_scan["config"]["factors"]
    seeds = previous_scan["config"]["noise_seeds"]
    def coordinates(name):
        parts = name.split("_")
        return int(parts[3]), float(parts[1][:-1])
    records = sorted(records, key=lambda r:(seeds.index(coordinates(r["case"])[0]), coordinates(r["case"])[1]))
    summary = dict(
        cases=records, adaptive=ADAPT,
        changes=["Independent 0.22 arcsec gain radius", "Two native pixels seed separation",
                 "Merge only after final physical, macro and source refinement"],
        fixed=["Same saved observations and neural fields", "Same five proposal budget",
               "Same source selection, physical bounds, optimizer starts and iteration budgets"],
        truth_used_for_generation_and_evaluation_only=True,
    )
    write_json(OUTPUT / "summary.json", summary)
    if len(records) == len(seeds)*len(factors):
        write_json(OUTPUT / "matrix_summary.json", summary)
    lines = ["# Competing PINN candidates: paired validation\n",
             "同一输入、同一 PINN 场、同一 5 个提名预算、同一源选择和优化预算。"
             "评分半径保持 0.22 arcsec；初始去重间距从 0.44 改为两个 native pixels（本例 0.10 arcsec）。"
             "各峰独立完成源、macro 与子晕精修后，以一个像素内的位置和 25% 内的强度判定重复解。"
             "只合并强度为正且收敛的解，保留 gain 最大者。原扫描不覆盖。\n",
             "真值不用于提名或精修；macro 形状、斜率和中心仍采用模拟输入值。"
             "恢复要求收敛、位置误差 <=50 mas、强度误差 <=30%。这是定向修复复测，不是独立完备率或假阳性率校准。\n",
             "| Case | Previous position [mas] | New position [mas] | New strength error | χ²/Npix old → new | Recovery old → new | Groups / seeds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        old = r["previous"]
        lines.append(f"| {r['case']} | {old['position_error_mas']:.2f} | {r['position_error_mas']:.2f} | "
                     f"{100*r['strength_error_fraction']:+.2f}% | {old['chi2_per_pixel']:.3f} → "
                     f"{r['with_subhalo']['chi2_per_pixel']:.3f} | {old['recovered']} → {r['recovered']} | "
                     f"{r['merged_candidates']}/{r['proposed_candidates']} |")
    lines += ["\n每个 case 包含输入 FITS、精修前后 JSON、未合并和合并后的结果 FITS/JSON、源重构、候选和两种残差图。"
              "候选仍是分别拟合的单子晕假设，合并后的组数不代表检出了几个子晕。\n",
              "API：\n```python\nresult = observation.search_pinn(top_k=5, gain_radius=.22)\n"
              "result = result.refine_joint(observation.macro_model, n_starts=2, retry_failed=True, maxiter=120)\n"
              "result = result.refine_adaptive(maxiter=120)\n"
              "unique = result.merge_candidates(radius=observation.pixel_scale)\n```\n",
              "默认 separation=None 解析为两个 native pixels；显式 separation=.44、gain_radius=.22 可复现旧提名规则。"
              "历史 noise_scan.py 已固定这两个旧值，以保留其原始协议。"
              "精修结果合并只是一种最终汇总，不进行多子晕联合拟合或存在性判定。\n"]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    if len(records) == len(seeds)*len(factors):
        plot_matrix(records, previous_scan)
    # Compare the two originally failing cases on the same normalized colour scale.
    shown = [r for r in records if r["case"] in CASES[:2]]
    if shown:
        fig, axes = plt.subplots(len(shown), 2, figsize=(9, 4*len(shown)), squeeze=False, layout="constrained")
        for row, record in enumerate(shown):
            for col, (label, path) in enumerate((
                    ("Previous", ROOT / "noise_scan" / "trials" / record["case"] / "result.fits"),
                    ("Competing candidates", OUTPUT / record["case"] / "result.fits"))):
                with fits.open(path) as hdus:
                    mask = hdus["MASK"].data.astype(bool)
                    residual = hdus["RESIDUAL"].data/hdus["RMS"].data
                chi2 = float(np.mean(residual[mask]**2))
                artist = axes[row,col].imshow(np.where(mask,residual,np.nan),origin="lower",
                    extent=(-2.5,2.5,-2.5,2.5),cmap="RdBu_r",vmin=-4,vmax=4)
                axes[row,col].set(title=f"{record['case']}\n{label}; χ²/Npix={chi2:.3f}",
                                  xlabel="θx [arcsec]",ylabel="θy [arcsec]")
        fig.colorbar(artist,ax=axes,shrink=.7,label="Observed − model / RMS")
        fig.savefig(OUTPUT / "failed_cases_residuals.png",dpi=175,bbox_inches="tight")
        plt.close(fig)


def plot_matrix(records, previous_scan):
    """Match the original paired matrix using newly fitted cells only."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    factors = previous_scan["config"]["factors"]
    seeds = previous_scan["config"]["noise_seeds"]
    lookup = {r["case"]:r for r in records}
    shape = (len(seeds),len(factors))
    success = np.zeros(shape,dtype=bool)
    position = np.empty(shape)
    strength = np.empty(shape)
    converged = np.zeros(shape,dtype=bool)
    cells = []
    fig,ax = plt.subplots(figsize=(10,4.4),layout="constrained")
    for iy,seed in enumerate(seeds):
        for ix,factor in enumerate(factors):
            name = f"noise_{factor:g}x_seed_{seed}"
            row = lookup[name]
            position[iy,ix] = row["position_error_mas"]
            strength[iy,ix] = row["strength_error_fraction"]
            converged[iy,ix] = row["converged"]
            success[iy,ix] = row["recovered"]
            expected = bool(row["converged"] and position[iy,ix] <= 50 and abs(strength[iy,ix]) <= .30)
            assert expected == row["recovered"]
            ax.text(ix,iy,f"{position[iy,ix]:.0f} mas\n{100*strength[iy,ix]:+.0f}%",
                    ha="center",va="center",fontsize=10)
            cells.append(dict(case=name,seed=seed,noise_factor=factor,noise_rms=.012*factor,
                              position_error_mas=row["position_error_mas"],
                              strength_error_fraction=row["strength_error_fraction"],
                              converged=row["converged"],recovered=row["recovered"],
                              previous_recovered=row["previous"]["recovered"]))
    ax.imshow(success.astype(int),cmap=ListedColormap(["#f4b79c","#a9d7c4"]),vmin=0,vmax=1,aspect="auto")
    ax.set(xticks=range(len(factors)),xticklabels=[f"{f:g}x\nRMS={.012*f:g}" for f in factors],
           yticks=range(len(seeds)),yticklabels=seeds,
           xlabel="Noise level (same field scaled within each row)",ylabel="Independent noise seed",
           title="Updated candidates: converged, position <=50 mas and strength error <=30%")
    fig.savefig(OUTPUT / "paired_recovery.png",dpi=180,bbox_inches="tight")
    fig.savefig(OUTPUT / "paired_recovery.pdf",bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(OUTPUT / "recovery_matrix.npz",seeds=np.array(seeds),noise_factors=np.array(factors),
                        recovered=success,position_error_mas=position,strength_error_fraction=strength,
                        converged=converged)
    groups = [dict(factor=f,noise_rms=.012*f,recovered=int(success[:,i].sum()),trials=len(seeds),
                   previous_recovered=sum(c["previous_recovered"] for c in cells if c["noise_factor"] == f))
              for i,f in enumerate(factors)]
    write_json(OUTPUT / "matrix_data.json",dict(seeds=seeds,noise_factors=factors,groups=groups,cells=cells,
        improved_cases=[c["case"] for c in cells if c["recovered"] and not c["previous_recovered"]],
        regressed_cases=[c["case"] for c in cells if c["previous_recovered"] and not c["recovered"]],
        success_rule="macro and halo converged; position <=50 mas; abs(strength error) <=30%"))
    with (OUTPUT / "README.md").open("a") as stream:
        stream.write("\n完整矩阵：paired_recovery.png / paired_recovery.pdf。"
                     "每格均来自新版流程的拟合，6 个先前复测格使用已验证缓存，其余 26 格重新计算。"
                     "matrix_data.json 和 recovery_matrix.npz 保存矩阵数值；原 noise_scan/ 矩阵保留。\n"
                     "\n复现全部 32 格：`.venv/bin/python sublens_testing/one_subhalo/test_competing_candidates.py --matrix --workers 3`。"
                     "仅根据已有结果重新汇总绘图：增加 `--summarize-only`。每档仅 4 个噪声场，属于当前固定场景的恢复试验。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--checks",action="store_true",help="Also rerun two good and two high-noise controls")
    parser.add_argument("--matrix",action="store_true",help="Complete all 32 cells of the original noise matrix")
    parser.add_argument("--summarize-only",action="store_true")
    parser.add_argument("--force",action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    matplotlib.use("Agg")
    OUTPUT.mkdir(exist_ok=True)
    if args.matrix:
        original = json.loads((ROOT / "noise_scan" / "summary.json").read_text())
        names = tuple(r["id"] for r in original["trials"])
        assert len(set(names)) == len(names) == 32
        # Preserve the earlier six-case report while extending its result set.
        for name,destination in (("summary.json","targeted_summary.json"),
                                 ("protocol.json","targeted_protocol.json"),("README.md","TARGETED_README.md")):
            if (OUTPUT / name).exists() and not (OUTPUT / destination).exists():
                shutil.copy2(OUTPUT / name,OUTPUT / destination)
    else:
        names = CASES if args.checks else CASES[:2]
    write_json(OUTPUT / "protocol.json",dict(cases=list(names),search={**SEARCH,"gain_radius":.22,
               "separation":"two native pixels"},adaptive=ADAPT,merge_radius="one native pixel",
               merge_strength_tolerance=.25,proposal_budget=5))
    if args.matrix:
        shutil.copy2(OUTPUT / "protocol.json",OUTPUT / "matrix_protocol.json")
    records=[]
    if args.summarize_only:
        records=[json.loads((OUTPUT / name / "metrics.json").read_text()) for name in names]
        summarize(records)
        return
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run_case,(name,args.force)) for name in names]):
            record=future.result()
            records.append(record)
            print(f"{record['case']}: position={record['position_error_mas']:.2f} mas, "
                  f"strength error={100*record['strength_error_fraction']:+.2f}%, "
                  f"chi2/pixel={record['with_subhalo']['chi2_per_pixel']:.4f}",flush=True)
    summarize(records)


if __name__ == "__main__":
    main()
