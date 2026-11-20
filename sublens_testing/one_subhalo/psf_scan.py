"""Controlled FWHM/profile scan with matched, pixel-integrated PSF kernels."""

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

from sublens import LensingObservation,LensingSimulator,MacroLensModel,sersic_profile,truncated_subhalo_deflection,build_psf_kernel
from sublens.inversion.potential import _arc_masks,_psf_guard_radius
from run import ROOT,TRUTH,SEARCH,ADAPT,residual_metrics,save_result


OUTPUT = ROOT / "psf_scan"
PROFILES = ("gaussian","moffat","diffraction")
FWHMS = (.09,.18,.36,.72)
SEEDS = (314159,271828,161803,141421)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    OUTPUT.mkdir(exist_ok=True)
    reference=OUTPUT/"reference"
    reference.mkdir(exist_ok=True)
    base=LensingObservation.from_fits(ROOT/"data"/"observation.fits")
    p=TRUTH
    sim=LensingSimulator(N_img=p["image_size"],pixel_scale=p["pixel_scale"],N_src=p["source_size"],
                         L_src=p["source_half_width"],psf_kernel=np.ones((1,1)),device="cpu")
    with torch.no_grad():
        y,x=torch.meshgrid(sim.src_coords_1d,sim.src_coords_1d,indexing="ij")
        source=sersic_profile(x,y,**p["source"])
        macro,_=MacroLensModel(**p["macro"])(sim.theta_grid)
        tx,ty=sim.theta_grid.unbind(dim=1)
        ax,ay=truncated_subhalo_deflection(tx,ty,p["halo"]["x"],p["halo"]["y"],p["halo"]["strength"])
        native0=sim.render(source[None,None],sim.theta_grid-macro).numpy().squeeze().astype(float)
        native1=sim.render(source[None,None],sim.theta_grid-macro-torch.stack((ax,ay),dim=-1)).numpy().squeeze().astype(float)
    image_header=fits.getheader(ROOT/"data"/"clean_image.fits")
    for name,image in (("native_macro",native0),("native_with_halo",native1)):
        if not (reference/f"{name}.fits").exists():
            fits.PrimaryHDU(image,header=image_header).writeto(reference/f"{name}.fits",checksum=True)
    config=dict(profiles=list(PROFILES),fwhm_arcsec=list(FWHMS),noise_seeds=list(SEEDS),noise_rms=.012,
                pixel_scale_arcsec=.05,search=SEARCH,adaptive=ADAPT,gain_radius_arcsec=.22,
                minimum_seed_separation_pixels=2,psf_guard_cap_fwhm=1.5,
                psf_known_exactly=True,flux_and_noise_fixed=True,paired_noise_across_all_psfs=True,
                moffat_beta=3.5,pupil=dict(obscuration=.3,spider_width=.015,spider_angle_rad=.35),
                psf_pixel_quadrature=7,psf_stamp_fraction=.995,
                recovery=dict(position_error_mas=50.,absolute_strength_error_fraction=.30,converged=True),
                macro_shape_slope_center_fixed_at_simulation_values=True)
    if (OUTPUT/"config.json").exists() and json.loads((OUTPUT/"config.json").read_text())!=json.loads(json.dumps(config)):
        raise ValueError("Existing PSF scan configuration differs; preserve the old experiment")
    write_json(OUTPUT/"config.json",config)
    write_json(reference/"truth.json",p)
    cases=[]
    profiles=[]
    for kind in PROFILES:
        for fwhm in FWHMS:
            key=f"{kind}_fwhm_{fwhm:.3f}"
            psf_path=OUTPUT/"psfs"/f"{key}.fits"
            metadata_path=psf_path.with_suffix(".json")
            psf_path.parent.mkdir(exist_ok=True)
            if psf_path.exists():
                kernel=fits.getdata(psf_path).astype(float)
                metadata=json.loads(metadata_path.read_text())
            else:
                kernel,metadata=build_psf_kernel(fwhm,.05,profile=kind,return_metadata=True)
                header=fits.Header(dict(PROFILE=kind,FWHMTAR=fwhm,FWHMX=metadata["fwhm_x_arcsec"],
                    FWHMY=metadata["fwhm_y_arcsec"],PIXSCALE=.05,FLUXKEEP=metadata["estimated_flux_before_stamp_renormalization"]))
                for axis in (1,2):
                    header[f"CTYPE{axis}"]="LINEAR";header[f"CUNIT{axis}"]="arcsec"
                    header[f"CRPIX{axis}"]=(kernel.shape[axis-1]+1)/2
                    header[f"CRVAL{axis}"]=0.;header[f"CDELT{axis}"]=.05
                fits.PrimaryHDU(kernel,header=header).writeto(psf_path,checksum=True)
                write_json(metadata_path,metadata)
            clean0=fftconvolve(native0,kernel,mode="same")
            clean1=fftconvolve(native1,kernel,mode="same")
            for suffix,image in (("macro",clean0),("halo",clean1)):
                path=reference/f"{key}_clean_{suffix}.fits"
                if not path.exists():fits.PrimaryHDU(image,header=image_header).writeto(path,checksum=True)
            audit=replace(base,psf=kernel,image=clean1,rms=np.full(base.image.shape,.012))
            x,y=(a.mean(axis=-1) for a in audit.image_rays())
            def coverage(cap):
                folds=list(_arc_masks(audit,x,y,cap))
                return dict(folds=len(folds),pixels=int(np.logical_or.reduce([a[2] for a in folds]).sum()) if folds else 0)
            strict=coverage(None)
            capped=coverage(1.5*fwhm)
            py,px=np.indices(kernel.shape)
            r=.05*np.hypot(px-kernel.shape[1]//2,py-kernel.shape[0]//2)
            audit_record=dict(profile=kind,fwhm_arcsec=fwhm,psf_metadata=metadata,
                              strict_guard_arcsec=.05*_psf_guard_radius(kernel),used_guard_arcsec=min(.05*_psf_guard_radius(kernel),1.5*fwhm),
                              strict_cross_fit=strict,capped_cross_fit=capped,
                              psf_flux_outside_guard_cap=float(kernel[r>1.5*fwhm].sum()),
                              rho_image=float(np.linalg.norm((clean1-clean0)[base.mask])/.012))
            profiles.append(audit_record)
            for seed in SEEDS:
                case=dict(id=f"{key}_seed_{seed}",profile=kind,fwhm_arcsec=fwhm,noise_seed=seed,
                          psf_key=key,rho_image=audit_record["rho_image"],coverage_audit=audit_record)
                folder=OUTPUT/"trials"/case["id"]
                folder.mkdir(parents=True,exist_ok=True)
                if not (folder/"observation.fits").exists():
                    z=np.random.default_rng(seed).standard_normal(base.image.shape)
                    obs=replace(audit,image=clean1+.012*z,lens_id=case["id"])
                    obs.to_fits(folder/"observation.fits")
                cases.append(case)
            print(f"Prepared {key}: kernel={kernel.shape}, strict/capped cross-fit pixels="
                  f"{strict['pixels']}/{capped['pixels']}, rho={audit_record['rho_image']:.1f}",flush=True)
    write_json(OUTPUT/"manifest.json",dict(config=config,cases=cases,profiles=profiles))
    plot_psfs(profiles)
    return cases


def plot_psfs(profiles):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig,axes=plt.subplots(3,4,figsize=(13,9),layout="constrained")
    for row,kind in enumerate(PROFILES):
        for col,fwhm in enumerate(FWHMS):
            psf=fits.getdata(OUTPUT/"psfs"/f"{kind}_fwhm_{fwhm:.3f}.fits")
            cy,cx=np.array(psf.shape)//2
            half=max(4,int(np.ceil(1.25*fwhm/.05)))
            stamp=psf[cy-half:cy+half+1,cx-half:cx+half+1] if cy>=half else psf
            limit=(stamp.shape[0]/2)*.05
            artist=axes[row,col].imshow(stamp/psf.max(),origin="lower",extent=(-limit,limit,-limit,limit),
                                         cmap="magma",norm=LogNorm(vmin=1e-5,vmax=1))
            axes[row,col].set(title=f"{kind}; FWHM={fwhm:g}″",xlabel="Offset [arcsec]",ylabel="Offset [arcsec]")
            axes[row,col].set(xlim=(-1.25*fwhm,1.25*fwhm),ylim=(-1.25*fwhm,1.25*fwhm))
    fig.colorbar(artist,ax=axes,shrink=.6,label="PSF / peak (log scale)")
    fig.savefig(OUTPUT/"psf_families.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,kind in zip(axes,PROFILES):
        fwhm=.36
        kernel=fits.getdata(OUTPUT/"psfs"/f"{kind}_fwhm_{fwhm:.3f}.fits")
        cy,cx=np.array(kernel.shape)//2
        half=min(cy,int(np.ceil(6*fwhm/.05)))
        stamp=kernel[cy-half:cy+half+1,cx-half:cx+half+1]
        limit=(half+.5)*.05/fwhm
        artist=ax.imshow(stamp/kernel.max(),origin="lower",extent=(-limit,limit,-limit,limit),
                         cmap="magma",norm=LogNorm(vmin=1e-6,vmax=1))
        ax.set_facecolor(plt.get_cmap("magma")(0))
        ax.set(title=kind,xlabel="Offset / FWHM",ylabel="Offset / FWHM",xlim=(-6,6),ylim=(-6,6))
    fig.colorbar(artist,ax=axes,shrink=.75,label="PSF / peak")
    fig.suptitle("PSF cores and wings; FWHM=0.36″; identical normalization")
    fig.savefig(OUTPUT/"psf_wings.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for kind in PROFILES:
        rows=[r for r in profiles if r["profile"]==kind]
        axes[0].plot([r["fwhm_arcsec"] for r in rows],[r["rho_image"] for r in rows],"o-",label=kind)
        kernel=fits.getdata(OUTPUT/"psfs"/f"{kind}_fwhm_0.720.fits")
        yy,xx=np.indices(kernel.shape)
        radius=.05*np.hypot(xx-kernel.shape[1]//2,yy-kernel.shape[0]//2)
        bins=np.ceil(radius/.05).astype(int)
        energy=np.cumsum(np.bincount(bins.ravel(),weights=kernel.ravel()))
        axes[1].plot(np.arange(len(energy))*.05/.72,energy,label=kind)
    axes[0].set(xlabel="Core FWHM [arcsec]",ylabel="Injected image perturbation SNR",xscale="log",yscale="log")
    axes[1].set(xlabel="Radius / core FWHM",ylabel="Encircled PSF energy",xscale="log",xlim=(.1,100),ylim=(0,1.01))
    axes[1].axhline(.98,color="gray",linestyle="--",label="Original guard energy")
    axes[1].axvline(1.5,color="gray",linestyle=":",label="Guard cap used")
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUTPUT/"psf_signal_and_energy.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


def run_trial(task):
    case,force=task
    torch.set_num_threads(1);matplotlib.use("Agg")
    folder=OUTPUT/"trials"/case["id"]
    if (folder/"metrics.json").exists() and not force:return json.loads((folder/"metrics.json").read_text())
    started=perf_counter()
    obs=LensingObservation.from_fits(folder/"observation.fits")
    result=obs.search_pinn(**SEARCH,gain_radius=.22,psf_guard_max_radius=1.5*case["fwhm_arcsec"])
    result.save_json(folder/"initial_search.json",overwrite=True)
    result=result.refine_joint(obs.macro_model,maxiter=120,n_starts=2,retry_failed=True)
    result.save_json(folder/"initial_joint.json",overwrite=True)
    if result.candidates:
        result=result.refine_adaptive(obs.macro_model,**ADAPT)
    result=result.merge_candidates(radius=obs.pixel_scale)
    save_result(result,folder)
    best=result.best_candidate
    metrics=dict(**case,seconds=perf_counter()-started,proposals=len(result.residual_potential.peaks),
                 unique_hypotheses=len(result.candidates),pinn_scoring_pixels=int(result.residual_potential.mask.sum()),
                 macro=residual_metrics(obs,result.baseline.image_model),recovered=False,
                 macro_converged=result.macro_converged,halo_converged=None,position_error_mas=None,
                 strength_error_fraction=None,position_arcsec=None,strength_arcsec=None,
                 observation_sha256=digest(folder/"observation.fits"))
    if best is not None:
        fit=best.fit
        truth=TRUTH["halo"]
        error=float(1000*np.linalg.norm(np.asarray(fit.position)-[truth["x"],truth["y"]]))
        strength_error=float(fit.strength/truth["strength"]-1)
        metrics.update(position_arcsec=list(fit.position),strength_arcsec=fit.strength,
                       position_error_mas=error,strength_error_fraction=strength_error,
                       halo_converged=fit.converged,with_subhalo=residual_metrics(obs,fit.image_model),
                       recovered=bool(fit.converged and result.macro_converged and error<=50 and abs(strength_error)<=.30))
    write_json(folder/"metrics.json",metrics)
    return metrics


def summarize(records, selected_seeds=SEEDS):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    selected_seeds=tuple(selected_seeds)
    records=sorted(records,key=lambda r:(PROFILES.index(r["profile"]),r["fwhm_arcsec"],selected_seeds.index(r["noise_seed"])))
    planned=len(PROFILES)*len(FWHMS)*len(selected_seeds)
    groups=[]
    for kind in PROFILES:
        for fwhm in FWHMS:
            rows=[r for r in records if r["profile"]==kind and r["fwhm_arcsec"]==fwhm]
            if rows:
                positioned=[r for r in rows if r["position_error_mas"] is not None]
                groups.append(dict(profile=kind,fwhm_arcsec=fwhm,trials=len(rows),recovered=sum(r["recovered"] for r in rows),
                                   no_coverage=sum(r["pinn_scoring_pixels"]==0 for r in rows),
                                   median_position_mas=None if not positioned else float(np.median([r["position_error_mas"] for r in positioned])),
                                   rho_image=rows[0]["rho_image"],coverage=rows[0]["coverage_audit"]))
    write_json(OUTPUT/"summary.json",dict(groups=groups,trials=records,completed=len(records),planned=planned,
               noise_seeds=list(selected_seeds),selected_case_ids=[f"{kind}_fwhm_{fwhm:.3f}_seed_{seed}"
                   for kind in PROFILES for fwhm in FWHMS for seed in selected_seeds]))
    lookup={r["id"]:r for r in records}
    if len(selected_seeds)==1:
        fig,ax=plt.subplots(figsize=(10,4.6),layout="constrained")
        panels=[(ax,[(kind,selected_seeds[0]) for kind in PROFILES],list(PROFILES),"PSF profile",f"Noise seed {selected_seeds[0]}")]
    else:
        fig,axes=plt.subplots(3,1,figsize=(10,3+1.75*len(selected_seeds)),layout="constrained")
        panels=[(ax,[(kind,seed) for seed in selected_seeds],list(selected_seeds),"Noise seed",kind)
                for ax,kind in zip(axes,PROFILES)]
    for ax,row_keys,row_labels,ylabel,title in panels:
        matrix=np.full((len(row_keys),len(FWHMS)),np.nan)
        for iy,(kind,seed) in enumerate(row_keys):
            for ix,fwhm in enumerate(FWHMS):
                row=lookup.get(f"{kind}_fwhm_{fwhm:.3f}_seed_{seed}")
                if row is None:continue
                matrix[iy,ix]=2 if row["pinn_scoring_pixels"]==0 else int(row["recovered"])
                label="No coverage" if row["pinn_scoring_pixels"]==0 else "No proposal" if row["position_error_mas"] is None else f"{row['position_error_mas']:.1f} mas\n{100*row['strength_error_fraction']:+.1f}%"
                ax.text(ix,iy,label,ha="center",va="center",fontsize=10)
        ax.imshow(matrix,cmap=ListedColormap(["#f4b79c","#a9d7c4","#d0d0d0"]),vmin=0,vmax=2,aspect="auto")
        ax.set(xticks=range(4),xticklabels=[f"{f:g}″" for f in FWHMS],yticks=range(len(row_labels)),yticklabels=row_labels,
               xlabel="PSF core FWHM [arcsec]",ylabel=ylabel,title=title)
    fig.suptitle("Fixed halo 0.05″; RMS=0.012; pixel=0.05″\nGreen: converged, position ≤50 mas and strength error ≤30%; gray: no cross-fit coverage")
    fig.savefig(OUTPUT/"psf_recovery_matrix.png",dpi=175,bbox_inches="tight")
    fig.savefig(OUTPUT/"psf_recovery_matrix.pdf",bbox_inches="tight")
    plt.close(fig)
    text=["# PSF width and shape recovery experiment\n",
          f"Completed {len(records)} / {planned} selected trials; noise seeds: {list(selected_seeds)}. "
          "Other generated/partially computed seeds are retained in trials/ but excluded from this report.\n",
          "固定一个 b=0.05 arcsec 子晕，位置 (−1.28,0.61) arcsec。各模拟共享相同的注入源与 macro、像素 0.05 arcsec、RMS=0.012、拟合 mask。"
          "同一 Gaussian 噪声场在全部 PSF 条件下配对使用。总源通量固定，不重新归一化图像峰值。\n",
          "Gaussian 为简单对照；Moffat β=3.5 描述较强翼部；diffraction 为单色环形 pupil 加四根支撑杆的理想衍射模型。"
          "空间模型不是 HST/JWST/Euclid 的标定 PSF。三种形状在共同 FWHM 网格下缩放，是受控形状比较，不是不同望远镜的完整性能预测。\n",
          "PSF 包含 7×7 子像素积分，调整光学宽度使 detector-response 中心切面的半高宽达到目标。"
          "记录 x/y FWHM、包围能量和有限核归一化前的通量。模拟与拟合使用相同完整已知 PSF；本轮不包含 PSF 估计误差。\n",
          "rho_image = sqrt(sum_mask[(clean_halo - clean_macro)^2 / RMS^2])，"
          "是固定真值源与 macro 时的图像扰动信噪比，未扣除重新拟合源/macro 可以吸收的部分，不是检测显著性。\n",
          "长翼使默认 98% 包围能量隔离半径过大。本轮对所有形状统一使用 1.5×FWHM 的隔离上限，"
          "完整 PSF（包含翼部）仍用于前向拟合；summary.json 同时记录严格规则和上限规则的覆盖率及上限外 PSF 通量。"
          "上限避免规则把弧段全部排除，但不宣称消除了弧段之间的物理扰动泄漏；存在性显著性需按该完整流程重新校准。\n",
          "| Profile | FWHM [arcsec] | Recovered / trials | Median position [mas] | rho_image | Strict/capped pixels |",
          "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        c=g["coverage"]
        median="—" if g["median_position_mas"] is None else f"{g['median_position_mas']:.1f}"
        text.append(f"| {g['profile']} | {g['fwhm_arcsec']:g} | {g['recovered']}/{g['trials']} | {median} | {g['rho_image']:.1f} | {c['strict_cross_fit']['pixels']}/{c['capped_cross_fit']['pixels']} |")
    text += ["\n恢复标准为 macro 与 halo 收敛、位置误差 ≤50 mas、强度误差 ≤30%。"
             "候选独立拟合、联合精修、源网格选择和最终重复解合并沿用当前 pipeline。"
             "拟合时源亮度重新求解，并联合优化 theta_E 与两个 shear 分量；macro 形状、斜率和中心固定在模拟输入值。"
             "子晕真值只用于生成与最后评估。"
             "当前选择的噪声场仅供探索，单个 seed 不能估计恢复概率或普适 FWHM 阈值。"
             "中位数包含失败样本；无覆盖不等于观测在物理上没有信息。\n",
             "数据：psfs/ 为可直接读入的 PSF FITS；reference/ 为原始无 PSF 图像、干净卷积图和真值；"
             "trials/ 为输入 image/RMS/PSF/mask FITS、拟合 JSON/FITS、重构和残差图。\n",
             "复现：`.venv/bin/python sublens_testing/one_subhalo/psf_scan.py --workers 3 --seeds "
             +" ".join(str(s) for s in selected_seeds)+"`。"
             "不指定 --seeds 时可恢复四个 seed 的完整扫描，已有结果直接复用。"
             "`--summarize-only` 仅重画，`--force` 重拟合。\n",
             "背景：[ESO 的 Moffat/seeing 讨论](https://doi.org/10.1051/0004-6361/201833716)，"
             "[STScI ACS PSF 与衍射、像素响应](https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-6-acs-point-spread-functions)。\n"]
    (OUTPUT/"README.md").write_text("\n".join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    parser.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=list(SEEDS),
                        help="Fit/report selected paired noise seeds; default: all four")
    args=parser.parse_args()
    selected_seeds=tuple(dict.fromkeys(args.seeds))
    matplotlib.use("Agg");torch.set_num_threads(1)
    if args.summarize_only:
        cases=json.loads((OUTPUT/"manifest.json").read_text())["cases"]
        records=[json.loads((OUTPUT/"trials"/c["id"]/"metrics.json").read_text()) for c in cases
                 if c["noise_seed"] in selected_seeds and (OUTPUT/"trials"/c["id"]/"metrics.json").exists()]
    else:
        cases=generate();records=[]
        ordered=sorted([c for c in cases if c["noise_seed"] in selected_seeds],
                       key=lambda c:(selected_seeds.index(c["noise_seed"]),FWHMS.index(c["fwhm_arcsec"]),PROFILES.index(c["profile"])))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for index,future in enumerate(as_completed([pool.submit(run_trial,(c,args.force)) for c in ordered]),1):
                row=future.result();records.append(row)
                print(f"[{index}/{len(ordered)}] {row['id']}: recovered={row['recovered']}; position={row['position_error_mas']}; {row['seconds']:.1f}s",flush=True)
    summarize(records,selected_seeds)


if __name__=="__main__":main()
