"""Matched-budget conditional simulation recovery on available Torch devices."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from sublens import LensingObservation,torch_backend
from euclid_dr1_grade_a_scan import write_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sublens_testing/euclid_dr1_grade_a_part1/torch_backend_validation/simulation'

if __name__=='__main__':
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    obs=LensingObservation.from_fits(ROOT/'sublens_testing/one_subhalo/data/observation.fits')
    devices=['reference','cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else [])
    records=[]
    for device in devices:
        context=nullcontext() if device=='reference' else torch_backend(device)
        with context:
            start=perf_counter()
            result=obs.search_pinn(source_nodes=300,regularization=20.,mesh_strategy='centroidal',top_k=5,pinn_steps=90,refine_steps=70)
            result=result.refine_joint(obs.macro_model,maxiter=80,n_starts=2,retry_failed=True)
            seconds=perf_counter()-start
        result.save_fits(OUT/f'{device}.fits',overwrite=True)
        best=result.best_candidate
        # Truth is opened only for evaluation after inference has finished.
        truth=json.loads((ROOT/'sublens_testing/one_subhalo/data/truth.json').read_text())['halo']
        fit=best.fit if best else None
        record=dict(device=device,seconds=seconds,macro_supplied_for_conditional_simulation=True,
            candidates=len(result.candidates),position=None if fit is None else list(fit.position),
            strength=None if fit is None else fit.strength,delta_chi2=None if fit is None else fit.delta_chi2,
            position_error_mas=None if fit is None else float(1000*np.hypot(fit.x-truth['x'],fit.y-truth['y'])),
            relative_strength_error=None if fit is None else fit.strength/truth['strength']-1)
        records.append(record);write_json(OUT/'summary.json',records);print(json.dumps(record),flush=True)
