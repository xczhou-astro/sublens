"""Identical-data reference / Torch CPU / Torch MPS (or CUDA) pipeline benchmark."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from sublens import LensingObservation,torch_backend
from contextlib import nullcontext
from euclid_dr1_grade_a_scan import OUT as INPUT,SEARCH,write_json

OUTPUT=INPUT/'torch_backend_validation'


def run(device):
    folder=OUTPUT/device.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
    obs=LensingObservation.from_fits(INPUT/'EUCL-DR1-L020080/observation.fits')
    context=nullcontext() if device=='reference' else torch_backend(device)
    with context:
        start=perf_counter()
        result=obs.search_pinn(source_nodes=500,regularization=2e-6,**SEARCH)
        search=perf_counter()-start
        start=perf_counter();result=result.refine_joint(obs.macro_model,maxiter=80,n_starts=2,retry_failed=True)
        joint=perf_counter()-start
    result.save_fits(folder/'result.fits',overwrite=True);result.save_json(folder/'result.json',overwrite=True)
    record=dict(device=device,search_seconds=search,joint_seconds=joint,total_seconds=search+joint,
        baseline_objective=result.baseline.objective,baseline_chi2=float(np.sum(((obs.image-result.baseline.image_model)[obs.mask]/obs.rms[obs.mask])**2)),
        macro_model=result.macro_model,macro_converged=result.macro_converged,
        execution=result.config.get('execution'),pinn_seeds=[list(c.seed) for c in result.candidates],
        candidates=[dict(rank=c.rank,x=c.fit.x,y=c.fit.y,strength=c.fit.strength,delta_chi2=c.fit.delta_chi2,
            penalized_gain=c.fit.delta_objective,converged=c.fit.converged) for c in result.candidates])
    write_json(folder/'benchmark.json',record);print(json.dumps(record),flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--devices',nargs='+',default=['reference','cpu','mps']);args=parser.parse_args()
    torch.set_num_threads(1)
    records=[]
    for device in args.devices:
        records.append(run(device));write_json(OUTPUT/'summary.json',records)
