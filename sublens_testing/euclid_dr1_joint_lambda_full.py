"""Full-image refit of the frozen conditional joint-inference specification.

The earlier blocked experiment is preserved. All valid VIS pixels are fitted;
previous validation blocks are now in-sample diagnostics, not heldout tests.
No parameter bounds, source prior, stopping budget or PSF is changed according
to official source agreement. Source geometry is rebuilt from all fitted data
using the same initial model, node count, centroidal algorithm and seed.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import euclid_dr1_joint_lambda as experiment

ORIGINAL_OUT=experiment.OUT
experiment.OUT=ORIGINAL_OUT/'full_data'
_original_reader=experiment.read_observation

def full_observation():
    obs,train,test,guard,folder=_original_reader()
    return obs,obs.mask.copy(),test,np.zeros(obs.mask.shape,dtype=bool),folder

experiment.read_observation=full_observation


def record_protocol(steps=3000,burn=1500):
    out=experiment.OUT;root=experiment.ROOT
    protocol=json.loads((out/'protocol.json').read_text())
    protocol.update(full_image_refit=True,all_valid_pixels_fitted=True,heldout_pixels=0,
        test_pixels=0,reference_block_pixels=32,excluded_guard_pixels=0,
        holdout='None: former heldout blocks are now fitted and only in-sample diagnostics',
        validation='No independent validation on full fit; read the preserved parent blocked experiment',
        four_chains_from_four_image_based_profile_fits=True,steps_per_chain=steps,burn_per_chain=burn,
        changed_from_blocked='Only the fitting mask and resulting data-adapted mesh/initializations; identical priors, model, settings and budget',
        official_source_not_used_for_refit_settings=True)
    experiment.save_json(out/'protocol.json',protocol)
    obs,train,test,guard,folder=full_observation()
    files=[folder/'dataset'/f'{folder.name}.fits',folder/'dataset/info.json',folder/'model.fits',
        folder/'dataset/segmentation/artefact_binary.fits',out/'design.npz',out/'initializations.json',
        root/'sublens_testing/euclid_dr1_joint_lambda.py',Path(__file__).resolve()]
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    experiment.save_json(out/'inference_input_sha256.json',{str(f.relative_to(root)):sha(f) for f in files})
    previous={str(f.relative_to(root)):sha(f) for f in experiment.PILOT.rglob('*.fits') if out not in f.parents}
    experiment.save_json(out/'previous_fits_sha256.json',previous)
    (out/'sampler_run_source.py').write_bytes((root/'sublens_testing/euclid_dr1_joint_lambda.py').read_bytes())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','sample','report'])
    parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--burn',type=int,default=1500)
    args=parser.parse_args()
    if args.stage=='prepare':
        experiment.prepare();record_protocol(args.steps,args.burn)
    elif args.stage=='sample':
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(experiment.run_chain,[(i,args.steps,args.burn) for i in range(4)]))
    else:
        import euclid_dr1_joint_lambda_report as report
        report.main()
        summary=json.loads((experiment.OUT/'summary.json').read_text())
        summary.update(full_data_fit=True,reference_blocks_are_heldout=False,
            validation_statement='All valid pixels fitted; no validation scores from this refit. See parent blocked results.')
        experiment.save_json(experiment.OUT/'summary.json',summary)
        with (experiment.OUT/'README.md').open('a') as handle:
            handle.write('\nFULL IMAGE REFIT: all valid pixels fitted. Previously heldout blocks are included in fitting and their scores are in-sample. '
                         'Use the parent folder for the independent-from-source-fit blocked diagnostic (still conditional on foreground). '
                         'Reproduce using euclid_dr1_joint_lambda_full.py prepare / sample / report.\n')

if __name__=='__main__':main()
