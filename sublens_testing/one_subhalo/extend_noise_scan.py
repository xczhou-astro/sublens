"""Extend the fixed noise scan to bracket its high-noise failure endpoint.

Run alongside or after noise_scan.py. This only adds observations and fits with
identical settings; it does not change the original protocol or recovery rule.
"""

import argparse
from dataclasses import replace
import json
from time import perf_counter

from astropy.io import fits
import matplotlib
import numpy as np
import torch

from sublens import LensingObservation
from noise_scan import OUTPUT, BASE_ROOT, TRUTH, SEEDS, run_trial, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    matplotlib.use("Agg")
    torch.set_num_threads(1)
    base = LensingObservation.from_fits(BASE_ROOT / "data" / "observation.fits")
    clean = fits.getdata(OUTPUT / "reference" / "clean_with_subhalo.fits")
    signal = json.loads((OUTPUT / "reference" / "signal_metrics.json").read_text())
    cases = []
    for seed in SEEDS:
        field = np.random.default_rng(seed).standard_normal(base.image.shape)
        for factor in (64., 128.):
            sigma = TRUTH["noise_rms"]*factor
            case = dict(id=f"noise_{factor:g}x_seed_{seed}", factor=factor, noise_rms=sigma,
                        seed=seed, rho_image=signal["signal_l2"]/sigma)
            folder = OUTPUT / "trials" / case["id"]
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "observation.fits").exists():
                replace(base, image=clean+sigma*field, rms=np.full(base.image.shape, sigma),
                        lens_id=case["id"]).to_fits(folder / "observation.fits")
            cases.append(case)
    write_json(OUTPUT / "extension_manifest.json", dict(
        factors=[64., 128.], cases=cases,
        reason="First completed noise field still recovered at 32x; extend range to bracket failure",
        same_inference_settings_and_recovery_rule=True,
    ))
    started = perf_counter()
    for index, case in enumerate(cases, 1):
        record = run_trial((case, args.force))
        print(f"[{index}/{len(cases)}] {case['id']}: {record['seconds']:.1f}s", flush=True)
    write_json(OUTPUT / "extension_runtime.json", dict(seconds=perf_counter()-started, workers=1))


if __name__ == "__main__":
    main()
