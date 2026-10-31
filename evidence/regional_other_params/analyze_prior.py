"""Phase 3B analysis: PSIS from the reference fit versus two independent full refits (A, B) per scenario,
for a prior change on one non-ROI field. Same estimators and block-bootstrap SEs as evidence/regional_roi/analyze_noise.py.

Usage: python -I evidence/regional_other_params/analyze_prior.py --ref <reference run> --refits_a runs/phase3b_refits_A \
           --refits_b runs/phase3b_refits_B --out evidence/regional_other_params/results
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "regional_other_params", "regional_roi"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np
import pandas as pd
import xarray as xr

import meridian.backend  # noqa: F401  (JAX x64)
from prior_reweight import PriorFactor, reweight, relative_efficiency
from scenarios_other import REFERENCE_PRIORS, SCENARIOS_3B, make
from analyze_noise import STATS, boot_se, wsummary


def load(d, var):
    x = np.asarray(xr.open_dataset(os.path.join(d, "posterior.nc"))[var])
    if x.ndim == 2:  # scalar variable such as sigma: (chain, draw) -> one "channel"
        x = x[..., None]
    return x, json.load(
        open(os.path.join(d, "run_metadata.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--refits_a", required=True)
    ap.add_argument("--refits_b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override_a", action="append", default=[], metavar="SCENARIO=DIR",
                    help="use this run directory as refit A for SCENARIO (e.g. a replacement for a non-converged run)")
    a = ap.parse_args()
    override_a = dict(x.split("=", 1) for x in a.override_a)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261030)

    rows = []
    for name, field, tspec, role in SCENARIOS_3B:
        da = [d for d in glob.glob(os.path.join(a.refits_a, f"{name}_seed*")) if "SMOKE" not in d]
        db = [d for d in glob.glob(os.path.join(a.refits_b, f"{name}_seed*")) if "SMOKE" not in d]
        if name in override_a:
            da = [override_a[name]]
        if len(da) != 1 or len(db) != 1:
            print(f"{name}: A/B refit missing or ambiguous; skipped")
            continue
        ref, _ = load(a.ref, field)
        n_ch, n_dr, J = ref.shape
        flat = ref.reshape(-1, J)
        S = flat.shape[0]
        A, ma = load(da[0], field)
        B, mb = load(db[0], field)
        res = reweight({field: flat}, [PriorFactor(field, make(REFERENCE_PRIORS[field]), make(tspec))],
                       reff=relative_efficiency([ref]))
        w = res.psis_weights if res.psis_weights is not None else res.raw_weights
        uni = np.full(S, 1.0 / S)
        sP, sR, sN = wsummary(flat, w), wsummary(flat, res.raw_weights), wsummary(flat, uni)
        sA, sB = wsummary(A.reshape(-1, J), uni), wsummary(B.reshape(-1, J), uni)
        seP = boot_se(flat, w, n_ch, n_dr, rng)
        seA = boot_se(A.reshape(-1, J), uni, *A.shape[:2], rng)
        seB = boot_se(B.reshape(-1, J), uni, *B.shape[:2], rng)
        for st in STATS:
            for j in range(J):
                rows.append(dict(
                    scenario=name, field=field, role=role, status=res.status, pareto_k=res.pareto_k,
                    psis_ess=res.psis_ess, raw_ess=res.raw_ess, ess_fraction=res.ess_fraction,
                    channel=j, stat=st, psis=sP[st][j], raw_is=sR[st][j], refit_A=sA[st][j], refit_B=sB[st][j],
                    noadjust=sN[st][j],
                    e_psis_A=sP[st][j] - sA[st][j], e_psis_B=sP[st][j] - sB[st][j], e_AB=sA[st][j] - sB[st][j],
                    e_raw_A=sR[st][j] - sA[st][j], e_noadjust_A=sN[st][j] - sA[st][j],
                    se_psis=seP[st][j], se_A=seA[st][j], se_B=seB[st][j],
                    z_AB=(sA[st][j] - sB[st][j]) / np.hypot(seA[st][j], seB[st][j]),
                    z_psis_A=(sP[st][j] - sA[st][j]) / np.hypot(seP[st][j], seA[st][j]),
                    z_psis_B=(sP[st][j] - sB[st][j]) / np.hypot(seP[st][j], seB[st][j]),
                    A_field_rhat=ma["field_diag"]["rhat_max"], B_field_rhat=mb["field_diag"]["rhat_max"],
                    A_div=int(sum(ma["divergences_per_chain"])), B_div=int(sum(mb["divergences_per_chain"])),
                    A_allvars_rhat=ma["all_variables"]["rhat_max"], B_allvars_rhat=mb["all_variables"]["rhat_max"]))
        print(f"{name}: k={res.pareto_k:.3f} {res.status} ess_frac={res.ess_fraction:.3f}", flush=True)

    ch = pd.DataFrame(rows)
    ch.to_csv(os.path.join(a.out, "channel_level.csv"), index=False)
    sc = ch.groupby(["scenario", "role", "status", "pareto_k", "ess_fraction", "stat"], sort=False).agg(
        mae_psis_A=("e_psis_A", lambda x: np.abs(x).mean()), mae_psis_B=("e_psis_B", lambda x: np.abs(x).mean()),
        mae_AB=("e_AB", lambda x: np.abs(x).mean()), mae_raw_A=("e_raw_A", lambda x: np.abs(x).mean()),
        mae_noadjust_A=("e_noadjust_A", lambda x: np.abs(x).mean()),
        max_abs_z_AB=("z_AB", lambda x: np.abs(x).max()), max_abs_z_psis_A=("z_psis_A", lambda x: np.abs(x).max()),
        A_div=("A_div", "first"), B_div=("B_div", "first"),
        A_field_rhat=("A_field_rhat", "first"), B_field_rhat=("B_field_rhat", "first"),
        A_allvars_rhat=("A_allvars_rhat", "first"), B_allvars_rhat=("B_allvars_rhat", "first")).reset_index()
    sc.to_csv(os.path.join(a.out, "scenario_level.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
