"""Phase 8: rerun of selected Phase 3B scenarios on upstream main versus the Oct-27 (Meridian 2.1.0) runs.

Usage: python -I evidence/parity_main/compare_main.py
Same seeds, data and MCMC settings as the old runs, so differences come from code changes on main
(and, if the sampler is not bitwise reproducible across installs, from sampling noise).

Reports, per scenario: (1) new reference vs old reference and new refit vs old refit (same seed) on the
changed field; (2) k-hat, ESS fraction and PSIS error against the new refit, next to the old numbers.
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.abspath(os.path.join(HERE, ".."))
for sub in ("common", "regional_other_params", "regional_roi"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np
import pandas as pd
import xarray as xr

import meridian.backend  # noqa: F401  (JAX x64)
from prior_reweight import PriorFactor, reweight, relative_efficiency
from scenarios_other import REFERENCE_PRIORS, SCENARIOS_3B, make
from analyze_noise import wsummary

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = os.path.join(ROOT, "runs", "phase8_main")
OLD_REF = os.path.join(ROOT, "runs", "ref_strong_ta0.95_seed124_20261027-132552")
NEW_REF = glob.glob(os.path.join(NEW, "ref_strong_ta0.95_seed124_*"))[0]
# scenario -> (new refit root, old refit root with the same seed)
RUNS = {
    "AM1_beta_2_2": ("refits_A", "phase3b_refits_A"),
    "AM5_beta_1_3": ("refits_A", "phase3b_refits_A"),
    "EC1_tn_1.0_0.8": ("refits_A", "phase3b_refits_A"),
    "EC4_tn_0.8_2.0": ("refits_B", "phase3b_refits_B"),
}
STATNAMES = ("mean", "q05", "q50", "q95")


def load(d, var):
    x = np.asarray(xr.open_dataset(os.path.join(d, "posterior.nc"))[var])
    return x[..., None] if x.ndim == 2 else x


def one(pattern):
    m = [d for d in glob.glob(pattern) if "SMOKE" not in d]
    assert len(m) == 1, (pattern, m)
    return m[0]


def mae(a, b):
    return {s: float(np.mean(np.abs(a[s] - b[s]))) for s in STATNAMES}


def summ(x):
    S = x.reshape(-1, x.shape[-1])
    return wsummary(S, np.full(len(S), 1.0 / len(S)))


def meta(d):
    return json.load(open(os.path.join(d, "run_metadata.json")))


def main():
    old_tbl = pd.read_csv(os.path.join(EVIDENCE, "regional_other_params", "results", "scenario_level.csv"))
    mo, mn = meta(OLD_REF), meta(NEW_REF)
    print("reference: old versions", mo["versions"]["meridian"], "| new", mn["versions"]["meridian"],
          "| div old/new", mo["sampler_stats"]["diverging"]["n_true"], mn["sampler_stats"]["diverging"]["n_true"])
    rows = []
    for name, field, tspec, role in SCENARIOS_3B:
        if name not in RUNS:
            continue
        new_root, old_root = RUNS[name]
        d_new = one(os.path.join(NEW, new_root, f"{name}_seed*"))
        d_old = one(os.path.join(ROOT, "runs", old_root, f"{name}_seed*"))
        ref_old, ref_new = load(OLD_REF, field), load(NEW_REF, field)
        fit_old, fit_new = load(d_old, field), load(d_new, field)
        n_ch, n_dr, J = ref_new.shape
        flat = ref_new.reshape(-1, J)
        res = reweight({field: flat}, [PriorFactor(field, make(REFERENCE_PRIORS[field]), make(tspec))],
                       reff=relative_efficiency([ref_new]))
        w = res.psis_weights if res.psis_weights is not None else res.raw_weights
        sP, sN = wsummary(flat, w), summ(ref_new)
        sFn, sFo, sRo = summ(fit_new), summ(fit_old), summ(ref_old)
        o = old_tbl[(old_tbl.scenario == name)]
        o_mean = o[o.stat == "mean"].iloc[0]
        e_new = mae(sP, sFn)
        r = dict(
            scenario=name, field=field,
            k_old=o_mean.pareto_k, k_new=res.pareto_k, status_old=o_mean.status, status_new=res.status,
            essfrac_old=o_mean.ess_fraction, essfrac_new=res.ess_fraction,
            ref_new_vs_old_mean=mae(sN, sRo)["mean"], fit_new_vs_old_mean=mae(sFn, sFo)["mean"],
            psis_vs_refit_new_mean=e_new["mean"], psis_vs_refit_new_q05=e_new["q05"],
            psis_vs_refit_new_q95=e_new["q95"],
            noadjust_vs_refit_new_mean=mae(sN, sFn)["mean"],
            psis_vs_A_old_mean=o_mean.mae_psis_A, AB_old_mean=o_mean.mae_AB,
            new_field_rhat=meta(d_new)["field_diag"]["rhat_max"],
            new_field_ess_bulk=meta(d_new)["field_diag"]["ess_bulk_min"],
            new_div=int(sum(meta(d_new)["divergences_per_chain"])),
            new_allvars_rhat=meta(d_new)["all_variables"]["rhat_max"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "compare_main.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(df.round(4).T.to_string())


if __name__ == "__main__":
    main()
