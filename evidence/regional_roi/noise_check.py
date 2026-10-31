"""Phase 3: express PSIS-vs-refit mean errors in units of Monte Carlo noise (rough, one number per channel).

z = err_psis(mean) / sqrt(mcse_ref^2 / ess_fraction + mcse_refit^2)
mcse_ref is the reference fit's MCSE of the mean; dividing its variance by the PSIS ESS fraction is a
crude inflation for weighting (it ignores reff and the covariance of weights and draws), so z is a
screening quantity, not a test. Reads CSVs written by analyze.py and the run directories only.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--refits", required=True)
ap.add_argument("--results", required=True)
a = ap.parse_args()

ref = pd.read_csv(os.path.join(a.ref, "roi_m_summary.csv")).set_index("channel")
chan = pd.read_csv(os.path.join(a.results, "channel_level.csv"))
scen = pd.read_csv(os.path.join(a.results, "scenario_level.csv")).set_index("scenario")
rows = []
for sc, g in chan[chan.stat == "mean"].groupby("scenario", sort=False):
    d = glob.glob(os.path.join(a.refits, f"{sc}_seed*"))[0]
    rf = pd.read_csv(os.path.join(d, "roi_m_summary.csv")).set_index("channel")
    f = scen.loc[sc, "ess_fraction"]
    for _, r in g.iterrows():
        noise = np.sqrt(ref.loc[r.channel, "mcse_mean"] ** 2 / f + rf.loc[r.channel, "mcse_mean"] ** 2)
        rows.append(dict(scenario=sc, channel=r.channel, err_psis=r.err_psis, err_noadjust=r.err_noadjust,
                         noise_mcse=noise, z_psis=r.err_psis / noise, z_noadjust=r.err_noadjust / noise))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(a.results, "mean_error_in_mcse_units.csv"), index=False)
s = out.groupby("scenario", sort=False).agg(max_abs_z_psis=("z_psis", lambda x: np.abs(x).max()),
                                            max_abs_z_noadjust=("z_noadjust", lambda x: np.abs(x).max()))
print(s.round(2).to_string())
