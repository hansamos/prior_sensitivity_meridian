# Regional test plan (written before any regional target refit)

Meridian 2.1.0, ROI parameterization, the demo dataset (40 geos, 156 weeks, 5 media channels), AKS on. Reference priors: `roi_m` LogNormal(0.2, 0.9); all others Meridian defaults (`eta_m` HalfNormal(1)).

**Question.** Does the national result (reweighting is accurate for close and mid prior changes; k̂ and ESS flag the far ones) also hold for the regional model, for the treatment prior `roi_m` and the geo-hierarchy prior `eta_m`?

## Design
- 2 parameters x 3 tiers (close, mid, far) x 10 priors = **60 target refits** (A), plus 6 second refits (B) of the 5th target by KL in each tier x parameter, for the refit-to-refit noise floor.
- One parameter changes at a time, common to all 5 channels.
- Targets: `calibration/regional_targets.csv`, built by `make_targets.py` with no model fitted. **SHA-256: 11a48a59f091199cad2eb352cd50955052326e42c20169ae3d9de8a43f742274**
  - `roi_m`: the national `roi_m` targets, split into rank terciles of KL(target ‖ reference prior), 10 per tercile at evenly spaced ranks (the same priors as the national test).
  - `eta_m`: HalfNormal(s) vs HalfNormal(1), KL = -ln s + (s² - 1)/2. Pool: 400 log-spaced s in [0.2, 4] with KL ≥ 0.01; terciles of KL; 5 per side of s = 1 (10 from one side if the other has fewer than 5).
- Every refit: 4 chains, 1000 adapt, 500 burn-in, 1000 kept, target_accept 0.95. Seeds 20000 + index (A), 30000 + index (B).
- Pilot first: the 6 second-refit targets (indices 5, 15, 25, 35, 45, 55) as A refits, to check runtime, convergence and outputs before the other 54.

## Reference
The existing regional reference (4 x 1000 draws) has `roi_m` R-hat 1.004 / bulk ESS 705 and `eta_m` R-hat 1.005 / bulk ESS 833, below the 1000 threshold set in advance for `eta_m`. A longer reference is used for all weights: same model and settings, **n_keep = 5000 per chain (20000 draws), seed 9101**, as in the national test.

## Metrics (as in the national test, fixed before results)
KL(full refit ‖ reweighted posterior), Wasserstein-1, KS, |mean error| / refit SD, |median error| / refit SD, refit mean outside the reweighted 90% interval, 90%-interval mismatch, channel-ranking agreement; Pareto k̂ and ESS fraction. Noise floor: split-chain baseline and A-vs-B refits.

## Refit quality rule
A refit is flagged if R-hat > 1.01 or bulk ESS < 400 on the perturbed parameter. Results are reported with and without flagged refits.

## Reading the results
Median KL and mean error by tier and parameter; k̂ x ESS decision table (same cut points as the national test: k̂ 0.5 / 0.7, ESS 10%); comparison with the national `roi_m` result at matching prior KL. No claim beyond one simulated dataset, two parameters and one reference fit.
