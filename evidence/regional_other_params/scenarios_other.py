"""Phase 3B cross-prior scenarios. Every scenario changes ONE active prior field of the ROI-parameterized
reference model (roi_m stays LogNormal(0.2, 0.9)); all else, including ModelSpec, data and MCMC, is fixed.

Targets were fixed before any k-hat or refit comparison was computed.
"""
import math

import numpy as np
import tensorflow_probability.substrates.jax as tfp_jax

tfd = tfp_jax.distributions

# Meridian 2.1.0 defaults for the reference fit (PRIOR_INVENTORY.md): alpha_m ~ Uniform(0,1),
# ec_m ~ TruncatedNormal(0.8, 0.8, 0.1, 10).
REFERENCE_PRIORS = {
    "alpha_m": ("Uniform", (0.0, 1.0)),
    "ec_m": ("TruncatedNormal", (0.8, 0.8, 0.1, 10.0)),
    "sigma": ("HalfNormal", (5.0,)),
    "eta_m": ("HalfNormal", (1.0,)),
}

# (scenario, field, (kind, params), role)
SCENARIOS_3B = [
    ("AM1_beta_2_2", "alpha_m", ("Beta", (2.0, 2.0)), "moderate"),
    ("AM2_beta_5_2", "alpha_m", ("Beta", (5.0, 2.0)), "shift_high"),
    ("AM3_beta_2_8", "alpha_m", ("Beta", (2.0, 8.0)), "shift_low"),
    ("AM4_beta_0.5_0.5", "alpha_m", ("Beta", (0.5, 0.5)), "boundary_heavy"),
    # added after the first four, to cover k-hat in 0.6-0.9; chosen from reference-draw k-hat only (no refit seen)
    ("AM5_beta_1_3", "alpha_m", ("Beta", (1.0, 3.0)), "borderline"),
    # ec_m: all keep the reference truncation bounds (0.1, 10). EC1-EC5 fixed before any k-hat or refit result.
    ("EC1_tn_1.0_0.8", "ec_m", ("TruncatedNormal", (1.0, 0.8, 0.1, 10.0)), "mild_shift"),
    ("EC2_tn_0.4_0.4", "ec_m", ("TruncatedNormal", (0.4, 0.4, 0.1, 10.0)), "tight_low"),
    ("EC3_tn_1.5_0.5", "ec_m", ("TruncatedNormal", (1.5, 0.5, 0.1, 10.0)), "tight_high"),
    ("EC4_tn_0.8_2.0", "ec_m", ("TruncatedNormal", (0.8, 2.0, 0.1, 10.0)), "wider"),
    ("EC5_tn_3.0_0.5", "ec_m", ("TruncatedNormal", (3.0, 0.5, 0.1, 10.0)), "far_shift"),
    # sigma (scalar; reference posterior mean 0.784, sd 0.0071, so only priors of width ~0.01-0.04 can matter).
    # All supports (0, inf) lie inside HalfNormal's. SG1-SG5 fixed before any k-hat or refit result.
    ("SG1_hn_1.0", "sigma", ("HalfNormal", (1.0,)), "near_null_control"),
    ("SG2_ln_0.8_0.05", "sigma", ("LogNormal", (math.log(0.8), 0.05)), "weak_informative"),
    ("SG3_ln_0.8_0.02", "sigma", ("LogNormal", (math.log(0.8), 0.02)), "informative"),
    ("SG4_ln_0.85_0.02", "sigma", ("LogNormal", (math.log(0.85), 0.02)), "informative_shifted"),
    ("SG5_ln_0.95_0.02", "sigma", ("LogNormal", (math.log(0.95), 0.02)), "far_shift"),
    # eta_m (geo-level media effect scale, per channel). Reference HalfNormal(1); reference posterior mean 0.53-0.93.
    # All supports (0, inf) lie inside HalfNormal's. EA1-EA5 fixed before any k-hat or refit result.
    ("EA1_hn_0.5", "eta_m", ("HalfNormal", (0.5,)), "shrink"),
    ("EA2_hn_2.0", "eta_m", ("HalfNormal", (2.0,)), "wider"),
    ("EA3_hn_0.2", "eta_m", ("HalfNormal", (0.2,)), "strong_shrink"),
    ("EA4_ln_0_0.5", "eta_m", ("LogNormal", (0.0, 0.5)), "shifted"),
    ("EA5_ln_log2_0.3", "eta_m", ("LogNormal", (math.log(2.0), 0.3)), "far_shift"),
    # added after EA1-EA5 to cover k-hat 0.5-0.7; chosen from reference-draw k-hat only (no refit seen)
    ("EA6_hn_0.35", "eta_m", ("HalfNormal", (0.35,)), "mid_k"),
]

# Multi-factor / channel-specific scenarios: (scenario, {field: (kind, params)}, role). Params may be length-5
# arrays (one value per media channel). Fixed before any k-hat or refit result; seed bases 6000 (A) / 7000 (B).
_A = lambda a, b: ("Beta", (a, b))  # noqa: E731
MULTI_SCENARIOS = [
    ("J1_joint_am_beta_2_2__ec_tn_1.0_0.8", {"alpha_m": _A(2.0, 2.0),
                                              "ec_m": ("TruncatedNormal", (1.0, 0.8, 0.1, 10.0))}, "joint_mild"),
    ("J2_joint_am_beta_5_2__ec_tn_1.5_0.5", {"alpha_m": _A(5.0, 2.0),
                                              "ec_m": ("TruncatedNormal", (1.5, 0.5, 0.1, 10.0))}, "joint_shift"),
    ("J3_joint_am_beta_0.5_0.5__ec_tn_0.4_0.4", {"alpha_m": _A(0.5, 0.5),
                                                  "ec_m": ("TruncatedNormal", (0.4, 0.4, 0.1, 10.0))}, "joint_demanding"),
    ("C1_chanspec_am_beta", {"alpha_m": _A([2.0, 5.0, 2.0, 1.5, 5.0], [2.0, 2.0, 2.0, 1.5, 2.0])}, "channel_specific"),
    ("C2_chan2_only_am_beta_3_3", {"alpha_m": _A([1.0, 1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 1.0, 1.0])}, "one_channel"),
]

SEED_BASE_A, SEED_BASE_B = 3000, 4000  # seed = base + 1-based index in SCENARIOS_3B


def make(spec, name=None):
    kind, p = spec
    f = lambda v: np.asarray(v, dtype=np.float64)  # noqa: E731  (x64 so TFP does not evaluate in float32)
    if kind == "Uniform":
        return tfd.Uniform(f(p[0]), f(p[1]), name=name)
    if kind == "Beta":
        return tfd.Beta(f(p[0]), f(p[1]), name=name)
    if kind == "HalfNormal":
        return tfd.HalfNormal(f(p[0]), name=name)
    if kind == "LogNormal":
        return tfd.LogNormal(f(p[0]), f(p[1]), name=name)
    if kind == "TruncatedNormal":
        return tfd.TruncatedNormal(f(p[0]), f(p[1]), f(p[2]), f(p[3]), name=name)
    raise ValueError(kind)
