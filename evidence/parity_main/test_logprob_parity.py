"""Phase 1: log-density parity and invariants for the prior ratio.

Run: python -I evidence/parity_main/test_logprob_parity.py   (env: meridian-psis, CPU, no MCMC)

Draws used here are samples from the reference *prior* objects Meridian builds, not
posterior draws; they exercise the density arithmetic, not posterior overlap.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from demo_data import build_data  # noqa: E402  (sets JAX env vars first)

import arviz as az
import jax
import numpy as np
from scipy import stats
from scipy.special import logsumexp

from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax

tfd = tfp_jax.distributions
LOG_2PI = np.log(2.0 * np.pi)
S = 600
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---- notebook's manual density, copied verbatim from cell 17 -----------------
def lognormal_logpdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("LogNormal ROI draws must be strictly positive.")
    z = (np.log(x) - mu) / sigma
    return -np.log(x) - np.log(sigma) - 0.5 * LOG_2PI - 0.5 * z**2


def make_model(data, **prior_kw):
    ms = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(**prior_kw),
        enable_aks=False,
        media_prior_type="roi",
    )
    m = model.Meridian(input_data=data, model_spec=ms)
    return m, m.model_context.prior_broadcast


def draws(dist, seed):
    return np.asarray(dist.sample(S, seed=jax.random.PRNGKey(seed)))


def main():
    data = build_data()
    f64 = np.float64

    # ---------------------------------------------------------------- roi_m
    m0, pb0 = make_model(data, roi_m=tfd.LogNormal(f64(0.2), f64(0.9), name="roi_m"))
    m1, pb1 = make_model(data, roi_m=tfd.LogNormal(f64(0.6), f64(0.6), name="roi_m"))
    roi_d = draws(pb0.roi_m, 1)  # (S, 5)
    record("roi_m: draw shape is (S, n_media)", roi_d.shape == (S, 5), str(roi_d.shape))

    lp0 = np.asarray(pb0.roi_m.log_prob(roi_d))
    record("roi_m: broadcast log_prob has shape (S, J), not reduced", lp0.shape == (S, 5), str(lp0.shape))
    man0 = lognormal_logpdf(roi_d, 0.2, 0.9)
    err = np.max(np.abs(lp0 - man0))
    record("roi_m: manual LogNormal == Meridian broadcast log_prob (ref)", err < 1e-10, f"max|diff|={err:.2e}")
    lp1 = np.asarray(pb1.roi_m.log_prob(roi_d))
    err = np.max(np.abs(lp1 - lognormal_logpdf(roi_d, 0.6, 0.6)))
    record("roi_m: manual LogNormal == Meridian broadcast log_prob (target)", err < 1e-10, f"max|diff|={err:.2e}")
    err = np.max(np.abs(lp0 - np.asarray(tfd.LogNormal(f64(0.2), f64(0.9)).log_prob(roi_d))))
    record("roi_m: scalar tfd.LogNormal == broadcast log_prob", err < 1e-10, f"max|diff|={err:.2e}")

    # sigma-term check: dropping -log(sigma) must change the ratio when sigma changes
    z0 = (np.log(roi_d) - 0.2) / 0.9
    z1 = (np.log(roi_d) - 0.6) / 0.6
    ratio_full = (-np.log(0.6) - 0.5 * z1**2) - (-np.log(0.9) - 0.5 * z0**2)
    ratio_noconst = (-0.5 * z1**2) - (-0.5 * z0**2)
    record("roi_m: omitting -log(sigma) changes the log ratio (constant is needed)",
           np.max(np.abs(ratio_full - ratio_noconst)) > 0.1,
           f"|diff|={np.max(np.abs(ratio_full - ratio_noconst)):.3f}")

    # joint log ratio = sum over channels
    lw = (lp1 - lp0).sum(axis=-1)
    man = (lognormal_logpdf(roi_d, 0.6, 0.6) - lognormal_logpdf(roi_d, 0.2, 0.9)).sum(axis=1)
    record("roi_m: joint log ratio sums channel axis only -> shape (S,)",
           lw.shape == (S,) and np.max(np.abs(lw - man)) < 1e-10, str(lw.shape))

    # identity
    lw_id = (lp0 - lp0).sum(axis=-1)
    w = np.exp(lw_id - logsumexp(lw_id))
    record("identity: raw log ratio == 0", np.max(np.abs(lw_id)) == 0.0)
    record("identity: weights uniform and ESS == S",
           np.allclose(w, 1.0 / S) and abs(1.0 / np.sum(w**2) - S) < 1e-8,
           f"ESS={1.0 / np.sum(w**2):.6f}")

    # normalization
    lw_n = lw - logsumexp(lw)
    record("normalize: weights sum to 1", abs(np.exp(lw_n).sum() - 1.0) < 1e-12)

    # invalid inputs: what does TFP do for non-positive ROI?
    bad = roi_d.copy()
    bad[0, 0] = -1.0
    bad[1, 1] = 0.0
    lpbad = np.asarray(pb0.roi_m.log_prob(bad))
    record("roi_m: TFP log_prob at x<=0 is non-finite (so it needs an explicit guard)",
           (not np.isfinite(lpbad[0, 0])) and (not np.isfinite(lpbad[1, 1])),
           f"x=-1 -> {lpbad[0, 0]}, x=0 -> {lpbad[1, 1]}")
    try:
        lognormal_logpdf(bad, 0.2, 0.9)
        record("roi_m: notebook helper rejects x<=0", False)
    except ValueError:
        record("roi_m: notebook helper rejects x<=0", True)
    # sigma <= 0: neither TFP (even with validate_args=True) nor the notebook helper raises;
    # both return NaN, which would silently poison the weights unless checked.
    v_tfp = np.asarray(tfd.LogNormal(f64(0.0), f64(-1.0), validate_args=True).log_prob(roi_d))
    record("roi_m: TFP LogNormal(sigma=-1, validate_args=True) does not raise, returns NaN",
           np.all(np.isnan(v_tfp)), "-> explicit sigma>0 and finite-ratio checks required")
    with np.errstate(all="ignore"):
        v_man = lognormal_logpdf(roi_d, 0.0, -1.0)
    record("roi_m: notebook helper with sigma=-1 returns NaN (no guard)",
           np.all(np.isnan(v_man)), "-> explicit sigma>0 and finite-ratio checks required")

    # ---------------------------------------------------------------- alpha_m
    _, pbA = make_model(data, alpha_m=tfd.Beta(f64(2.0), f64(2.0), name="alpha_m"))
    a_d = draws(pb0.alpha_m, 2)  # reference is Uniform(0,1) default
    record("alpha_m: Meridian accepts Beta(2,2) as a prior (same [0,1] support)", True)
    lpU = np.asarray(pb0.alpha_m.log_prob(a_d))
    lpB = np.asarray(pbA.alpha_m.log_prob(a_d))
    record("alpha_m: Uniform(0,1) log_prob == 0 on interior", np.max(np.abs(lpU)) < 1e-12)
    err = np.max(np.abs(lpB - stats.beta(2, 2).logpdf(a_d)))
    record("alpha_m: Beta(2,2) log_prob == scipy", err < 1e-10, f"max|diff|={err:.2e}")
    edge = np.array([[0.0] * 5, [1.0] * 5])
    lpU_e = np.asarray(pb0.alpha_m.log_prob(edge))
    lpB_e = np.asarray(pbA.alpha_m.log_prob(edge))
    record("alpha_m: edge draws 0 and 1 (boundary behaviour)",
           True, f"Uniform: {lpU_e[0, 0]}/{lpU_e[1, 0]}  Beta(2,2): {lpB_e[0, 0]}/{lpB_e[1, 0]}")

    # ---------------------------------------------------------------- ec_m
    ref_ec = pb0.ec_m
    e_d = draws(ref_ec, 3)
    _, pbE = make_model(data, ec_m=tfd.TruncatedNormal(f64(1.2), f64(0.5), f64(0.1), f64(10.0), name="ec_m"))
    lp_ref = np.asarray(ref_ec.log_prob(e_d))
    sp_ref = stats.truncnorm((0.1 - 0.8) / 0.8, (10 - 0.8) / 0.8, loc=0.8, scale=0.8).logpdf(e_d)
    record("ec_m: default TruncatedNormal log_prob == scipy truncnorm", np.max(np.abs(lp_ref - sp_ref)) < 1e-8,
           f"max|diff|={np.max(np.abs(lp_ref - sp_ref)):.2e}")
    lp_t = np.asarray(pbE.ec_m.log_prob(e_d))
    sp_t = stats.truncnorm((0.1 - 1.2) / 0.5, (10 - 1.2) / 0.5, loc=1.2, scale=0.5).logpdf(e_d)
    record("ec_m: target TruncatedNormal log_prob == scipy truncnorm (normalizer included)",
           np.max(np.abs(lp_t - sp_t)) < 1e-8, f"max|diff|={np.max(np.abs(lp_t - sp_t)):.2e}")
    below = np.full((1, 5), 0.05)
    record("ec_m: log_prob below the truncation bound is -inf (support check)",
           np.all(np.asarray(ref_ec.log_prob(below)) == -np.inf), str(np.asarray(ref_ec.log_prob(below))[0, 0]))

    # support-violation illustration: wider target truncation
    _, pbW = make_model(data, ec_m=tfd.TruncatedNormal(f64(0.8), f64(0.8), f64(0.01), f64(10.0), name="ec_m"))
    record("ec_m: Meridian accepts a wider truncation (lower bound 0.01 < 0.1)", True,
           "accepted: the ratio is finite on reference draws but target mass in [0.01,0.1) is unreachable")

    # ------------------------------------------------- joint two-factor change
    lw_a = (lpB - lpU).sum(axis=-1)
    lw_e = (lp_t - lp_ref).sum(axis=-1)
    record("alpha_m+ec_m: total log ratio is the sum of the two factors, each once",
           lw_a.shape == lw_e.shape == (S,))

    # ---------------------------------------------------------------- sigma (scalar)
    _, pbS = make_model(data, sigma=tfd.HalfNormal(f64(2.0), name="sigma"))
    sg = draws(pb0.sigma, 4)
    record("sigma: node value is scalar per draw", sg.shape == (S,), str(sg.shape))
    err = np.max(np.abs(np.asarray(pbS.sigma.log_prob(sg)) - stats.halfnorm(scale=2.0).logpdf(sg)))
    record("sigma: HalfNormal(2) log_prob == scipy", err < 1e-10, f"max|diff|={err:.2e}")

    # ------------------------------------------------- tau_g_excl_baseline (Sample wrapper)
    tg = draws(pb0.tau_g_excl_baseline, 5)  # (S, G-1)
    lp_tg = np.asarray(pb0.tau_g_excl_baseline.log_prob(tg))
    record("tau_g_excl_baseline: broadcast log_prob has shape (S, G-1)", lp_tg.shape == (S, 2), str(lp_tg.shape))
    err = np.max(np.abs(lp_tg - stats.norm(0, 5).logpdf(tg)))
    record("tau_g_excl_baseline: Normal(0,5) log_prob == scipy", err < 1e-10, f"max|diff|={err:.2e}")

    # ---------------------------------------------------------------- slope_m deterministic
    sl = pb0.slope_m
    inner = sl
    while hasattr(inner, "distribution") and inner.distribution is not inner:
        inner = inner.distribution
    record("slope_m: default is Deterministic (point mass, not reweightable)",
           isinstance(inner, tfd.Deterministic), type(inner).__name__)

    # ---------------------------------------------------------------- PSIS behaviour
    rng = np.random.default_rng(0)
    lw_r = rng.normal(size=S) * 0.5
    lw_p, k = az.psislw(lw_r)
    record("psislw: output is normalized and shape (S,)",
           lw_p.shape == (S,) and abs(logsumexp(lw_p)) < 1e-10, f"k={float(k):.3f}")
    lw_z, k_z = az.psislw(np.zeros(S))
    record("psislw: identity input (all-zero log ratios) gives uniform weights but k=inf",
           np.allclose(np.exp(lw_z), 1.0 / S) and np.isinf(float(k_z)),
           f"k={float(k_z)} -> a status rule on k alone would call the identity case 'fail'; special-case constant log ratios")
    lw_t, k_t = az.psislw(rng.normal(size=S) * 1e-8)
    record("psislw: near-identity input (1e-8 noise) gives a finite, sensible k",
           np.isfinite(float(k_t)) and float(k_t) < 0.5, f"k={float(k_t):.3f}")
    lw_inf = lw_r.copy()
    lw_inf[3] = np.inf
    try:
        out = az.psislw(lw_inf)
        record("psislw: +inf log weight is NOT rejected by ArviZ (needs explicit guard)",
               not np.all(np.isfinite(out[0])), f"k={out[1]}")
    except Exception as e:  # noqa: BLE001
        record("psislw: +inf log weight raises", True, type(e).__name__)
    lw_nan = lw_r.copy()
    lw_nan[3] = np.nan
    try:
        out = az.psislw(lw_nan)
        record("psislw: NaN log weight silently propagates (needs explicit guard)",
               not np.all(np.isfinite(out[0])), "output contains non-finite values")
    except Exception as e:  # noqa: BLE001
        record("psislw: NaN log weight raises", True, type(e).__name__)

    # ---------------------------------------------------------------- summary
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
