"""Unit tests for evidence/common/prior_reweight.py.

Run: python -I evidence/parity_main/test_prior_reweight.py   (env: meridian-psis, CPU, no MCMC)
The last test builds a Meridian model (downloads the demo CSV); the rest are model-free.
Inputs are synthetic draws from the reference prior, so these test the arithmetic and the
guards, not posterior overlap.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import arviz as az
import jax
import meridian.backend  # noqa: F401  (enables JAX x64, as in real use)
import numpy as np
from scipy.special import logsumexp

import tensorflow_probability.substrates.jax as tfp_jax

from prior_reweight import PriorFactor, good_k_threshold, relative_efficiency, reweight

tfd = tfp_jax.distributions
f64 = np.float64
S, J = 600, 5
RESULTS = []


def draws_from(dist, seed, n=S):
    return np.array(dist.sample(n, seed=jax.random.PRNGKey(seed)))


def LN(mu, sigma):
    return tfd.LogNormal(f64(mu), f64(sigma))


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa: BLE001
        print("   wrong exception:", type(e).__name__, e)
        return False
    return False


# ------------------------------- notebook's computation, copied from cells 17 ------------
def nb_prior_reweight_roi(roi, tmu, tsig, bmu=0.2, bsig=0.9):
    def lp(x, mu, s):
        return -np.log(x) - np.log(s) - 0.5 * np.log(2 * np.pi) - 0.5 * ((np.log(x) - mu) / s) ** 2

    logw = lp(roi, tmu, tsig).sum(1) - lp(roi, bmu, bsig).sum(1)
    raw_w = np.exp(logw - logsumexp(logw))
    lw_p, k = az.psislw(logw.copy())  # the notebook passes logw itself, which psislw mutates
    psis_w = np.exp(lw_p - logsumexp(lw_p))
    return logw, raw_w, psis_w, float(np.asarray(k)), 1 / np.sum(raw_w**2), 1 / np.sum(psis_w**2)


def test_requires_x64():
    roi = np.ones((S, J))
    factor = PriorFactor("roi_m", LN(0.2, 0.9), LN(0.5, 0.7))  # built while x64 is on
    jax.config.update("jax_enable_x64", False)
    try:
        assert raises(RuntimeError, lambda: reweight({"roi_m": roi}, [factor]))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_matches_notebook_roi():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 1)
    for tmu, tsig in [(0.2, 0.55), (0.6, 0.9), (0.0, 0.6), (0.8, 0.35)]:
        r = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(tmu, tsig))])
        logw, raw_w, psis_w, k, raw_ess, psis_ess = nb_prior_reweight_roi(roi, tmu, tsig)
        assert np.max(np.abs(r.log_ratio - logw)) < 1e-10, (tmu, tsig)
        assert np.max(np.abs(r.raw_weights - raw_w)) < 1e-12
        assert np.max(np.abs(r.psis_weights - psis_w)) < 1e-12
        assert abs(r.pareto_k - k) < 1e-12 and abs(r.raw_ess - raw_ess) < 1e-8
        assert abs(r.psis_ess - psis_ess) < 1e-8 and r.n_draws == S
        assert r.raw_weights.shape == (S,) and abs(r.raw_weights.sum() - 1) < 1e-12
        assert abs(r.psis_weights.sum() - 1) < 1e-12


def test_psislw_mutates_input_but_reweight_does_not():
    rng = np.random.default_rng(3)
    lw = rng.normal(size=S) * 2.0
    lw0 = lw.copy()
    az.psislw(lw)
    assert not np.array_equal(lw, lw0), "expected az.psislw to modify its input in place"
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 12)
    f = PriorFactor("roi_m", LN(0.2, 0.9), LN(-0.4, 0.35))
    r = reweight({"roi_m": roi}, [f])
    truth = (f.target.log_prob(roi) - f.reference.log_prob(roi)).sum(1)
    assert np.max(np.abs(r.log_ratio - np.asarray(truth))) < 1e-12
    assert np.max(np.abs(r.log_ratio - logsumexp(r.log_ratio) - np.log(r.raw_weights))) < 1e-9


def test_truncation_only_is_bounded_not_failure():
    a = draws_from(tfd.Sample(tfd.Uniform(f64(0), f64(1)), [1]), 13)
    r = reweight({"a": a}, [PriorFactor("a", tfd.Uniform(f64(0), f64(1)), tfd.Uniform(f64(0), f64(0.5)))])
    inside = (a[:, 0] < 0.5)
    assert r.status == "bounded" and np.isnan(r.pareto_k)
    assert r.n_zero_weight == int((~inside).sum()) and r.n_zero_weight > 0
    assert np.allclose(r.raw_weights[inside], 1.0 / inside.sum()) and np.all(r.raw_weights[~inside] == 0)
    assert abs(r.raw_ess - inside.sum()) < 1e-8 and r.psis_weights is None


def test_psis_runs_on_positive_density_draws_only():
    # 80% of draws get zero target density; the remaining log ratios are smooth.
    rng = np.random.default_rng(5)
    x = draws_from(tfd.Sample(tfd.Normal(f64(0), f64(1)), [1]), 14)
    x[:, 0] = np.abs(x[:, 0])  # all positive; target below is a HalfNormal-like truncation
    ref = tfd.Normal(f64(0), f64(1))
    tgt = tfd.TruncatedNormal(f64(0.5), f64(0.8), f64(0.0), f64(0.4))  # support [0, 0.4]
    r = reweight({"x": x}, [PriorFactor("x", ref, tgt)])
    nz = r.n_zero_weight
    assert 0 < nz < S - 10
    n_f = S - nz
    assert abs(r.k_threshold - good_k_threshold(n_f)) < 1e-12
    lr_f = r.log_ratio[np.isfinite(r.log_ratio)]
    k_direct = float(az.psislw(lr_f.copy())[1])
    assert abs(r.pareto_k - k_direct) < 1e-12, "k must come from the positive-density draws"
    assert np.all(r.psis_weights[~np.isfinite(r.log_ratio)] == 0) and abs(r.psis_weights.sum() - 1) < 1e-12
    assert r.ess_fraction <= n_f / S + 1e-9


def test_relative_efficiency():
    rng = np.random.default_rng(6)
    iid = rng.normal(size=(2, 500, 3))
    assert 0.8 < relative_efficiency([iid]) <= 1.0
    ar = np.zeros((2, 500, 3))
    e = rng.normal(size=(2, 500, 3))
    for t in range(1, 500):
        ar[:, t] = 0.9 * ar[:, t - 1] + e[:, t]
    r_ar = relative_efficiency([ar])
    assert 0.02 < r_ar < 0.3, r_ar
    assert raises(ValueError, lambda: relative_efficiency([np.ones(5)]))
    assert raises(ValueError, lambda: relative_efficiency([]))
    # reff changes the tail size PSIS uses, hence (generally) k
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 15)
    f = PriorFactor("roi_m", LN(0.2, 0.9), LN(0.8, 0.35))
    k1 = reweight({"roi_m": roi}, [f], reff=1.0).pareto_k
    k2 = reweight({"roi_m": roi}, [f], reff=0.1).pareto_k
    assert k1 != k2


def test_identity():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 2)
    r = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.2, 0.9))])
    assert r.status == "identical" and np.isnan(r.pareto_k) and r.psis_weights is None
    assert np.all(r.log_ratio == 0) and np.allclose(r.raw_weights, 1 / S)
    assert abs(r.raw_ess - S) < 1e-8 and abs(r.ess_fraction - 1) < 1e-12


def test_near_identity_is_not_flagged_identical():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 3)
    r = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.2000001, 0.9))])
    assert r.status == "good" and np.isfinite(r.pareto_k)


def test_channel_specific_only_changed_channel_contributes():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 4)
    mu_t = np.full(J, 0.2)
    mu_t[2] = 0.7
    target = tfd.LogNormal(mu_t.astype(f64), f64(0.9) * np.ones(J))  # batch shape (5,)
    r = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), target)])
    only2 = reweight({"roi_m": roi[:, 2:3]}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.7, 0.9))])
    assert np.max(np.abs(r.log_ratio - only2.log_ratio)) < 1e-10


def test_two_factors_sum_exactly_once():
    a = draws_from(tfd.Sample(tfd.Uniform(f64(0), f64(1)), [J]), 5)
    ec = draws_from(tfd.Sample(tfd.TruncatedNormal(f64(0.8), f64(0.8), f64(0.1), f64(10)), [J]), 6)
    fa = PriorFactor("alpha_m", tfd.Uniform(f64(0), f64(1)), tfd.Beta(f64(2), f64(2)))
    fe = PriorFactor(
        "ec_m",
        tfd.TruncatedNormal(f64(0.8), f64(0.8), f64(0.1), f64(10)),
        tfd.TruncatedNormal(f64(1.2), f64(0.5), f64(0.1), f64(10)),
    )
    both = reweight({"alpha_m": a, "ec_m": ec}, [fa, fe])
    ra, re_ = reweight({"alpha_m": a}, [fa]), reweight({"ec_m": ec}, [fe])
    assert np.max(np.abs(both.log_ratio - (ra.log_ratio + re_.log_ratio))) < 1e-12
    assert set(both.factor_log_ratios) == {"alpha_m", "ec_m"}


def test_sample_wrapped_vector_factor():
    # event-dim distribution (like tau_g_excl_baseline): log_prob already reduces the event axis
    ref = tfd.Sample(tfd.Normal(f64(0), f64(5)), [2])
    tgt = tfd.Sample(tfd.Normal(f64(0), f64(3)), [2])
    x = draws_from(ref, 7)
    r = reweight({"tau": x}, [PriorFactor("tau", ref, tgt)])
    man = (-np.log(3) - 0.5 * (x / 3) ** 2 + np.log(5) + 0.5 * (x / 5) ** 2).sum(1)
    assert r.log_ratio.shape == (S,) and np.max(np.abs(r.log_ratio - man)) < 1e-10


def test_zero_target_density_draws_get_zero_weight():
    a = draws_from(tfd.Sample(tfd.Uniform(f64(0), f64(1)), [1]), 8)
    a[:5, 0] = 0.0  # on Beta(2,2)'s boundary: density 0
    r = reweight({"alpha_m": a}, [PriorFactor("alpha_m", tfd.Uniform(f64(0), f64(1)), tfd.Beta(f64(2), f64(2)))])
    assert r.n_zero_weight == 5 and np.all(r.raw_weights[:5] == 0)
    assert np.all(r.psis_weights[:5] == 0) and abs(r.psis_weights.sum() - 1) < 1e-12


def test_invalid_inputs_fail_loudly():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 9)
    ok = PriorFactor("roi_m", LN(0.2, 0.9), LN(0.5, 0.7))
    bad = roi.copy()
    bad[0, 0] = -1.0
    assert raises(ValueError, lambda: reweight({"roi_m": bad}, [ok])), "non-positive ROI draw"
    bad = roi.copy()
    bad[0, 0] = 0.0
    assert raises(ValueError, lambda: reweight({"roi_m": bad}, [ok])), "zero ROI draw"
    bad = roi.copy()
    bad[3, 1] = np.nan
    assert raises(ValueError, lambda: reweight({"roi_m": bad}, [ok])), "NaN draw"
    bad = roi.copy()
    bad[3, 1] = np.inf
    assert raises(ValueError, lambda: reweight({"roi_m": bad}, [ok])), "inf draw"
    assert raises(ValueError, lambda: reweight(
        {"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.5, -0.7))])), "target sigma<=0"
    assert raises(ValueError, lambda: reweight(
        {"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, -0.9), LN(0.5, 0.7))])), "reference sigma<=0"
    det = tfd.Deterministic(f64(1.0))
    assert raises(ValueError, lambda: reweight(
        {"slope": np.ones((S, J))}, [PriorFactor("slope", det, det)])), "Deterministic prior"
    assert raises(ValueError, lambda: reweight({"roi_m": roi}, [])), "no factors"
    assert raises(ValueError, lambda: reweight({"roi_m": roi}, [ok, ok])), "duplicate factor"
    assert raises(KeyError, lambda: reweight({}, [ok])), "missing draws"
    assert raises(ValueError, lambda: reweight(
        {"roi_m": roi, "b": roi[:10]},
        [ok, PriorFactor("b", LN(0.2, 0.9), LN(0.5, 0.7))])), "mismatched draw counts"
    assert raises(ValueError, lambda: reweight({"roi_m": roi[:1]}, [ok])), "single draw"
    # target has zero density everywhere
    a = np.zeros((S, 1))
    assert raises(ValueError, lambda: reweight(
        {"a": a}, [PriorFactor("a", tfd.Uniform(f64(0), f64(1)), tfd.Beta(f64(2), f64(2)))])), "all-zero target"


def test_status_levels():
    roi = draws_from(tfd.Sample(LN(0.2, 0.9), [J]), 10)
    good = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.3, 0.9))])
    assert good.status == "good"
    stress = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(-0.4, 0.35))])
    assert stress.status in ("warning", "fail") and stress.pareto_k > good.pareto_k
    assert stress.ess_fraction < good.ess_fraction


def test_meridian_prior_broadcast_roi():
    from demo_data import build_data
    from meridian.model import model, prior_distribution, spec

    data = build_data()

    def pb(mu, sg):
        p = prior_distribution.PriorDistribution(roi_m=tfd.LogNormal(f64(mu), f64(sg), name="roi_m"))
        m = model.Meridian(input_data=data, model_spec=spec.ModelSpec(prior=p, enable_aks=False, media_prior_type="roi"))
        return m.model_context.prior_broadcast.roi_m

    ref, tgt = pb(0.2, 0.9), pb(0.6, 0.6)
    roi = draws_from(ref, 11)
    assert roi.shape == (S, J)
    via_meridian = reweight({"roi_m": roi}, [PriorFactor("roi_m", ref, tgt)])
    via_tfp = reweight({"roi_m": roi}, [PriorFactor("roi_m", LN(0.2, 0.9), LN(0.6, 0.6))])
    assert np.max(np.abs(via_meridian.log_ratio - via_tfp.log_ratio)) < 1e-10
    assert abs(via_meridian.pareto_k - via_tfp.pareto_k) < 1e-12


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - n_fail}/{len(tests)} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
