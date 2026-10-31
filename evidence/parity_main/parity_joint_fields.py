import os, sys, warnings, dataclasses
warnings.filterwarnings("ignore")
import numpy as np
from meridian import backend, constants
from meridian.model import model, model_test_data, spec, prior_distribution

tfd = backend.tfd
f = backend.np_float_dtype
model_test_data.WithInputDataSamples.setup()
inst = model_test_data.WithInputDataSamples()
N = 40

def build(data, **prior):
    ms = spec.ModelSpec(prior=prior_distribution.PriorDistribution(**prior), media_prior_type="roi")
    return model.Meridian(input_data=data, model_spec=ms)

def my_ratio(m_ref, m_tgt, state, fields):
    """log-density ratio from the broadcast priors of the changed fields only."""
    pr, pt = m_ref.model_context.prior_broadcast, m_tgt.model_context.prior_broadcast
    tot = 0.0
    for fld in fields:
        x = state[fld]
        d = np.asarray(getattr(pt, fld).log_prob(x)) - np.asarray(getattr(pr, fld).log_prob(x))
        tot = tot + d.reshape(N, -1).sum(axis=1)
    return tot

def joint_ratio(m_ref, m_tgt, state):
    jr = m_ref.posterior_sampler_callable._get_joint_dist_sampling_unpinned()
    jt = m_tgt.posterior_sampler_callable._get_joint_dist_sampling_unpinned()
    return np.asarray(jt.log_prob(state)) - np.asarray(jr.log_prob(state))

def case(label, data, targets, rf=False):
    m_ref = build(data)
    m_tgt = build(data, **targets)
    jd = m_ref.posterior_sampler_callable._get_joint_dist_sampling_unpinned()
    st = jd.sample(N, seed=(backend.random.prng_key(7) if backend.config.get_backend() == backend.config.Backend.JAX else 7))
    state = st._asdict()
    a = joint_ratio(m_ref, m_tgt, st)
    b = my_ratio(m_ref, m_tgt, state, list(targets))
    print(f"{label:55s} max|joint-mine|={np.max(np.abs(a-b)):.2e}  ratio sd={a.std():.3f}  finite={np.isfinite(a).all()}")

mo = inst.short_input_data_with_media_only
mr = inst.short_input_data_with_media_and_rf
case("media_only: alpha_m Beta(2,2)", mo, dict(alpha_m=tfd.Beta(f(2), f(2))))
case("media_only: ec_m TN(1.0,0.8,0.1,10)", mo, dict(ec_m=tfd.TruncatedNormal(f(1.0), f(0.8), f(0.1), f(10.0))))
case("media_only: alpha_m + ec_m", mo, dict(alpha_m=tfd.Beta(f(2), f(2)), ec_m=tfd.TruncatedNormal(f(1.0), f(0.8), f(0.1), f(10.0))))
case("media_only: alpha_m per channel", mo, dict(alpha_m=tfd.Beta(np.array([2, 5, 1.5], dtype=f), np.array([2, 2, 1.5], dtype=f))))
case("media_only: eta_m HalfNormal(0.5)", mo, dict(eta_m=tfd.HalfNormal(f(0.5))))
case("media_only: sigma HalfNormal(2)", mo, dict(sigma=tfd.HalfNormal(f(2.0))))
case("media_only: gamma_c Normal(0,2)", mo, dict(gamma_c=tfd.Normal(f(0), f(2))))
case("media_only: xi_c HalfNormal(2)", mo, dict(xi_c=tfd.HalfNormal(f(2.0))))
case("media_only: knot_values Normal(0,2)", mo, dict(knot_values=tfd.Normal(f(0), f(2))))
case("media_only: tau_g_excl_baseline Normal(0,2)", mo, dict(tau_g_excl_baseline=tfd.Normal(f(0), f(2))))
case("media_only: roi_m LogNormal(0.5,0.6)", mo, dict(roi_m=tfd.LogNormal(f(0.5), f(0.6))))
case("media_and_rf: alpha_rf Beta(2,2)", mr, dict(alpha_rf=tfd.Beta(f(2), f(2))))
case("media_and_rf: roi_rf LogNormal(0.5,0.6)+roi_m", mr, dict(roi_rf=tfd.LogNormal(f(0.5), f(0.6)), roi_m=tfd.LogNormal(f(0.4), f(0.7))))
