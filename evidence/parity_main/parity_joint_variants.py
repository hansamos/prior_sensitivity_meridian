import warnings; warnings.filterwarnings("ignore")
import numpy as np
from meridian import backend, constants
from meridian.model import model, model_test_data, spec, prior_distribution

tfd = backend.tfd; f = backend.np_float_dtype
model_test_data.WithInputDataSamples.setup()
inst = model_test_data.WithInputDataSamples()
N = 40
JAX = backend.config.get_backend() == backend.config.Backend.JAX
TOL = 1e-9 if JAX else 5e-3

def build(data, spec_kw, **prior):
    ms = spec.ModelSpec(prior=prior_distribution.PriorDistribution(**prior), **spec_kw)
    return model.Meridian(input_data=data, model_spec=ms)

def case(label, data, spec_kw, targets):
    try:
        m_ref, m_tgt = build(data, spec_kw), build(data, spec_kw, **targets)
        pr, pt = m_ref.model_context.prior_broadcast, m_tgt.model_context.prior_broadcast
        jr = m_ref.posterior_sampler_callable._get_joint_dist_sampling_unpinned()
        jt = m_tgt.posterior_sampler_callable._get_joint_dist_sampling_unpinned()
        st = jr.sample(N, seed=backend.random.prng_key(11) if JAX else 11)
        state = st._asdict()
        a = np.asarray(jt.log_prob(st), dtype=np.float64) - np.asarray(jr.log_prob(st), dtype=np.float64)
        b = 0.0
        for fld in targets:
            x = state[fld]
            d = np.asarray(getattr(pt, fld).log_prob(x), dtype=np.float64) - np.asarray(getattr(pr, fld).log_prob(x), dtype=np.float64)
            b = b + d.reshape(N, -1).sum(axis=1)
        err = np.max(np.abs(a - b))
        ok = np.isfinite(a).all() and err < TOL
        print(f"[{'OK ' if ok else 'BAD'}] {label:62s} err={err:.1e} sd={a.std():.2f}")
    except Exception as e:
        print(f"[ERR] {label:62s} {type(e).__name__}: {str(e)[:150]}")

d = inst
mo, mr = d.short_input_data_with_media_only, d.short_input_data_with_media_and_rf
nat = d.national_input_data_media_only
nmo = d.short_input_data_non_media
org = d.input_data_non_media_and_organic
nrev = d.input_data_non_revenue_no_revenue_per_kpi
R, C, K = constants.TREATMENT_PRIOR_TYPE_ROI, constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION, constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
M = constants.TREATMENT_PRIOR_TYPE_MROI
LN = lambda m, s: tfd.LogNormal(f(m), f(s))

case("contribution: contribution_m Beta(2,50)", mo, dict(media_prior_type=C), dict(contribution_m=tfd.Beta(f(2), f(50))))
case("contribution: alpha_m Beta(2,2) + contribution_m", mo, dict(media_prior_type=C), dict(alpha_m=tfd.Beta(f(2), f(2)), contribution_m=tfd.Beta(f(2), f(50))))
case("coefficient: beta_m LogNormal(0.1,0.6)", mo, dict(media_prior_type=K), dict(beta_m=LN(0.1, 0.6)))
case("mroi: mroi_m LogNormal(0.1,0.6)", mo, dict(media_prior_type=M), dict(mroi_m=LN(0.1, 0.6)))
case("media_effects_dist normal: eta_m HalfNormal(0.5)", mo, dict(media_prior_type=R, media_effects_dist=constants.MEDIA_EFFECTS_NORMAL), dict(eta_m=tfd.HalfNormal(f(0.5))))
case("media_effects_dist normal: roi_m", mo, dict(media_prior_type=R, media_effects_dist=constants.MEDIA_EFFECTS_NORMAL), dict(roi_m=LN(0.5, 0.6)))
case("unique_sigma_for_each_geo: sigma HalfNormal(2)", mo, dict(media_prior_type=R, unique_sigma_for_each_geo=True), dict(sigma=tfd.HalfNormal(f(2.0))))
case("hill_before_adstock: alpha_m Beta(2,2)", mo, dict(media_prior_type=R, hill_before_adstock=True), dict(alpha_m=tfd.Beta(f(2), f(2))))
case("enable_aks: knot_values Normal(0,2)", mo, dict(media_prior_type=R, enable_aks=True), dict(knot_values=tfd.Normal(f(0), f(2))))
case("national: roi_m", nat, dict(media_prior_type=R), dict(roi_m=LN(0.5, 0.6)))
case("national: alpha_m + ec_m", nat, dict(media_prior_type=R), dict(alpha_m=tfd.Beta(f(2), f(2)), ec_m=tfd.TruncatedNormal(f(1.0), f(0.8), f(0.1), f(10.0))))
case("national: sigma", nat, dict(media_prior_type=R), dict(sigma=tfd.HalfNormal(f(2.0))))
case("non_media: xi_n HalfNormal(2)", nmo, dict(media_prior_type=R), dict(xi_n=tfd.HalfNormal(f(2.0))))
case("non_media: contribution_n TN(0,0.2,-1,1)", nmo, dict(media_prior_type=R), dict(contribution_n=tfd.TruncatedNormal(f(0), f(0.2), f(-1), f(1))))
case("organic: contribution_om Beta(2,50)", org, dict(media_prior_type=R), dict(contribution_om=tfd.Beta(f(2), f(50))))
case("organic: alpha_om Beta(2,2) + ec_om", org, dict(media_prior_type=R), dict(alpha_om=tfd.Beta(f(2), f(2)), ec_om=tfd.TruncatedNormal(f(1.0), f(0.8), f(0.1), f(10.0))))
case("organic: eta_om HalfNormal(0.5)", org, dict(media_prior_type=R), dict(eta_om=tfd.HalfNormal(f(0.5))))
case("non_revenue, no rev_per_kpi (default roi prior substituted): alpha_m", nrev, dict(), dict(alpha_m=tfd.Beta(f(2), f(2))))
case("non_revenue, no rev_per_kpi: roi_m custom vs substituted default", nrev, dict(), dict(roi_m=LN(0.5, 0.6)))
case("rf: ec_rf + slope_rf stochastic", mr, dict(media_prior_type=R, rf_prior_type=R), dict(ec_rf=tfd.TruncatedNormal(f(1.0), f(0.8), f(0.1), f(10.0)), slope_rf=LN(0.6, 0.3)))
