import os, json
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MERIDIAN_BACKEND"] = "jax"
import jax
import numpy as np
import pandas as pd
from meridian.data import data_frame_input_data_builder
from meridian.model import model, spec
import tensorflow_probability.substrates.jax as tfp_jax

URL = ("https://raw.githubusercontent.com/google/meridian/refs/heads/main/"
       "meridian/data/simulated_data/csv/geo_all_channels.csv")
df_full = pd.read_csv(URL, index_col=0)
df_full["time"] = df_full["time"].astype(str)
geos = sorted(df_full.geo.unique())[:3]
times = sorted(df_full.time.unique())[-80:]
df = df_full[df_full.geo.isin(geos) & df_full.time.isin(times)]
df = df.sort_values(["geo", "time"]).reset_index(drop=True)
channels = [f"Channel{i}" for i in range(5)]
b = data_frame_input_data_builder.DataFrameInputDataBuilder(
    kpi_type="non_revenue", default_kpi_column="conversions",
    default_revenue_per_kpi_column="revenue_per_conversion")
b = (b.with_kpi(df).with_revenue_per_kpi(df).with_population(df)
     .with_controls(df, control_cols=["sentiment_score_control", "competitor_sales_control"])
     .with_media(df, media_cols=[f"{c}_impression" for c in channels],
                 media_spend_cols=[f"{c}_spend" for c in channels],
                 media_channels=channels)
     .with_non_media_treatments(df, non_media_treatment_cols=["Promo"])
     .with_organic_media(df, organic_media_cols=["Organic_channel0_impression"],
                         organic_media_channels=["Organic_channel0"]))
data = b.build()
print("InputData built")

PARAM_KEYS = ("loc", "scale", "low", "high", "concentration1", "concentration0", "rate")


def describe(tag, **kw):
    m = model.Meridian(input_data=data, model_spec=spec.ModelSpec(**kw))
    ms = m.model_spec
    jd = m.posterior_sampler_callable._get_joint_dist_unpinned()
    dists, sample = jd.sample_distributions(seed=jax.random.PRNGKey(0))
    names = jd._flat_resolve_names()
    dl = jax.tree_util.tree_leaves(dists, is_leaf=lambda x: isinstance(x, tfp_jax.distributions.Distribution))
    vl = jax.tree_util.tree_leaves(sample)
    rows = []
    for n, d, v in zip(names, dl, vl):
        inner, chain = d, []
        while hasattr(inner, "distribution") and inner.distribution is not inner:
            chain.append(type(inner).__name__)
            inner = inner.distribution
        chain.append(type(inner).__name__)
        params = {}
        for k in PARAM_KEYS:
            val = inner.parameters.get(k)
            if val is not None:
                params[k] = np.asarray(val).tolist()
        rows.append(dict(
            name=n, wrappers=">".join(chain), batch=list(d.batch_shape),
            event=list(d.event_shape), value_shape=list(np.shape(v)),
            deterministic=isinstance(inner, tfp_jax.distributions.Deterministic),
            params=params))
    print(f"\n===== {tag}: {kw}")
    print(" effective types: media", ms.effective_media_prior_type, "| rf",
          ms.effective_rf_prior_type, "| om", ms.organic_media_prior_type,
          "| n", ms.non_media_treatments_prior_type)
    for r in rows:
        flag = "DET " if r["deterministic"] else "RV  "
        extra = "" if r["deterministic"] else r["params"]
        print(f"  {flag}{r['name']:22s} value={str(r['value_shape']):14s} "
              f"batch={str(r['batch']):10s} event={str(r['event']):6s} {r['wrappers']} {extra}")
    return rows


if __name__ == "__main__":
    out = {}
    out["roi_notebook"] = describe("notebook config", enable_aks=False, media_prior_type="roi")
    out["mroi"] = describe("mroi", enable_aks=False, media_prior_type="mroi")
    out["contribution"] = describe("contribution", enable_aks=False, media_prior_type="contribution")
    out["coefficient"] = describe(
        "coefficient", enable_aks=False, media_prior_type="coefficient",
        organic_media_prior_type="coefficient",
        non_media_treatments_prior_type="coefficient")
    out["roi_aks"] = describe("roi + aks", enable_aks=True, media_prior_type="roi")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "introspection.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
