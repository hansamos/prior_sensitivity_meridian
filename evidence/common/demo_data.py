"""Builds the notebook's cpu_pilot InputData (3 geos x 80 weeks) for Phase 1 checks."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")

import pandas as pd
from meridian.data import data_frame_input_data_builder

DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/"
    "refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv"
)
CHANNELS = [f"Channel{i}" for i in range(5)]


def build_data(n_geos=3, n_times=80):
    df_full = pd.read_csv(DATA_URL, index_col=0)
    df_full["time"] = df_full["time"].astype(str)
    geos = sorted(df_full.geo.unique())
    times = sorted(df_full.time.unique())
    if n_geos is not None:
        geos = geos[:n_geos]
    if n_times is not None:
        times = times[-n_times:]
    df = df_full[df_full.geo.isin(geos) & df_full.time.isin(times)]
    df = df.sort_values(["geo", "time"]).reset_index(drop=True)
    b = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column="conversions",
        default_revenue_per_kpi_column="revenue_per_conversion",
    )
    b = (
        b.with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(
            df, control_cols=["sentiment_score_control", "competitor_sales_control"]
        )
        .with_media(
            df,
            media_cols=[f"{c}_impression" for c in CHANNELS],
            media_spend_cols=[f"{c}_spend" for c in CHANNELS],
            media_channels=CHANNELS,
        )
        .with_non_media_treatments(df, non_media_treatment_cols=["Promo"])
        .with_organic_media(
            df,
            organic_media_cols=["Organic_channel0_impression"],
            organic_media_channels=["Organic_channel0"],
        )
    )
    return b.build()
