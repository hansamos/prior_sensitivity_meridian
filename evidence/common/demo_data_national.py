"""National InputData: the 40-geo simulated dataset aggregated to one national series (156 weeks).

Rules: sum across geos for KPI, media impressions, media spend, organic impressions and population;
population-weighted mean for the controls and the Promo indicator; KPI-weighted mean for revenue per conversion.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")

import numpy as np
import pandas as pd
from meridian.data import data_frame_input_data_builder

DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/"
    "refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv"
)
CHANNELS = [f"Channel{i}" for i in range(5)]
SUM_COLS = (["conversions", "population", "Organic_channel0_impression"]
            + [f"{c}_impression" for c in CHANNELS] + [f"{c}_spend" for c in CHANNELS])
POP_MEAN_COLS = ["sentiment_score_control", "competitor_sales_control", "Promo"]


def national_frame():
    df = pd.read_csv(DATA_URL, index_col=0)
    df["time"] = df["time"].astype(str)
    out = []
    for t, g in df.groupby("time", sort=True):
        row = {"time": t, "geo": "national"}
        for c in SUM_COLS:
            row[c] = g[c].sum()
        for c in POP_MEAN_COLS:
            row[c] = np.average(g[c], weights=g["population"])
        row["revenue_per_conversion"] = np.average(g["revenue_per_conversion"], weights=g["conversions"])
        out.append(row)
    return pd.DataFrame(out).sort_values("time").reset_index(drop=True)


def build_national_data():
    df = national_frame()
    b = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue", default_kpi_column="conversions",
        default_revenue_per_kpi_column="revenue_per_conversion")
    b = (b.with_kpi(df).with_revenue_per_kpi(df).with_population(df)
         .with_controls(df, control_cols=["sentiment_score_control", "competitor_sales_control"])
         .with_media(df, media_cols=[f"{c}_impression" for c in CHANNELS],
                     media_spend_cols=[f"{c}_spend" for c in CHANNELS], media_channels=CHANNELS)
         .with_non_media_treatments(df, non_media_treatment_cols=["Promo"])
         .with_organic_media(df, organic_media_cols=["Organic_channel0_impression"],
                             organic_media_channels=["Organic_channel0"]))
    return b.build()
