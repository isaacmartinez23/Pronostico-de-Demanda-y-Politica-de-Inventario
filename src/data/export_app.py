"""Exporta a app/data/ lo mínimo que necesita la app de Streamlit.

Uso: python -m src.data.export_app

La app no entrena ni lee data/processed (que no se sube a git): solo consume estos parquet
livianos, que sí se versionan para poder desplegar en Streamlit Community Cloud.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.inventory import policy as pol

MODEL = "LightGBM"
HISTORY_DAYS = 182


def sku_table() -> pd.DataFrame:
    """Una fila por SKU: clasificación, pronóstico, dispersión del error por lead time y desempeño."""
    table = pd.read_parquet(config.POLICY_TABLE)
    cv = pd.read_parquet(config.CV_LGBM)
    future = pd.read_parquet(config.FORECAST_FUTURE).sort_values(["unique_id", "ds"])
    ids = list(table["unique_id"])

    resid = pol.cv_cube(cv, "y", ids) - pol.cv_cube(cv, MODEL, ids)
    by_sku = future.groupby("unique_id")[MODEL]
    for lt in config.LEAD_TIMES:
        table[f"sigma_lead_{lt}"] = pol.lead_time_errors(resid, lt).std(axis=0, ddof=1)
        demand = by_sku.apply(lambda s, lt=lt: s.iloc[:lt].sum())
        table[f"demand_lead_{lt}"] = table["unique_id"].map(demand)

    metrics = pd.read_parquet(config.METRICS_SKU).query("split == 'eval' and model == @MODEL")
    perf = metrics.groupby("unique_id")[["demand", "abs_err", "err"]].sum()
    table["wape_backtest"] = table["unique_id"].map(perf["abs_err"] / perf["demand"].where(perf["demand"] > 0))
    table["bias_backtest"] = table["unique_id"].map(perf["err"] / perf["demand"].where(perf["demand"] > 0))

    sim = pd.read_parquet(config.SIM_BY_SKU).query("policy == @MODEL and ss_method == 'sqrt'")
    table["fill_rate_sim"] = table["unique_id"].map(sim.set_index("unique_id")["fill_rate"])

    revenue = pd.read_parquet(config.ABC_XYZ).set_index("unique_id")["revenue"]
    table["revenue_365d"] = table["unique_id"].map(revenue)
    drop = ["safety_stock", "rop", "order_up_to", "lead_time", "service_level", "ss_method",
            "demand_lead_time", "sigma_lead"]
    return table.drop(columns=drop).sort_values("revenue_365d", ascending=False, ignore_index=True)


def main() -> None:
    config.ensure_dirs()
    out = config.APP_DATA_DIR

    sales = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "ds", "y"])
    recent = sales[sales["ds"] > sales["ds"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    recent.astype({"y": "float32"}).to_parquet(out / "history.parquet", index=False)

    cv = pd.read_parquet(config.CV_LGBM, columns=["unique_id", "ds", "cutoff", MODEL])
    eval_cutoffs = sorted(cv["cutoff"].unique())[-config.N_WINDOWS :]
    back = cv.loc[cv["cutoff"].isin(eval_cutoffs), ["unique_id", "ds", MODEL]]
    back.astype({MODEL: "float32"}).to_parquet(out / "backtest.parquet", index=False)

    future = pd.read_parquet(config.FORECAST_FUTURE)
    num = future.select_dtypes("number").columns
    future.astype(dict.fromkeys(num, "float32")).to_parquet(out / "forecast.parquet", index=False)

    sku_table().to_parquet(out / "skus.parquet", index=False)

    for src in (config.SIM_FRONTIER, config.SIM_SCENARIOS, config.SIM_COSTS, config.METRICS_COVERAGE,
                config.ABLATION):
        pd.read_parquet(src).to_parquet(out / src.name, index=False)
    for src in (config.METRICS_WRMSSE, config.METRICS_SEGMENT):
        pd.read_parquet(src).query("split == 'eval'").to_parquet(out / src.name, index=False)

    size = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
    print(f"{out}: {len(list(out.glob('*.parquet')))} archivos, {size:.1f} MB")


if __name__ == "__main__":
    main()
