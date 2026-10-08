"""Evaluación del backtesting: WRMSSE por ventana y métricas por segmento ABC/XYZ.

Uso: python -m src.models.backtest

Une los pronósticos de baselines y LightGBM y los evalúa ventana por ventana, usando como
entrenamiento solo lo anterior a cada corte. Cada fila lleva `split`:
  - "eval" : las N_WINDOWS ventanas que se reportan.
  - "calib": las ventanas previas (solo modelos baratos y LightGBM); son las únicas que se
             pueden usar para tomar decisiones de modelado sin contaminar el reporte.

Salidas (data/processed/):
  - metrics_wrmsse.parquet : cutoff, split, level (total/dept/sku/WRMSSE), model, value
  - metrics_sku.parquet    : métricas por SKU, modelo y ventana
  - metrics_segment.parquet: MAE, WAPE, sesgo y RMSSE por modelo y segmento
  - metrics_coverage.parquet: cobertura fuera de muestra de los intervalos de LightGBM
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.models import intervals
from src.utils import metrics

GROUPINGS = ("abc", "xyz", "segment", "pattern", "dept_id")


def load_cv() -> pd.DataFrame:
    """Backtesting de todos los modelos en una tabla (NaN donde un modelo no corrió la ventana)."""
    base = pd.read_parquet(config.CV_BASELINES)
    lgbm = pd.read_parquet(config.CV_LGBM).drop(columns="y")
    return base.merge(lgbm, on=["unique_id", "ds", "cutoff"], how="outer")


def model_columns(cv: pd.DataFrame) -> list[str]:
    return [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y", "dept_id")]


def split_of(cutoffs, n_eval: int = config.N_WINDOWS) -> dict:
    """Asigna 'eval' a los últimos `n_eval` cortes y 'calib' al resto."""
    cutoffs = sorted(cutoffs)
    return {c: "eval" if i >= len(cutoffs) - n_eval else "calib" for i, c in enumerate(cutoffs)}


def evaluate(sales: pd.DataFrame, cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (wrmsse largo, métricas por SKU) para todas las ventanas.

    sales: unique_id, ds, y, sell_price, dept_id
    cv   : unique_id, ds, cutoff, y + una columna por modelo
    """
    dept = sales.drop_duplicates("unique_id").set_index("unique_id")["dept_id"]
    cv = cv.assign(dept_id=cv["unique_id"].map(dept))
    splits = split_of(cv["cutoff"].unique())

    wr_frames, sku_frames = [], []
    for cutoff, win in cv.groupby("cutoff", sort=True):
        models = [m for m in model_columns(cv) if win[m].notna().all()]
        train = sales[sales["ds"] <= cutoff]
        key = {"cutoff": cutoff, "split": splits[cutoff]}

        wr = metrics.wrmsse(train, win, models)
        wr = wr.rename_axis("level").reset_index().melt("level", var_name="model")
        wr_frames.append(wr.assign(**key))
        sku_frames.append(metrics.per_series_metrics(train, win, models).assign(**key))

    return pd.concat(wr_frames, ignore_index=True), pd.concat(sku_frames, ignore_index=True)


def segment_table(by_sku: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Agrega las métricas por SKU a cada segmento (y al total), por split y modelo.

    WAPE y sesgo se calculan sobre sumas (ponderan por volumen); MAE y RMSSE son el
    promedio simple entre SKUs.
    """
    df = by_sku.merge(classes, on="unique_id", how="left").assign(total="Total")
    frames = []
    for grouping in ("total", *GROUPINGS):
        g = df.groupby(["split", "model", grouping], observed=True)
        tab = g.agg(
            n_skus=("unique_id", "nunique"),
            demand=("demand", "sum"),
            abs_err=("abs_err", "sum"),
            err=("err", "sum"),
            mae=("mae", "mean"),
            rmsse=("rmsse", "mean"),
        ).reset_index()
        tab = tab.rename(columns={grouping: "group"}).assign(grouping=grouping)
        frames.append(tab)
    out = pd.concat(frames, ignore_index=True)
    out["wape"] = out["abs_err"] / out["demand"]
    out["bias"] = out["err"] / out["demand"]
    cols = ["split", "grouping", "group", "model", "n_skus", "demand", "mae", "wape", "bias", "rmsse"]
    return out[cols]


def interval_coverage(
    cv: pd.DataFrame, classes: pd.DataFrame, model: str = "LightGBM", n_eval: int = config.N_WINDOWS
) -> pd.DataFrame:
    """Cobertura de los intervalos conformales medida FUERA de muestra.

    Los márgenes se calibran con los residuos de las ventanas de calibración y se evalúan
    en las de evaluación, que no intervinieron. Devuelve cobertura y ancho medio por segmento.
    """
    split = cv["cutoff"].map(split_of(cv["cutoff"].unique(), n_eval))
    cols = ["unique_id", "ds", "cutoff", "y", model]
    quantiles = intervals.residual_quantiles(cv.loc[split == "calib", cols], model)
    ev = intervals.add_intervals(cv.loc[split == "eval", cols], quantiles, model)
    ev = ev.merge(classes, on="unique_id", how="left").assign(total="Total")

    rows = []
    for grouping in ("total", "abc", "xyz", "pattern"):
        for group, g in ev.groupby(grouping, observed=True):
            cov = intervals.coverage(g, model)
            for lv in intervals.LEVELS:
                width = (g[f"{model}-hi-{lv}"] - g[f"{model}-lo-{lv}"]).mean()
                rows.append(
                    {"grouping": grouping, "group": group, "level": lv, "nominal": lv / 100,
                     "coverage": cov[lv], "mean_width": width, "n_obs": len(g)}
                )
    return pd.DataFrame(rows)


def main() -> None:
    sales = pd.read_parquet(
        config.SALES_LONG, columns=["unique_id", "ds", "y", "sell_price", "dept_id"]
    )
    classes = pd.read_parquet(config.ABC_XYZ)[["unique_id", "abc", "xyz", "segment", "pattern"]]
    classes = classes.merge(sales[["unique_id", "dept_id"]].drop_duplicates(), on="unique_id")

    cv = load_cv()
    wr, by_sku = evaluate(sales, cv)
    seg = segment_table(by_sku, classes)
    cov = interval_coverage(cv, classes)
    cov.to_parquet(config.METRICS_COVERAGE, index=False)
    wr.to_parquet(config.METRICS_WRMSSE, index=False)
    by_sku.to_parquet(config.METRICS_SKU, index=False)
    seg.to_parquet(config.METRICS_SEGMENT, index=False)

    pd.set_option("display.width", 200)
    ev = wr[(wr["split"] == "eval") & (wr["level"] == "WRMSSE")]
    piv = ev.pivot(index="cutoff", columns="model", values="value")
    piv.index = piv.index.strftime("%Y-%m-%d")
    piv.loc["promedio"] = piv.mean()
    print("WRMSSE por ventana de evaluación (menor es mejor):")
    print(piv.round(4).to_string())

    for grouping in ("total", "abc", "pattern"):
        view = seg[(seg["split"] == "eval") & (seg["grouping"] == grouping)]
        print(f"\nWAPE por {grouping}:")
        print(view.pivot(index="group", columns="model", values="wape").round(3).to_string())
    view = seg[(seg["split"] == "eval") & (seg["grouping"] == "total")]
    print("\nSesgo total (>0 sobre-pronostica):")
    print(view.set_index("model")["bias"].round(3).to_string())

    print("\nCobertura de los intervalos de LightGBM (calibrados en ventanas previas):")
    view = cov[cov["grouping"].isin(["total", "abc"])]
    print(view.pivot(index="group", columns="level", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
