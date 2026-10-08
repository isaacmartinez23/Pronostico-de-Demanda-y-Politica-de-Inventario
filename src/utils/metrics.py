"""Métricas de pronóstico: MAE, WAPE, sesgo, RMSSE y WRMSSE.

El WRMSSE sigue la definición de la competencia M5 (RMSSE por serie, escalado por el error
de un naive de un paso en entrenamiento, ponderado por el ingreso de los últimos 28 días),
pero sobre la jerarquía del subconjunto: total → departamento → SKU. Por eso el valor no es
comparable con el leaderboard de Kaggle (que usa 12 niveles sobre las 30k series); sí lo es
entre los modelos de este proyecto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS: dict[str, list[str]] = {
    "total": [],
    "dept": ["dept_id"],
    "sku": ["unique_id"],
}


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error absoluto total sobre demanda total (robusto a series con ceros)."""
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(np.asarray(y) - np.asarray(yhat))) / denom) if denom > 0 else np.nan


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Sesgo relativo: >0 sobre-pronostica, <0 sub-pronostica."""
    denom = np.sum(y)
    return float(np.sum(np.asarray(yhat) - np.asarray(y)) / denom) if denom > 0 else np.nan


def naive_scale(y: np.ndarray) -> float:
    """Denominador del RMSSE: MSE del naive de un paso, desde la primera venta no nula."""
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y)
    if len(nz) == 0:
        return np.nan
    y = y[nz[0]:]
    if len(y) < 2:
        return np.nan
    scale = np.mean(np.diff(y) ** 2)
    return float(scale) if scale > 0 else np.nan


def rmsse(y: np.ndarray, yhat: np.ndarray, scale: float) -> float:
    if not np.isfinite(scale):
        return np.nan
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2) / scale))


def _aggregate(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Suma las series al nivel `keys` (por fecha) y devuelve formato largo con id `_series`."""
    agg = df.groupby([*keys, "ds"], observed=True)[value_cols].sum().reset_index()
    if keys:
        agg["_series"] = agg[keys].astype(str).agg("|".join, axis=1)
    else:
        agg["_series"] = "total"
    return agg


def rmsse_by_level(
    train: pd.DataFrame, cv: pd.DataFrame, model_cols: list[str], keys: list[str]
) -> pd.DataFrame:
    """RMSSE y peso (ingreso de los últimos 28 días de train) por serie del nivel `keys`.

    train: unique_id, ds, y, sell_price (+ columnas de `keys`)
    cv   : unique_id, ds, y, <model_cols> (+ columnas de `keys`) de UNA ventana
    """
    train = train.assign(_rev=train["y"] * train["sell_price"])
    tr = _aggregate(train, keys, ["y", "_rev"])
    last28 = tr["ds"] > tr["ds"].max() - pd.Timedelta(days=28)

    scale = tr.sort_values("ds").groupby("_series")["y"].apply(lambda s: naive_scale(s.to_numpy()))
    weight = tr[last28].groupby("_series")["_rev"].sum()

    te = _aggregate(cv, keys, ["y", *model_cols])
    rows = {}
    for m in model_cols:
        mse = ((te["y"] - te[m]) ** 2).groupby(te["_series"]).mean()
        rows[m] = np.sqrt(mse / scale.reindex(mse.index))
    out = pd.DataFrame(rows)
    out["weight"] = weight.reindex(out.index).fillna(0.0)
    return out


def wrmsse(
    train: pd.DataFrame,
    cv: pd.DataFrame,
    model_cols: list[str],
    levels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """WRMSSE por nivel y global (promedio simple entre niveles) para una ventana.

    Devuelve un DataFrame indexado por nivel (+ 'WRMSSE') con una columna por modelo.
    """
    levels = levels or LEVELS
    res = {}
    for name, keys in levels.items():
        tab = rmsse_by_level(train, cv, model_cols, keys).dropna(subset=model_cols)
        w = tab["weight"] / tab["weight"].sum()
        res[name] = tab[model_cols].mul(w, axis=0).sum()
    out = pd.DataFrame(res).T
    out.loc["WRMSSE"] = out.mean()
    return out


def per_series_metrics(train: pd.DataFrame, cv: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Métricas por SKU y modelo para una ventana (formato largo)."""
    scale = train.sort_values("ds").groupby("unique_id")["y"].apply(
        lambda s: naive_scale(s.to_numpy())
    )
    g = cv.groupby("unique_id", observed=True)
    demand = g["y"].sum()
    frames = []
    for m in model_cols:
        err = cv[m] - cv["y"]
        by = err.groupby(cv["unique_id"], observed=True)
        frames.append(
            pd.DataFrame(
                {
                    "model": m,
                    "demand": demand,
                    "abs_err": by.apply(lambda e: e.abs().sum()),
                    "err": by.sum(),
                    "mae": by.apply(lambda e: e.abs().mean()),
                    "rmsse": np.sqrt(by.apply(lambda e: (e**2).mean()) / scale.reindex(demand.index)),
                }
            ).reset_index()
        )
    return pd.concat(frames, ignore_index=True)
