"""Intervalos de predicción conformales a partir de los residuos del backtesting.

Para cada SKU se toman los errores fuera de muestra (y - ŷ) de las ventanas de backtesting
y se usan sus cuantiles empíricos como márgenes del intervalo (split conformal por serie,
agrupando los 28 pasos del horizonte). No asume normalidad: útil con demanda intermitente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = (80, 95)


def _conformal_quantile(x: np.ndarray, q: float) -> float:
    """Cuantil con corrección de muestra finita (n+1), acotado a [0, 1]."""
    n = len(x)
    if q >= 0.5:
        level = min(1.0, np.ceil((n + 1) * q) / n)
    else:
        # Simétrico al caso superior: la cola inferior también se corre hacia afuera.
        level = max(0.0, 1.0 - np.ceil((n + 1) * (1 - q)) / n)
    return float(np.quantile(x, level))


def residual_quantiles(
    cv: pd.DataFrame, model_col: str, levels: tuple[int, ...] = LEVELS
) -> pd.DataFrame:
    """Márgenes inferior/superior por SKU: columnas lo-<nivel>, hi-<nivel> (índice unique_id)."""
    resid = (cv["y"] - cv[model_col]).groupby(cv["unique_id"], observed=True)
    out = {}
    for lv in levels:
        alpha = 1 - lv / 100
        out[f"lo-{lv}"] = resid.apply(lambda e, a=alpha: _conformal_quantile(e.to_numpy(), a / 2))
        out[f"hi-{lv}"] = resid.apply(
            lambda e, a=alpha: _conformal_quantile(e.to_numpy(), 1 - a / 2)
        )
    return pd.DataFrame(out)


def add_intervals(
    fcst: pd.DataFrame, quantiles: pd.DataFrame, model_col: str, levels: tuple[int, ...] = LEVELS
) -> pd.DataFrame:
    """Agrega <modelo>-lo-<nivel> / <modelo>-hi-<nivel> al pronóstico (demanda ≥ 0)."""
    out = fcst.copy()
    for lv in levels:
        lo = out["unique_id"].map(quantiles[f"lo-{lv}"])
        hi = out["unique_id"].map(quantiles[f"hi-{lv}"])
        out[f"{model_col}-lo-{lv}"] = (out[model_col] + lo).clip(lower=0)
        out[f"{model_col}-hi-{lv}"] = (out[model_col] + hi).clip(lower=0)
    return out


def coverage(df: pd.DataFrame, model_col: str, levels: tuple[int, ...] = LEVELS) -> dict[int, float]:
    """Cobertura empírica: fracción de observaciones dentro del intervalo."""
    return {
        lv: float(
            ((df["y"] >= df[f"{model_col}-lo-{lv}"]) & (df["y"] <= df[f"{model_col}-hi-{lv}"])).mean()
        )
        for lv in levels
    }
