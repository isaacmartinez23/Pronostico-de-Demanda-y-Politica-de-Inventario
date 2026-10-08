"""Feature engineering para el modelo global.

Dos familias de variables:
  - Exógenas conocidas a futuro (calendario, eventos, SNAP, precio): se calculan aquí.
  - Autorregresivas (lags y medias móviles de `y`): las calcula mlforecast a partir de
    LAGS / lag_transforms(), tanto en entrenamiento como de forma recursiva al predecir.

En M5 el calendario y los precios del horizonte se conocen por adelantado, así que las
exógenas no filtran información de la demanda futura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

LAGS = [7, 14, 28]
DATE_FEATURES = ["dayofweek", "day", "month"]
# `item_id` (1,429 niveles) quedó fuera: en las ventanas de calibración el modelo generaliza
# mejor sin él (ver src/models/ablation.py). El nivel de cada SKU ya lo dan sus medias móviles.
STATIC_FEATURES = ["dept_id"]
DYNAMIC_FEATURES = [
    "snap",
    "is_event",
    "event_name",
    "event_type",
    "days_to_event",
    "sell_price",
    "price_rel_item",
    "price_rel_dept",
    "price_change_7",
]
CATEGORICAL = ["dept_id", "event_name", "event_type"]

MAX_DAYS_TO_EVENT = 7
PRICE_WINDOW = 91


def lag_transforms() -> dict:
    """Medias y desviación móviles sobre los lags (ventanas que terminan en t-lag)."""
    return {
        7: [RollingMean(window_size=7), RollingMean(window_size=28)],
        14: [RollingMean(window_size=14)],
        28: [RollingMean(window_size=28), RollingStd(window_size=28)],
    }


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Eventos y cercanía al próximo evento (la gente compra *antes* del feriado)."""
    out = df.copy()
    out["is_event"] = out["event_name_1"].notna().astype("int8")
    out["event_name"] = out["event_name_1"].fillna("none")
    out["event_type"] = out["event_type_1"].fillna("none")

    cal = out[["ds", "is_event"]].drop_duplicates("ds").sort_values("ds")
    next_event = cal["ds"].where(cal["is_event"] == 1).bfill()
    days = (next_event - cal["ds"]).dt.days
    cal["days_to_event"] = days.clip(upper=MAX_DAYS_TO_EVENT).fillna(MAX_DAYS_TO_EVENT).astype("int8")
    out["days_to_event"] = out["ds"].map(cal.set_index("ds")["days_to_event"])
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Precio relativo: contra la historia reciente del propio SKU y contra su departamento."""
    out = df.sort_values(["unique_id", "ds"]).copy()
    by_item = out.groupby("unique_id", sort=False)["sell_price"]

    trailing_mean = by_item.transform(lambda s: s.rolling(PRICE_WINDOW, min_periods=1).mean())
    out["price_rel_item"] = out["sell_price"] / trailing_mean

    dept_mean = out.groupby(["dept_id", "ds"], sort=False)["sell_price"].transform("mean")
    out["price_rel_dept"] = out["sell_price"] / dept_mean

    prev = by_item.shift(7)
    out["price_change_7"] = (out["sell_price"] / prev - 1).fillna(0.0)
    return out


def add_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las exógenas. `df` puede incluir filas futuras (sin `y`)."""
    out = add_price_features(add_calendar_features(df))
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    float_cols = ["sell_price", "price_rel_item", "price_rel_dept", "price_change_7"]
    out[float_cols] = out[float_cols].astype("float32")
    out["snap"] = out["snap"].astype("int8")
    return out


def make_model_frames(
    sales: pd.DataFrame, future: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Devuelve (train, future) listos para mlforecast.

    Las exógenas se calculan sobre historia + futuro concatenados para que las ventanas
    móviles de precio y las categorías sean consistentes entre ambos.
    """
    cols = ["unique_id", "ds", "y", *STATIC_FEATURES, *DYNAMIC_FEATURES]
    if future is None:
        feats = add_exogenous_features(sales)
        return feats[cols].reset_index(drop=True), None

    full = pd.concat([sales, future.assign(y=np.nan)], ignore_index=True)
    feats = add_exogenous_features(full)
    is_future = feats["ds"] > sales["ds"].max()
    train = feats.loc[~is_future, cols].reset_index(drop=True)
    fut = feats.loc[is_future, ["unique_id", "ds", *DYNAMIC_FEATURES]].reset_index(drop=True)
    return train, fut
