"""Ablaciones del modelo global, evaluadas SOLO en las ventanas de calibración.

Uso: python -m src.models.ablation

Responde qué aporta cada grupo de variables y si el objetivo Tweedie se justifica, sin
tocar las ventanas de evaluación: los datos se cortan antes de que empiecen, de modo que
ninguna decisión de modelado se toma mirando el periodo que se reporta.
"""

from __future__ import annotations

import time

import pandas as pd

from src import config
from src.features import build_features as bf
from src.models import ml_model
from src.utils import metrics

PRICE = ["sell_price", "price_rel_item", "price_rel_dept", "price_change_7"]
CALENDAR = ["snap", "is_event", "event_name", "event_type", "days_to_event"]

ITEM_ID = "item_id"  # se agrega como categórica solo en la variante que la prueba

# nombre → (columnas que se quitan, columnas estáticas que se agregan, cambios a los hiperparámetros)
VARIANTS: dict[str, tuple[list[str], list[str], dict]] = {
    "base": ([], [], {}),
    "con item_id": ([], [ITEM_ID], {}),
    "sin precio": (PRICE, [], {}),
    "sin calendario": (CALENDAR, [], {}),
    "objetivo Poisson": ([], [], {"objective": "poisson", "tweedie_variance_power": None}),
}


def run_variant(
    train: pd.DataFrame, drop: list[str], add: list[str], overrides: dict, n_windows: int
) -> pd.DataFrame:
    params = {k: v for k, v in {**ml_model.LGBM_PARAMS, **overrides}.items() if v is not None}
    static = [*bf.STATIC_FEATURES, *add]
    cv = ml_model.make_forecaster(params).cross_validation(
        df=train.drop(columns=drop + [c for c in [ITEM_ID] if c not in add]),
        n_windows=n_windows,
        h=config.HORIZON,
        step_size=config.HORIZON,
        static_features=static,
        refit=True,
    )
    cv[ml_model.MODEL] = cv[ml_model.MODEL].clip(lower=0)
    return cv[["unique_id", "ds", "cutoff", "y", ml_model.MODEL]]


def score(sales: pd.DataFrame, cv: pd.DataFrame) -> dict:
    """WRMSSE promedio entre ventanas, y WAPE / sesgo sobre el total."""
    dept = sales.drop_duplicates("unique_id").set_index("unique_id")["dept_id"]
    cv = cv.assign(dept_id=cv["unique_id"].map(dept))
    wr = [
        metrics.wrmsse(sales[sales["ds"] <= cutoff], win, [ml_model.MODEL]).loc["WRMSSE", ml_model.MODEL]
        for cutoff, win in cv.groupby("cutoff")
    ]
    return {
        "wrmsse": sum(wr) / len(wr),
        "wape": metrics.wape(cv["y"], cv[ml_model.MODEL]),
        "bias": metrics.bias(cv["y"], cv[ml_model.MODEL]),
    }


def main() -> None:
    sales = pd.read_parquet(config.SALES_LONG)
    eval_start = sales["ds"].max() - pd.Timedelta(days=config.N_WINDOWS * config.HORIZON)
    sales = sales[sales["ds"] <= eval_start]  # las ventanas de evaluación quedan fuera
    train, _ = bf.make_model_frames(sales)
    item = sales.drop_duplicates("unique_id").set_index("unique_id")["item_id"]
    train[ITEM_ID] = train["unique_id"].map(item).astype("category")

    rows = []
    for name, (drop, add, overrides) in VARIANTS.items():
        t0 = time.time()
        cv = run_variant(train, drop, add, overrides, config.N_CALIB_WINDOWS)
        rows.append({"variant": name, **score(sales, cv), "seconds": time.time() - t0})
        print(f"{name}: {rows[-1]['wrmsse']:.4f} ({rows[-1]['seconds']:.0f}s)")

    out = pd.DataFrame(rows)
    out["wrmsse_vs_base"] = out["wrmsse"] / out.loc[out["variant"] == "base", "wrmsse"].iloc[0] - 1
    out.to_parquet(config.ABLATION, index=False)
    print(f"\nAblaciones en {config.N_CALIB_WINDOWS} ventanas de calibración (hasta {eval_start.date()}):")
    print(out.drop(columns="seconds").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
