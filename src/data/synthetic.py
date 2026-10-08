"""Genera un mini-M5 sintético con el mismo formato crudo que el original.

Uso: DFI_DATA_DIR=/ruta/temporal python -m src.data.synthetic

M5 no se puede redistribuir ni descargar sin credenciales, así que CI corre el pipeline
completo sobre estos datos: ~30 SKUs de FOODS en CA_1 con estacionalidad semanal, efecto
SNAP, eventos, promociones de precio y una mezcla de demanda suave e intermitente.
Sirve para verificar que el código corre de punta a punta, no para sacar conclusiones.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config

START = "2015-01-03"  # sábado, como las semanas de Walmart
EVENTS = {30: ("SuperBowl", "Sporting"), 130: ("Mother's day", "Cultural"), 330: ("Thanksgiving", "National"),
          357: ("Christmas", "National"), 400: ("SuperBowl", "Sporting"), 500: ("Mother's day", "Cultural")}
WEEKLY = np.array([1.35, 1.30, 0.90, 0.85, 0.85, 0.90, 1.05])  # sáb, dom, lun, ...


def make_calendar(n_days: int) -> pd.DataFrame:
    date = pd.date_range(START, periods=n_days)
    cal = pd.DataFrame(
        {
            "date": date.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11501 + np.arange(n_days) // 7,
            "weekday": date.day_name(),
            "wday": np.arange(n_days) % 7 + 1,
            "month": date.month,
            "year": date.year,
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "event_name_1": [EVENTS.get(i, (None, None))[0] for i in range(n_days)],
            "event_type_1": [EVENTS.get(i, (None, None))[1] for i in range(n_days)],
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    snap = (date.day <= 10).astype(int)
    return cal.assign(snap_CA=snap, snap_TX=snap, snap_WI=snap)


def make_items(n_items: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Atributos por SKU: tasa base, precio y día de entrada al surtido."""
    dept = rng.choice(["FOODS_1", "FOODS_2", "FOODS_3"], size=n_items, p=[0.2, 0.3, 0.5])
    smooth = rng.random(n_items) < 0.4
    items = pd.DataFrame(
        {
            "item_id": [f"{d}_{i + 1:03d}" for i, d in enumerate(dept)],
            "dept_id": dept,
            "rate": np.where(smooth, rng.uniform(4, 20, n_items), rng.uniform(0.05, 0.9, n_items)),
            "price": rng.uniform(1, 12, n_items).round(2),
            "start": 0,
        }
    )
    # Dos SKUs entran tarde al surtido; el último tiene menos historia que MIN_HISTORY_DAYS.
    items.loc[items.index[-2], "start"] = 98
    items.loc[items.index[-1], "start"] = n_days - 210
    return items


def make_sales_and_prices(
    items: pd.DataFrame, cal: pd.DataFrame, n_sales_days: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_days = len(cal)
    weeks = cal["wm_yr_wk"].to_numpy()
    week_ids = np.unique(weeks)
    snap = cal["snap_CA"].to_numpy()
    pre_event = np.zeros(n_days)
    for day in EVENTS:
        pre_event[max(day - 3, 0) : day] = 1

    sales, prices = {}, []
    for it in items.itertuples():
        promo = rng.random(len(week_ids)) < 0.08
        price_wk = np.where(promo, it.price * 0.75, it.price).round(2)
        price = price_wk[np.searchsorted(week_ids, weeks)]
        lam = (
            it.rate
            * WEEKLY[np.arange(n_days) % 7]
            * (1 + 0.15 * snap)
            * (1 + 0.30 * pre_event)
            * (price / it.price) ** -2.0
        )
        y = rng.poisson(lam)
        y[: it.start] = 0
        sales[it.item_id] = y[:n_sales_days]

        listed = week_ids >= weeks[it.start]
        prices.append(
            pd.DataFrame(
                {"store_id": "CA_1", "item_id": it.item_id, "wm_yr_wk": week_ids[listed],
                 "sell_price": price_wk[listed]}
            )
        )

    wide = pd.DataFrame.from_dict(sales, orient="index", columns=cal["d"][:n_sales_days])
    wide = wide.rename_axis("item_id").reset_index()
    meta = pd.DataFrame(
        {
            "id": wide["item_id"] + "_CA_1_evaluation",
            "item_id": wide["item_id"],
            "dept_id": items["dept_id"].to_numpy(),
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        }
    )
    return pd.concat([meta, wide.drop(columns="item_id")], axis=1), pd.concat(prices, ignore_index=True)


def generate(n_items: int = 30, n_days: int = 560, seed: int = config.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Devuelve los tres CSV de M5 como DataFrames. El calendario cubre el horizonte futuro."""
    rng = np.random.default_rng(seed)
    cal = make_calendar(n_days + config.HORIZON)
    items = make_items(n_items, n_days, rng)
    sales, prices = make_sales_and_prices(items, cal, n_days, rng)
    return {"calendar.csv": cal, "sales_train_evaluation.csv": sales, "sell_prices.csv": prices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--days", type=int, default=560)
    parser.add_argument("--force", action="store_true", help="sobrescribe CSV existentes")
    args = parser.parse_args()

    config.ensure_dirs()
    existing = [f for f in config.RAW_FILES if (config.RAW_DIR / f).exists()]
    if existing and not args.force:
        raise SystemExit(
            f"Ya hay datos en {config.RAW_DIR} ({', '.join(existing)}). Apunta DFI_DATA_DIR a otra "
            "carpeta para no pisar el M5 real, o usa --force."
        )
    for name, df in generate(args.items, args.days).items():
        df.to_csv(config.RAW_DIR / name, index=False)
    print(f"M5 sintético ({args.items} SKUs × {args.days} días) en {config.RAW_DIR}")


if __name__ == "__main__":
    main()
