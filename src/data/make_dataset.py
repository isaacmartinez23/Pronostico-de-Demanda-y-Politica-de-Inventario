"""Convierte M5 de formato ancho a largo y une calendario + precios para el subconjunto elegido.

Uso: python -m src.data.make_dataset

Salidas (data/processed/):
  - sales_long.parquet : unique_id, ds, y + atributos del SKU, calendario y precio
  - future_exog.parquet: calendario y precio de los 28 días posteriores al último dato
                         (M5 los publica; son las variables exógenas del pronóstico final)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src import config

CALENDAR_COLS = """
    c.date::DATE AS ds,
    c.wm_yr_wk,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.snap_{state} AS snap
"""


def _state(store_id: str) -> str:
    return store_id.split("_")[0]


def build_long(store_id: str = config.STORE_ID, cat_id: str = config.CAT_ID) -> pd.DataFrame:
    """Melt wide→long con DuckDB (UNPIVOT) y join con calendario y precios."""
    raw = config.RAW_DIR
    con = duckdb.connect()
    query = f"""
        WITH sales AS (
            SELECT * FROM read_csv_auto('{(raw / "sales_train_evaluation.csv").as_posix()}')
            WHERE store_id = '{store_id}' AND cat_id = '{cat_id}'
        ),
        long AS (
            UNPIVOT sales ON COLUMNS('^d_\\d+$') INTO NAME d VALUE y
        ),
        cal AS (
            SELECT * FROM read_csv_auto('{(raw / "calendar.csv").as_posix()}', all_varchar=false)
        ),
        prices AS (
            SELECT * FROM read_csv_auto('{(raw / "sell_prices.csv").as_posix()}')
            WHERE store_id = '{store_id}'
        )
        SELECT
            l.item_id || '_' || l.store_id AS unique_id,
            {CALENDAR_COLS.format(state=_state(store_id))},
            l.y::DOUBLE AS y,
            l.item_id, l.dept_id, l.cat_id, l.store_id,
            p.sell_price
        FROM long l
        JOIN cal c USING (d)
        LEFT JOIN prices p
            ON p.item_id = l.item_id AND p.store_id = l.store_id AND p.wm_yr_wk = c.wm_yr_wk
        ORDER BY unique_id, ds
    """
    df = con.execute(query).df()
    con.close()
    df["ds"] = pd.to_datetime(df["ds"]).astype("datetime64[ns]")

    # Un SKU sin precio todavía no estaba en el surtido: esos ceros no son demanda.
    df = df[df["sell_price"].notna()]
    n_days = df.groupby("unique_id")["ds"].transform("size")
    return df[n_days >= config.MIN_HISTORY_DAYS].reset_index(drop=True)


def build_future_exog(
    sales: pd.DataFrame, store_id: str = config.STORE_ID, horizon: int = config.HORIZON
) -> pd.DataFrame:
    """Calendario y precios de los `horizon` días siguientes al último dato de ventas."""
    raw = config.RAW_DIR
    last = sales["ds"].max()
    ids = sales[["unique_id", "item_id", "dept_id", "cat_id", "store_id"]].drop_duplicates()
    con = duckdb.connect()
    con.register("ids", ids)
    query = f"""
        WITH cal AS (
            SELECT * FROM read_csv_auto('{(raw / "calendar.csv").as_posix()}')
            WHERE date::DATE > DATE '{last.date()}'
              AND date::DATE <= DATE '{last.date()}' + INTERVAL {horizon} DAY
        ),
        prices AS (
            SELECT * FROM read_csv_auto('{(raw / "sell_prices.csv").as_posix()}')
            WHERE store_id = '{store_id}'
        )
        SELECT
            i.unique_id,
            {CALENDAR_COLS.format(state=_state(store_id))},
            i.item_id, i.dept_id, i.cat_id, i.store_id,
            p.sell_price
        FROM ids i
        CROSS JOIN cal c
        LEFT JOIN prices p
            ON p.item_id = i.item_id AND p.store_id = i.store_id AND p.wm_yr_wk = c.wm_yr_wk
        ORDER BY unique_id, ds
    """
    fut = con.execute(query).df()
    con.close()
    fut["ds"] = pd.to_datetime(fut["ds"]).astype("datetime64[ns]")
    # Si faltara algún precio futuro, se asume que se mantiene el último conocido.
    last_price = sales.groupby("unique_id")["sell_price"].last()
    fut["sell_price"] = fut["sell_price"].fillna(fut["unique_id"].map(last_price))
    return fut


def main() -> None:
    config.ensure_dirs()
    sales = build_long()
    sales.to_parquet(config.SALES_LONG, index=False)
    fut = build_future_exog(sales)
    fut.to_parquet(config.FUTURE_EXOG, index=False)

    n = sales["unique_id"].nunique()
    print(
        f"{config.SALES_LONG.name}: {len(sales):,} filas | {n:,} series | "
        f"{sales['ds'].min().date()} → {sales['ds'].max().date()} | "
        f"ceros: {(sales['y'] == 0).mean():.1%}"
    )
    print(f"{config.FUTURE_EXOG.name}: {len(fut):,} filas | "
          f"{fut['ds'].min().date()} → {fut['ds'].max().date()}")


if __name__ == "__main__":
    main()
