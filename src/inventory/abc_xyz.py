"""Clasificación ABC (por ingreso) y XYZ (por variabilidad) para priorizar SKUs.

Uso: python -m src.inventory.abc_xyz

  - ABC: A = SKUs que acumulan el 80% del ingreso, B = siguiente 15%, C = último 5%.
  - XYZ: coeficiente de variación de la demanda SEMANAL (la diaria castigaría de más a
         los productos de baja rotación): X ≤ 0.5 estable, Y ≤ 1.0 variable, Z > 1.0 errática.
  - Patrón de demanda (Syntetos-Boylan): ADI y CV² para identificar series intermitentes.

Se calcula con los 365 días previos al periodo de evaluación, para no mirar el futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ABC_CUTS = (0.80, 0.95)
XYZ_CUTS = (0.5, 1.0)
ADI_CUT = 1.32
CV2_CUT = 0.49


def abc_classify(revenue: pd.Series, cuts: tuple[float, float] = ABC_CUTS) -> pd.Series:
    """Clase ABC por participación acumulada de ingreso (índice = SKU)."""
    order = revenue.sort_values(ascending=False)
    total = order.sum()
    # Participación acumulada ANTES de cada SKU: el que cruza el corte queda en la clase alta.
    cum_before = (order.cumsum() - order) / total if total > 0 else order * 0.0
    cls = np.where(cum_before < cuts[0], "A", np.where(cum_before < cuts[1], "B", "C"))
    return pd.Series(cls, index=order.index, name="abc").reindex(revenue.index)


def xyz_classify(cv: pd.Series, cuts: tuple[float, float] = XYZ_CUTS) -> pd.Series:
    """Clase XYZ por coeficiente de variación. CV indefinido (sin ventas) → Z."""
    cls = np.where(cv <= cuts[0], "X", np.where(cv <= cuts[1], "Y", "Z"))
    cls = np.where(cv.isna(), "Z", cls)
    return pd.Series(cls, index=cv.index, name="xyz")


def demand_pattern(adi: pd.Series, cv2: pd.Series) -> pd.Series:
    """Suave / errática / intermitente / irregular (lumpy) según ADI y CV²."""
    pat = np.select(
        [
            (adi < ADI_CUT) & (cv2 < CV2_CUT),
            (adi < ADI_CUT) & (cv2 >= CV2_CUT),
            (adi >= ADI_CUT) & (cv2 < CV2_CUT),
        ],
        ["suave", "errática", "intermitente"],
        default="irregular",
    )
    return pd.Series(pat, index=adi.index, name="pattern")


def classify(sales: pd.DataFrame, end: pd.Timestamp | None = None, days: int = 365) -> pd.DataFrame:
    """Tabla por SKU con ingreso, CV semanal, ABC, XYZ y patrón de demanda.

    sales: unique_id, ds, y, sell_price. Usa la ventana (end - days, end].
    """
    end = end or sales["ds"].max()
    win = sales[(sales["ds"] > end - pd.Timedelta(days=days)) & (sales["ds"] <= end)].copy()
    win["revenue"] = win["y"] * win["sell_price"]

    g = win.groupby("unique_id")
    tab = g.agg(
        revenue=("revenue", "sum"),
        units=("y", "sum"),
        mean_daily=("y", "mean"),
        price=("sell_price", "last"),
        zero_share=("y", lambda s: (s == 0).mean()),
    )

    week = (end - win["ds"]).dt.days // 7  # semanas completas contadas hacia atrás
    weekly = win.assign(week=week).groupby(["unique_id", "week"])["y"].sum().unstack(fill_value=0)
    weekly = weekly.loc[:, weekly.columns < days // 7]
    w_mean = weekly.mean(axis=1)
    tab["cv_weekly"] = (weekly.std(axis=1, ddof=1) / w_mean.where(w_mean > 0)).reindex(tab.index)

    nonzero = win[win["y"] > 0].groupby("unique_id")["y"]
    n_nz = nonzero.size().reindex(tab.index).fillna(0)
    tab["adi"] = (g.size() / n_nz.where(n_nz > 0)).fillna(np.inf)
    tab["cv2"] = ((nonzero.std(ddof=0) / nonzero.mean()) ** 2).reindex(tab.index).fillna(0.0)

    tab["abc"] = abc_classify(tab["revenue"])
    tab["xyz"] = xyz_classify(tab["cv_weekly"])
    tab["segment"] = tab["abc"] + tab["xyz"]
    tab["pattern"] = demand_pattern(tab["adi"], tab["cv2"])
    tab["revenue_share"] = tab["revenue"] / tab["revenue"].sum()
    return tab.reset_index()


def eval_start(sales: pd.DataFrame) -> pd.Timestamp:
    """Último día antes del periodo de evaluación (primer cutoff de las ventanas reportadas)."""
    return sales["ds"].max() - pd.Timedelta(days=config.N_WINDOWS * config.HORIZON)


def main() -> None:
    sales = pd.read_parquet(config.SALES_LONG)
    tab = classify(sales, end=eval_start(sales))
    tab.to_parquet(config.ABC_XYZ, index=False)

    matrix = pd.crosstab(tab["abc"], tab["xyz"], margins=True)
    share = tab.groupby("abc")["revenue_share"].sum()
    print("Matriz ABC/XYZ (número de SKUs):")
    print(matrix.to_string())
    print("\nParticipación de ingreso por clase ABC:")
    print(share.map("{:.1%}".format).to_string())
    print("\nPatrón de demanda:")
    print(tab["pattern"].value_counts().to_string())


if __name__ == "__main__":
    main()
