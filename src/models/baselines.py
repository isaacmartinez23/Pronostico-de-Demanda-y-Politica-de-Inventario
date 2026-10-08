"""Baselines estadísticos con statsforecast y su backtesting.

Uso: python -m src.models.baselines

  - Naive, SeasonalNaive(7) y media móvil de 28 días: baratos, se evalúan en todas las
    ventanas (calibración + evaluación) porque también alimentan la política de inventario.
  - AutoETS y AutoARIMA: solo en las 3 ventanas de evaluación y con historia acotada
    (STATS_INPUT_SIZE) para que el proyecto corra en una laptop.
"""

from __future__ import annotations

import time

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    Naive,
    SeasonalNaive,
    WindowAverage,
)

from src import config

CHEAP_MODELS = ["Naive", "SeasonalNaive", "MediaMovil28"]
HEAVY_MODELS = ["AutoETS", "AutoARIMA"]
BASELINE_MODELS = CHEAP_MODELS + HEAVY_MODELS


def cheap_models() -> list:
    return [
        Naive(),
        SeasonalNaive(season_length=config.SEASON_LENGTH),
        WindowAverage(window_size=28, alias="MediaMovil28"),
    ]


def heavy_models() -> list:
    models = [AutoETS(season_length=config.SEASON_LENGTH)]
    if not config.SKIP_ARIMA:
        models.append(AutoARIMA(season_length=config.SEASON_LENGTH))
    return models


def run_cv(
    sales: pd.DataFrame,
    models: list,
    n_windows: int,
    h: int = config.HORIZON,
    input_size: int | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Backtesting con ventanas móviles no solapadas de `h` días.

    Devuelve unique_id, ds, cutoff, y + una columna por modelo (pronósticos ≥ 0).
    """
    sf = StatsForecast(
        models=models,
        freq=config.FREQ,
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=config.SEASON_LENGTH),
    )
    cv = sf.cross_validation(
        df=sales[["unique_id", "ds", "y"]],
        h=h,
        n_windows=n_windows,
        step_size=h,
        input_size=input_size,
    ).reset_index(drop=True)
    model_cols = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    cv[model_cols] = cv[model_cols].clip(lower=0)
    return cv


def main() -> None:
    sales = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "ds", "y"])
    total_windows = config.N_WINDOWS + config.N_CALIB_WINDOWS

    t0 = time.time()
    cheap = run_cv(sales, cheap_models(), n_windows=total_windows)
    print(f"Baselines simples ({total_windows} ventanas): {time.time() - t0:.0f}s")

    t0 = time.time()
    heavy = run_cv(
        sales, heavy_models(), n_windows=config.N_WINDOWS, input_size=config.STATS_INPUT_SIZE
    )
    names = " + ".join(m.alias for m in heavy_models())
    print(f"{names} ({config.N_WINDOWS} ventanas): {time.time() - t0:.0f}s")

    cv = cheap.merge(heavy.drop(columns="y"), on=["unique_id", "ds", "cutoff"], how="left")
    cv.to_parquet(config.CV_BASELINES, index=False)
    print(f"{config.CV_BASELINES.name}: {len(cv):,} filas | cutoffs: "
          f"{[str(c.date()) for c in sorted(cv['cutoff'].unique())]}")


if __name__ == "__main__":
    main()
