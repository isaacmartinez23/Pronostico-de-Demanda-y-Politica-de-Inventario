"""LightGBM global con mlforecast: backtesting, pronóstico final e intervalos conformales.

Uso: python -m src.models.ml_model

Un solo modelo aprende de todas las series a la vez (los SKUs de baja rotación se
benefician de los patrones de los demás). Objetivo Tweedie: adecuado para conteos con
muchos ceros. El horizonte de 28 días se pronostica de forma recursiva.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast

from src import config
from src.features import build_features as bf
from src.models import intervals

MODEL = "LightGBM"

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "cat_smooth": 20,
    "random_state": config.RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
    "verbose": -1,
}


def make_forecaster(params: dict | None = None) -> MLForecast:
    return MLForecast(
        models={MODEL: lgb.LGBMRegressor(**(params or LGBM_PARAMS))},
        freq=config.FREQ,
        lags=bf.LAGS,
        lag_transforms=bf.lag_transforms(),
        date_features=bf.DATE_FEATURES,
    )


def run_cv(train: pd.DataFrame, n_windows: int, h: int = config.HORIZON) -> pd.DataFrame:
    """Backtesting con reentrenamiento en cada ventana (mismos cortes que los baselines)."""
    fcst = make_forecaster()
    cv = fcst.cross_validation(
        df=train,
        n_windows=n_windows,
        h=h,
        step_size=h,
        static_features=bf.STATIC_FEATURES,
        refit=True,
    )
    cv[MODEL] = cv[MODEL].clip(lower=0)
    return cv[["unique_id", "ds", "cutoff", "y", MODEL]]


def fit_predict(
    train: pd.DataFrame, future: pd.DataFrame, h: int = config.HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena con toda la historia y pronostica `h` días. Devuelve (pronóstico, importancias)."""
    fcst = make_forecaster()
    fcst.fit(train, static_features=bf.STATIC_FEATURES)
    pred = fcst.predict(h=h, X_df=future)
    pred[MODEL] = pred[MODEL].clip(lower=0)

    model = fcst.models_[MODEL]
    importance = pd.DataFrame(
        {
            "feature": model.feature_name_,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()
    return pred, importance


def main() -> None:
    sales = pd.read_parquet(config.SALES_LONG)
    future = pd.read_parquet(config.FUTURE_EXOG)
    train, fut = bf.make_model_frames(sales, future)
    total_windows = config.N_WINDOWS + config.N_CALIB_WINDOWS

    t0 = time.time()
    cv = run_cv(train, n_windows=total_windows)
    cv.to_parquet(config.CV_LGBM, index=False)
    print(f"Backtesting LightGBM ({total_windows} ventanas): {time.time() - t0:.0f}s")

    t0 = time.time()
    pred, importance = fit_predict(train, fut)
    # Intervalos calibrados con los residuos fuera de muestra de todas las ventanas.
    quantiles = intervals.residual_quantiles(cv, MODEL)
    pred = intervals.add_intervals(pred, quantiles, MODEL)
    pred.to_parquet(config.FORECAST_FUTURE, index=False)
    importance.to_parquet(config.FEATURE_IMPORTANCE, index=False)
    print(f"Modelo final + pronóstico {config.HORIZON} días: {time.time() - t0:.0f}s")
    print(f"{config.FORECAST_FUTURE.name}: {pred['ds'].min().date()} → {pred['ds'].max().date()} | "
          f"demanda total pronosticada: {pred[MODEL].sum():,.0f} unidades")
    print("\nTop 10 variables (ganancia):")
    print(importance.head(10)[["feature", "gain_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
