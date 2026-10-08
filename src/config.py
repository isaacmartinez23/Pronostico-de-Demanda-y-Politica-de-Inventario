"""Configuración central del proyecto: rutas, subconjunto de M5 y parámetros del experimento."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Carpeta de datos sobreescribible: CI corre el pipeline sobre un M5 sintético en otra ruta.
DATA_DIR = Path(os.getenv("DFI_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DATA_DIR = ROOT / "app" / "data"

# Subconjunto de M5 (configurable por variable de entorno para probar otra tienda/categoría).
STORE_ID = os.getenv("M5_STORE", "CA_1")
CAT_ID = os.getenv("M5_CAT", "FOODS")

FREQ = "D"
SEASON_LENGTH = 7
HORIZON = 28  # 4 semanas

# Backtesting: 3 ventanas de evaluación de 28 días (las que se reportan).
N_WINDOWS = 3
# Ventanas previas, solo para los modelos baratos: sirven para estimar σ(error)
# sin usar el periodo que después se simula (evita fuga de información).
N_CALIB_WINDOWS = 3

# Historia que ven los modelos estadísticos en cada ajuste (acota el costo de AutoARIMA/AutoETS).
STATS_INPUT_SIZE = 2 * 365

# SKUs con menos historia quedan fuera: no alcanzan para las ventanas de backtesting
# (son un problema distinto: pronóstico de producto nuevo).
MIN_HISTORY_DAYS = 365

RANDOM_SEED = 42

# Inventario
SERVICE_LEVELS = (0.90, 0.95, 0.99)
LEAD_TIMES = (3, 7, 14)
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_LEAD_TIME = 7
# Método de safety stock de la tabla de recomendación (ver policy.SS_METHODS).
DEFAULT_SS_METHOD = "sqrt"

# Supuestos de costos (M5 no los trae): se usan solo para el EOQ.
UNIT_COST_RATIO = 0.70  # costo unitario = 70% del precio de venta
HOLDING_RATE_ANNUAL = 0.25  # costo de mantener = 25% anual del costo unitario
ORDER_COST = 5.0  # costo fijo por pedido (USD)
MAX_COVER_DAYS = 14  # tope de cobertura del lote (perecederos)

# Archivos
SALES_LONG = PROCESSED_DIR / "sales_long.parquet"
FUTURE_EXOG = PROCESSED_DIR / "future_exog.parquet"
ABC_XYZ = PROCESSED_DIR / "abc_xyz.parquet"
CV_BASELINES = PROCESSED_DIR / "cv_baselines.parquet"
CV_LGBM = PROCESSED_DIR / "cv_lgbm.parquet"
FORECAST_FUTURE = PROCESSED_DIR / "forecast_future.parquet"
FEATURE_IMPORTANCE = PROCESSED_DIR / "feature_importance.parquet"
METRICS_WRMSSE = PROCESSED_DIR / "metrics_wrmsse.parquet"
METRICS_SKU = PROCESSED_DIR / "metrics_sku.parquet"
METRICS_SEGMENT = PROCESSED_DIR / "metrics_segment.parquet"
METRICS_COVERAGE = PROCESSED_DIR / "metrics_coverage.parquet"
POLICY_TABLE = PROCESSED_DIR / "policy_table.parquet"
SIM_SCENARIOS = PROCESSED_DIR / "sim_scenarios.parquet"
SIM_FRONTIER = PROCESSED_DIR / "sim_frontier.parquet"
SIM_BY_SKU = PROCESSED_DIR / "sim_by_sku.parquet"

RAW_FILES = ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, APP_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
