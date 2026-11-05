"""Extracción de los datos del dashboard a partir de los resultados del pipeline.

Uso: python -m src.dashboard.datos   (escribe reports/dashboard/datos.json)

Solo salen resultados DERIVADOS: pronósticos, márgenes de los intervalos, métricas, resultados
de la simulación y agregados por departamento. Ningún conjunto contiene la venta diaria ni el
precio de un SKU: las reglas de M5 prohíben redistribuir los datos de la competencia, y
`validar` falla si aparece una columna con ese contenido.

Los conjuntos son columnares, como en el dashboard de PROFECO: ``{columna: [valores]}``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.inventory import policy as pol
from src.models import backtest as bt
from src.models import intervals
from src.models.baselines import BASELINE_MODELS

MODEL = "LightGBM"
SS_METHODS = ("sqrt", "acumulado")  # los dos que ofrece el dashboard (el empírico no se recomienda)
TARGETS = (0.90, 0.95, 0.98)  # fill rate objetivo para comparar políticas a igual servicio
DEPT_MODELS = (MODEL, "AutoETS", "SeasonalNaive", "MediaMovil28")

# Columnas que nunca pueden salir: identificarían la venta o el precio originales de M5.
PROHIBIDAS = {"y", "sell_price", "price", "revenue", "revenue_365d", "units", "demand"}

Columnas = dict[str, list[Any]]


def _r(valores, dec: int) -> list:
    """Redondea y convierte a tipos de JSON; NaN → None."""
    out = []
    for v in np.asarray(valores, dtype=float):
        if not np.isfinite(v):
            out.append(None)
        else:
            v = round(float(v), dec)
            out.append(int(v) if v.is_integer() else v)
    return out


def _columnas(df: pd.DataFrame, decimales: dict[str, int] | None = None) -> Columnas:
    decimales = decimales or {}
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = _r(s, decimales.get(col, 4))
        else:
            out[col] = [None if pd.isna(v) else str(v) for v in s]
    return out


def combos() -> list[tuple[int, float, str]]:
    """Orden fijo de los escenarios de política por SKU: lead time × nivel de servicio × método."""
    return [(lt, sl, m) for lt in config.LEAD_TIMES for sl in config.SERVICE_LEVELS for m in SS_METHODS]


# ------------------------------------------------------------------------------- SKUs


def skus() -> tuple[Columnas, Columnas]:
    """Catálogo y política por SKU (ordenados por ingreso, sin exponer el ingreso).

    Devuelve (skus, valor_ss): el valor del safety stock necesita el precio, así que se entrega
    ya agregado por escenario × segmento × departamento.
    """
    classes = pd.read_parquet(config.ABC_XYZ).set_index("unique_id")
    meta = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "item_id", "dept_id"]).drop_duplicates(
        "unique_id"
    ).set_index("unique_id")
    future = pd.read_parquet(config.FORECAST_FUTURE).sort_values(["unique_id", "ds"])
    cv = pd.read_parquet(config.CV_LGBM)

    tables = {c: pol.recommendation_table(*c).set_index("unique_id") for c in combos()}
    base = tables[combos()[0]]
    ids = list(classes.loc[base.index].sort_values("revenue", ascending=False).index)

    fc = future.pivot(index="unique_id", columns="ds", values=MODEL).loc[ids]
    q = intervals.residual_quantiles(cv, MODEL).loc[ids]

    sku_metrics = pd.read_parquet(config.METRICS_SKU).query("split == 'eval' and model == @MODEL")
    perf = sku_metrics.groupby("unique_id")[["demand", "abs_err", "err"]].sum().reindex(ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        wape = perf["abs_err"] / perf["demand"].where(perf["demand"] > 0)
        bias = perf["err"] / perf["demand"].where(perf["demand"] > 0)
    sim = pd.read_parquet(config.SIM_BY_SKU).query("policy == @MODEL and ss_method == 'sqrt'")
    fill = sim.set_index("unique_id")["fill_rate"].reindex(ids)

    out: Columnas = {
        "id": [meta.loc[i, "item_id"] for i in ids],
        "dept": [meta.loc[i, "dept_id"] for i in ids],
        "abc": [classes.loc[i, "abc"] for i in ids],
        "xyz": [classes.loc[i, "xyz"] for i in ids],
        "patron": [classes.loc[i, "pattern"] for i in ids],
        "pronostico": [_r(row, 2) for row in fc.to_numpy()],
        "lote": _r(base.loc[ids, "lot"], 0),
        "q_lo80": _r(q["lo-80"], 2), "q_hi80": _r(q["hi-80"], 2),
        "q_lo95": _r(q["lo-95"], 2), "q_hi95": _r(q["hi-95"], 2),
        "wape": _r(wape, 4), "sesgo": _r(bias, 4), "fill_sim": _r(fill, 4),
    }
    for lt in config.LEAD_TIMES:
        out[f"dem_lt{lt}"] = _r(tables[(lt, config.SERVICE_LEVELS[0], SS_METHODS[0])].loc[ids, "demand_lead_time"], 2)
    # ss y rop: una lista por SKU con un valor por escenario, en el orden de `combos()`.
    out["ss"] = [[int(tables[c].loc[i, "safety_stock"]) for c in combos()] for i in ids]
    out["rop"] = [[int(tables[c].loc[i, "rop"]) for c in combos()] for i in ids]

    seg = classes.loc[ids, "segment"].to_numpy()
    dept = meta.loc[ids, "dept_id"].to_numpy()
    rows = []
    for k, c in enumerate(combos()):
        t = tables[c].loc[ids]
        value = t["safety_stock"] * t["price"] * config.UNIT_COST_RATIO
        frame = pd.DataFrame({"segmento": seg, "dept": dept, "valor": value.to_numpy()})
        for (s, d), g in frame.groupby(["segmento", "dept"]):
            rows.append({"escenario": k, "segmento": s, "dept": d, "valor": g["valor"].sum()})
    return out, _columnas(pd.DataFrame(rows), {"valor": 0})


# --------------------------------------------------------------------------- pronóstico


def pronostico() -> dict[str, Columnas]:
    cv = bt.load_cv()
    cutoffs = sorted(cv["cutoff"].unique())[-config.N_WINDOWS :]
    ev = cv[cv["cutoff"].isin(cutoffs)].copy()
    dept = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "dept_id"]).drop_duplicates("unique_id")
    ev = ev.merge(dept, on="unique_id")

    # Real vs pronóstico, sumado por departamento: un agregado de cientos de SKUs, no el dato original.
    cols = ["y", *DEPT_MODELS]
    by_dept = ev.groupby(["dept_id", "ds"])[cols].sum().reset_index()
    total = ev.groupby("ds")[cols].sum().reset_index().assign(dept_id="Total")
    agg = pd.concat([total, by_dept], ignore_index=True)
    agg = agg.rename(columns={"dept_id": "dept", "y": "real"})
    agg["ds"] = agg["ds"].dt.strftime("%Y-%m-%d")

    ev["paso"] = (ev["ds"] - ev["cutoff"]).dt.days
    steps = {"paso": list(range(1, config.HORIZON + 1))}
    demand = ev.groupby("paso")["y"].sum()
    for m in (MODEL, "AutoETS", "SeasonalNaive"):
        steps[m] = _r((ev[m] - ev["y"]).abs().groupby(ev["paso"]).sum() / demand, 4)

    wr = pd.read_parquet(config.METRICS_WRMSSE).query("split == 'eval'")
    wr = wr.assign(cutoff=wr["cutoff"].dt.strftime("%Y-%m-%d"))[["cutoff", "level", "model", "value"]]
    seg = pd.read_parquet(config.METRICS_SEGMENT).query("split == 'eval'")
    seg = seg[["grouping", "group", "model", "n_skus", "mae", "wape", "bias", "rmsse"]]
    cov = pd.read_parquet(config.METRICS_COVERAGE)
    imp = pd.read_parquet(config.FEATURE_IMPORTANCE).head(12)[["feature", "gain_share"]]
    abl = pd.read_parquet(config.ABLATION)[["variant", "wrmsse", "wape", "bias", "wrmsse_vs_base"]]

    return {
        "real_vs_pronostico": _columnas(agg, dict.fromkeys(cols, 1)),
        "error_horizonte": steps,
        "wrmsse": _columnas(wr),
        "segmentos": _columnas(seg),
        "cobertura": _columnas(cov[["grouping", "group", "level", "nominal", "coverage", "mean_width"]]),
        "importancia": _columnas(imp),
        "ablaciones": _columnas(abl),
    }


# ---------------------------------------------------------------------------- política


def politica() -> dict[str, Columnas]:
    frontier = pd.read_parquet(config.SIM_FRONTIER)
    cols = ["policy", "ss_method", "lead_time", "service_level", "group", "fill_rate", "cycle_service",
            "avg_inv_value"]
    need = pd.concat([pol.inventory_at_fill_rate(frontier, t) for t in TARGETS], ignore_index=True)
    sc = pd.read_parquet(config.SIM_SCENARIOS)
    sc = sc[[*cols, "skus_meeting_target", "days_of_supply", "stockout_days"]]
    costs = pd.read_parquet(config.SIM_COSTS)
    return {
        "frontera": _columnas(frontier[cols], {"avg_inv_value": 0}),
        "necesario": _columnas(need, {"avg_inv_value": 0}),
        "escenarios": _columnas(sc, {"avg_inv_value": 0}),
        "costos": _columnas(costs, {"inv_classic": 0, "inv_lgbm": 0}),
    }


# --------------------------------------------------------------------------------- meta


def _commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=config.ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def meta() -> dict[str, Any]:
    sales = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "ds"])
    future = pd.read_parquet(config.FORECAST_FUTURE, columns=["ds"])
    last = sales["ds"].max()
    eval_start = last - pd.Timedelta(days=config.N_WINDOWS * config.HORIZON - 1)
    return {
        "tienda": config.STORE_ID,
        "categoria": config.CAT_ID,
        "n_skus": int(sales["unique_id"].nunique()),
        "historia_desde": sales["ds"].min().strftime("%Y-%m-%d"),
        "historia_hasta": last.strftime("%Y-%m-%d"),
        "evaluacion_desde": eval_start.strftime("%Y-%m-%d"),
        "pronostico_fechas": sorted(future["ds"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "lead_times": list(config.LEAD_TIMES),
        "niveles_servicio": list(config.SERVICE_LEVELS),
        "metodos_ss": list(SS_METHODS),
        "escenarios": [{"lead_time": lt, "nivel": sl, "metodo": m} for lt, sl, m in combos()],
        "objetivos_fill_rate": list(TARGETS),
        "z": {str(sl): round(pol.z_score(sl), 4) for sl in config.SERVICE_LEVELS},
        "costos": {
            "costo_unitario": config.UNIT_COST_RATIO,
            "mantener_anual": config.HOLDING_RATE_ANNUAL,
            "costo_pedido": config.ORDER_COST,
            "cobertura_max_lote": config.MAX_COVER_DAYS,
        },
        "modelos_base": list(BASELINE_MODELS),
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit_git": os.getenv("GITHUB_SHA") or _commit(),
    }


# ----------------------------------------------------------------------------- armado


def extraer() -> dict[str, Any]:
    sku_cols, ss_value = skus()
    return {
        "meta": meta(),
        "skus": sku_cols,
        "valor_ss": ss_value,
        **pronostico(),
        **politica(),
    }


def validar(datos: dict[str, Any]) -> None:
    """Falla si falta un conjunto, si una columna tiene otro largo o si sale una columna prohibida."""
    requeridos = {"meta", "skus", "valor_ss", "real_vs_pronostico", "error_horizonte", "wrmsse", "segmentos",
                  "cobertura", "importancia", "ablaciones", "frontera", "necesario", "escenarios", "costos"}
    faltan = requeridos - set(datos)
    if faltan:
        raise ValueError(f"[dashboard] faltan conjuntos: {sorted(faltan)}")
    for nombre, conjunto in datos.items():
        if nombre == "meta":
            continue
        prohibidas = PROHIBIDAS & set(conjunto)
        if prohibidas:
            raise ValueError(f"[dashboard] {nombre}: columnas con datos originales de M5 {sorted(prohibidas)}")
        largos = {len(v) for v in conjunto.values()}
        if len(largos) != 1:
            raise ValueError(f"[dashboard] {nombre}: columnas de distinto largo {sorted(largos)}")
    n_esc = len(datos["meta"]["escenarios"])
    if any(len(v) != n_esc for v in datos["skus"]["ss"] + datos["skus"]["rop"]):
        raise ValueError("[dashboard] skus: ss/rop deben tener un valor por escenario")
    n_dias = len(datos["meta"]["pronostico_fechas"])
    if any(len(v) != n_dias for v in datos["skus"]["pronostico"]):
        raise ValueError("[dashboard] skus: el pronóstico debe cubrir todo el horizonte")


def main() -> None:
    datos = extraer()
    validar(datos)
    config.DASHBOARD_DATA.parent.mkdir(parents=True, exist_ok=True)
    config.DASHBOARD_DATA.write_text(
        json.dumps(datos, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    size = config.DASHBOARD_DATA.stat().st_size / 1e6
    print(f"[dashboard] {config.DASHBOARD_DATA} ({size:.2f} MB) · {datos['meta']['n_skus']:,} SKUs")


if __name__ == "__main__":
    main()
