"""Del pronóstico a la decisión: safety stock, punto de reorden, lote y simulación.

Política (s, S) con revisión diaria y ventas perdidas:

    safety stock  SS  = z(nivel de servicio) · σ(error diario) · √(lead time)
    punto reorden ROP = demanda esperada en el lead time + SS
    lote          Q   = EOQ, acotado entre 1 y MAX_COVER_DAYS días de demanda
    pedido             si posición de inventario ≤ ROP → pedir hasta ROP + Q

La simulación recorre los 84 días del periodo de evaluación usando la demanda real y solo
la información disponible en cada momento (el pronóstico vigente y el σ estimado con
ventanas anteriores).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config

EXTEND_CYCLE = 14  # el pronóstico se extiende repitiendo sus últimas 2 semanas
CLASSIC_WINDOW = 56  # días de historia para la política clásica (media y desviación)


@dataclass(frozen=True)
class CostParams:
    unit_cost_ratio: float = config.UNIT_COST_RATIO
    holding_rate_annual: float = config.HOLDING_RATE_ANNUAL
    order_cost: float = config.ORDER_COST
    max_cover_days: int = config.MAX_COVER_DAYS


# --------------------------------------------------------------------------- fórmulas


def z_score(service_level: float) -> float:
    """Cuantil normal del nivel de servicio de ciclo (0.95 → 1.645)."""
    if not 0 < service_level < 1:
        raise ValueError("service_level debe estar en (0, 1)")
    return float(norm.ppf(service_level))


def safety_stock(sigma_error, lead_time: float, service_level: float):
    """SS = z · σ(error diario) · √(lead time). Nunca negativo."""
    if lead_time < 0:
        raise ValueError("lead_time no puede ser negativo")
    ss = z_score(service_level) * np.asarray(sigma_error, dtype=float) * np.sqrt(lead_time)
    return np.maximum(ss, 0.0)


def reorder_point(demand_lead_time, ss):
    """ROP = demanda esperada durante el lead time + safety stock."""
    return np.asarray(demand_lead_time, dtype=float) + np.asarray(ss, dtype=float)


def eoq(annual_demand, order_cost: float, holding_cost):
    """Cantidad económica de pedido: √(2·D·K / h). Cero si no hay demanda."""
    d = np.maximum(np.asarray(annual_demand, dtype=float), 0.0)
    h = np.asarray(holding_cost, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(2.0 * d * order_cost / h)
    return np.where((h > 0) & (d > 0), q, 0.0)


def lot_size(daily_demand, price, costs: CostParams | None = None):
    """Lote a pedir: EOQ acotado a [1 día, max_cover_days] de demanda, en unidades enteras ≥ 1."""
    costs = costs or CostParams()
    daily = np.maximum(np.asarray(daily_demand, dtype=float), 0.0)
    holding = np.asarray(price, dtype=float) * costs.unit_cost_ratio * costs.holding_rate_annual
    q = eoq(daily * 365.0, costs.order_cost, holding)
    q = np.clip(q, daily, daily * costs.max_cover_days)
    return np.maximum(np.ceil(q), 1.0)


def order_quantity(inventory_position, rop, lot):
    """Cantidad a pedir hoy: si la posición ≤ ROP, reponer hasta ROP + lote; si no, cero."""
    ip = np.asarray(inventory_position, dtype=float)
    rop = np.asarray(rop, dtype=float)
    target = rop + np.asarray(lot, dtype=float)
    return np.where(ip <= rop, np.ceil(np.maximum(target - ip, 0.0)), 0.0)


# ------------------------------------------------------------------------- simulación


@dataclass
class SimResult:
    on_hand: np.ndarray  # inventario al cierre de cada día (T, n)
    sales: np.ndarray
    lost: np.ndarray
    orders: np.ndarray


def simulate_policy(
    demand: np.ndarray,
    rop: np.ndarray,
    lot: np.ndarray,
    lead_time: int,
    initial_on_hand: np.ndarray | None = None,
) -> SimResult:
    """Simula la política día a día para n SKUs a la vez.

    demand, rop, lot: matrices (T, n). Cada día: llega lo pedido hace `lead_time` días,
    se atiende la demanda con lo disponible (lo no atendido se pierde) y al cierre se
    revisa la posición de inventario para decidir el pedido.
    """
    if lead_time < 1:
        raise ValueError("lead_time debe ser ≥ 1 día")
    demand = np.asarray(demand, dtype=float)
    n_days, n = demand.shape
    on_hand = (rop[0] + lot[0] if initial_on_hand is None else initial_on_hand).astype(float).copy()
    pipeline = np.zeros((n_days + lead_time, n))
    on_order = np.zeros(n)

    res = SimResult(*(np.zeros((n_days, n)) for _ in range(4)))
    for t in range(n_days):
        on_hand += pipeline[t]
        on_order -= pipeline[t]

        sales = np.minimum(on_hand, demand[t])
        on_hand -= sales

        q = order_quantity(on_hand + on_order, rop[t], lot[t])
        pipeline[t + lead_time] += q
        on_order += q

        res.on_hand[t], res.sales[t], res.lost[t], res.orders[t] = on_hand, sales, demand[t] - sales, q
    return res


def summarize_by_sku(
    sim: SimResult, demand: np.ndarray, price: np.ndarray, ids: list[str],
    costs: CostParams | None = None,
) -> pd.DataFrame:
    costs = costs or CostParams()
    dem = demand.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(dem > 0, sim.sales.sum(0) / dem, np.nan)
    return pd.DataFrame(
        {
            "unique_id": ids,
            "demand": dem,
            "daily_demand": demand.mean(0),
            "sales": sim.sales.sum(0),
            "lost": sim.lost.sum(0),
            "fill_rate": fill,
            "stockout_days": (sim.lost > 0).mean(0),
            "avg_on_hand": sim.on_hand.mean(0),
            "avg_inv_value": sim.on_hand.mean(0) * price * costs.unit_cost_ratio,
            "n_orders": (sim.orders > 0).sum(0),
            "price": price,
        }
    )


def aggregate(by_sku: pd.DataFrame, target: float | None = None) -> dict:
    """Agrega resultados por SKU. Los fill rates se ponderan por valor (precio)."""
    dem_v = (by_sku["demand"] * by_sku["price"]).sum()
    out = {
        "n_skus": len(by_sku),
        "fill_rate": (by_sku["sales"] * by_sku["price"]).sum() / dem_v if dem_v > 0 else np.nan,
        "fill_rate_units": by_sku["sales"].sum() / by_sku["demand"].sum(),
        "stockout_days": by_sku["stockout_days"].mean(),
        "avg_inv_units": by_sku["avg_on_hand"].sum(),
        "avg_inv_value": by_sku["avg_inv_value"].sum(),
        "days_of_supply": by_sku["avg_on_hand"].sum() / by_sku["daily_demand"].sum(),
        "orders_per_sku": by_sku["n_orders"].mean(),
    }
    if target is not None:
        out["skus_meeting_target"] = (by_sku["fill_rate"].dropna() >= target).mean()
    return out


# --------------------------------------------------- insumos de la política (por SKU/día)


@dataclass
class PolicyBasis:
    """Lo que la política sabe en cada punto de decisión (fila 0 = decisión inicial)."""

    mu_lead: np.ndarray  # demanda esperada en el lead time (T+1, n)
    sigma: np.ndarray  # σ del error diario (T+1, n)
    daily: np.ndarray  # demanda diaria esperada, para dimensionar el lote (T+1, n)


def cv_cube(cv: pd.DataFrame, col: str, ids: list[str], h: int = config.HORIZON) -> np.ndarray:
    """Pasa el backtesting a un cubo (ventanas, h, n) ordenado por fecha."""
    piv = cv.pivot(index="ds", columns="unique_id", values=col)[ids].sort_index()
    return piv.to_numpy(dtype=float).reshape(-1, h, len(ids))


def extend_forecast(f: np.ndarray, extra: int) -> np.ndarray:
    """Extiende el pronóstico `extra` días repitiendo las últimas 2 semanas (respeta el patrón semanal)."""
    if extra <= 0:
        return f
    reps = int(np.ceil(extra / EXTEND_CYCLE))
    tail = np.concatenate([f[-EXTEND_CYCLE:]] * reps, axis=0)[:extra]
    return np.concatenate([f, tail], axis=0)


def _expand(per_window: list[tuple[np.ndarray, np.ndarray, np.ndarray]], h: int) -> PolicyBasis:
    """Reparte los insumos de cada ventana a los T+1 puntos de decisión."""
    n_win = len(per_window)
    mu, sg, dy = [], [], []
    for d in range(n_win * h + 1):
        k = min(d // h, n_win - 1)
        mu_by_offset, sigma, daily = per_window[k]
        mu.append(mu_by_offset[d - h * k])
        sg.append(sigma)
        dy.append(daily)
    return PolicyBasis(np.array(mu), np.array(sg), np.array(dy))


def forecast_basis(
    y: np.ndarray, f: np.ndarray, lead_time: int, n_eval: int = config.N_WINDOWS
) -> PolicyBasis:
    """Política basada en un pronóstico.

    y, f: cubos (ventanas, h, n) de demanda real y pronóstico. Las últimas `n_eval`
    ventanas se simulan; para cada una, σ se estima con los errores de TODAS las
    ventanas anteriores (nunca con el periodo que se está simulando).
    """
    n_win, h, _ = f.shape
    resid = y - f
    per_window = []
    for g in range(n_win - n_eval, n_win):
        sigma = resid[:g].reshape(-1, resid.shape[2]).std(axis=0, ddof=1)
        cs = np.concatenate([np.zeros((1, f.shape[2])), extend_forecast(f[g], lead_time).cumsum(0)])
        mu_by_offset = cs[lead_time : lead_time + h + 1] - cs[: h + 1]
        per_window.append((mu_by_offset, sigma, f[g].mean(0)))
    return _expand(per_window, h)


def classic_basis(
    history: np.ndarray, lead_time: int, n_eval: int = config.N_WINDOWS,
    h: int = config.HORIZON, window: int = CLASSIC_WINDOW,
) -> PolicyBasis:
    """Política clásica de libro de texto: media y desviación de la demanda reciente.

    history: matriz (días, n) de demanda real que termina en el último día simulado.
    En cada corte usa los `window` días previos; se actualiza con la misma cadencia
    que el pronóstico (cada 28 días) para que la comparación sea justa.
    """
    n_days = history.shape[0]
    per_window = []
    for j in range(n_eval):
        cutoff = n_days - (n_eval - j) * h  # índice del primer día de la ventana j
        past = history[cutoff - window : cutoff]
        mean, std = past.mean(0), past.std(0, ddof=1)
        mu_by_offset = np.tile(mean * lead_time, (h + 1, 1))
        per_window.append((mu_by_offset, std, mean))
    return _expand(per_window, h)


def run_policy(
    basis: PolicyBasis, demand: np.ndarray, price: np.ndarray, ids: list[str],
    lead_time: int, service_level: float, costs: CostParams | None = None,
) -> pd.DataFrame:
    """Simula una política para un lead time y nivel de servicio. Devuelve resultados por SKU."""
    costs = costs or CostParams()
    ss = safety_stock(basis.sigma, lead_time, service_level)
    rop = np.ceil(reorder_point(basis.mu_lead, ss))
    lot = lot_size(basis.daily, price, costs)
    sim = simulate_policy(demand, rop[1:], lot[1:], lead_time, initial_on_hand=rop[0] + lot[0])
    out = summarize_by_sku(sim, demand, price, ids, costs)
    out["safety_stock"] = ss[1:].mean(0)
    out["rop"] = rop[1:].mean(0)
    return out


# ----------------------------------------------------------------------------- pipeline

POLICIES = {"Clásica (media móvil)": None, "SeasonalNaive": "SeasonalNaive", "LightGBM": "LightGBM"}
FRONTIER_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 0.99, 0.995, 0.999)


def load_inputs() -> dict:
    """Carga demanda, pronósticos de backtesting, precios y clases, alineados por SKU."""
    sales = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "ds", "y", "sell_price"])
    cv = pd.read_parquet(config.CV_BASELINES).merge(
        pd.read_parquet(config.CV_LGBM).drop(columns="y"), on=["unique_id", "ds", "cutoff"]
    )
    abc = pd.read_parquet(config.ABC_XYZ).set_index("unique_id")
    ids = sorted(cv["unique_id"].unique())

    sim_days = config.N_WINDOWS * config.HORIZON
    y = cv_cube(cv, "y", ids)
    recent = sales[sales["ds"] > sales["ds"].max() - pd.Timedelta(days=sim_days + CLASSIC_WINDOW)]
    history = recent.pivot(index="ds", columns="unique_id", values="y")[ids].sort_index().to_numpy()
    eval_start = sales["ds"].max() - pd.Timedelta(days=sim_days)
    price = (
        sales[sales["ds"] > eval_start].groupby("unique_id")["sell_price"].mean().reindex(ids).to_numpy()
    )
    return {
        "ids": ids,
        "cv": cv,
        "y": y,
        "demand": y[-config.N_WINDOWS :].reshape(sim_days, len(ids)),
        "history": history,
        "price": price,
        "abc": abc.reindex(ids),
    }


def make_basis(inputs: dict, policy: str, lead_time: int) -> PolicyBasis:
    col = POLICIES[policy]
    if col is None:
        return classic_basis(inputs["history"], lead_time)
    return forecast_basis(inputs["y"], cv_cube(inputs["cv"], col, inputs["ids"]), lead_time)


def _grouped(by_sku: pd.DataFrame, abc: pd.Series, target: float) -> list[dict]:
    rows = [{"group": "Total", **aggregate(by_sku, target)}]
    for cls in ("A", "B", "C"):
        rows.append({"group": cls, **aggregate(by_sku[abc.to_numpy() == cls], target)})
    return rows


def run_grid(inputs: dict, lead_times, service_levels, keep_sku=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre política × lead time × nivel de servicio. Devuelve (agregado, detalle por SKU)."""
    agg_rows, sku_frames = [], []
    for lt in lead_times:
        for policy in POLICIES:
            basis = make_basis(inputs, policy, lt)
            for sl in service_levels:
                by_sku = run_policy(basis, inputs["demand"], inputs["price"], inputs["ids"], lt, sl)
                key = {"policy": policy, "lead_time": lt, "service_level": sl}
                agg_rows += [{**key, **r} for r in _grouped(by_sku, inputs["abc"]["abc"], sl)]
                if keep_sku and (lt, sl) in keep_sku:
                    sku_frames.append(by_sku.assign(**key))
    detail = pd.concat(sku_frames, ignore_index=True) if sku_frames else pd.DataFrame()
    return pd.DataFrame(agg_rows), detail


def inventory_at_fill_rate(frontier: pd.DataFrame, target: float) -> pd.DataFrame:
    """Inventario promedio (valor) que necesita cada política para ALCANZAR un fill rate dado.

    Interpola sobre la curva servicio-inventario; NaN si la curva no cubre el objetivo.
    """
    rows = []
    for (policy, lt, group), g in frontier.groupby(["policy", "lead_time", "group"], sort=False):
        g = g.sort_values("avg_inv_value")
        fr, inv = g["fill_rate"].to_numpy(), g["avg_inv_value"].to_numpy()
        fr = np.maximum.accumulate(fr)
        ok = fr.min() <= target <= fr.max()
        rows.append(
            {
                "policy": policy, "lead_time": lt, "group": group, "target_fill_rate": target,
                "avg_inv_value": float(np.interp(target, fr, inv)) if ok else np.nan,
            }
        )
    return pd.DataFrame(rows)


def recommendation_table(
    lead_time: int = config.DEFAULT_LEAD_TIME, service_level: float = config.DEFAULT_SERVICE_LEVEL
) -> pd.DataFrame:
    """Política sugerida HOY por SKU, con el pronóstico de las próximas 4 semanas."""
    fut = pd.read_parquet(config.FORECAST_FUTURE)
    cv = pd.read_parquet(config.CV_LGBM)
    abc = pd.read_parquet(config.ABC_XYZ).set_index("unique_id")
    price = pd.read_parquet(config.FUTURE_EXOG).groupby("unique_id")["sell_price"].first()

    fut = fut.sort_values(["unique_id", "ds"])
    g = fut.groupby("unique_id")["LightGBM"]
    tab = pd.DataFrame(
        {
            "forecast_28d": g.sum(),
            "forecast_daily": g.mean(),
            "demand_lead_time": g.apply(lambda s: s.iloc[:lead_time].sum()),
        }
    )
    tab["sigma_error"] = (cv["y"] - cv["LightGBM"]).groupby(cv["unique_id"]).std(ddof=1)
    tab["price"] = price
    tab["safety_stock"] = np.ceil(safety_stock(tab["sigma_error"], lead_time, service_level))
    tab["rop"] = np.ceil(reorder_point(tab["demand_lead_time"], tab["safety_stock"]))
    tab["lot"] = lot_size(tab["forecast_daily"], tab["price"])
    tab["order_up_to"] = tab["rop"] + tab["lot"]
    tab["lead_time"], tab["service_level"] = lead_time, service_level
    meta = pd.read_parquet(config.SALES_LONG, columns=["unique_id", "item_id", "dept_id"])
    meta = meta.drop_duplicates("unique_id").set_index("unique_id")
    meta = meta.join(abc[["abc", "xyz", "segment", "pattern"]])
    return meta.join(tab, how="inner").reset_index()


def main() -> None:
    inputs = load_inputs()
    default = (config.DEFAULT_LEAD_TIME, config.DEFAULT_SERVICE_LEVEL)

    scenarios, by_sku = run_grid(inputs, config.LEAD_TIMES, config.SERVICE_LEVELS, keep_sku={default})
    scenarios.to_parquet(config.SIM_SCENARIOS, index=False)
    by_sku.join(inputs["abc"][["abc", "xyz", "segment"]], on="unique_id").to_parquet(
        config.SIM_BY_SKU, index=False
    )

    frontier, _ = run_grid(inputs, config.LEAD_TIMES, FRONTIER_LEVELS)
    frontier.to_parquet(config.SIM_FRONTIER, index=False)

    table = recommendation_table()
    table.to_parquet(config.POLICY_TABLE, index=False)

    pd.set_option("display.width", 200)
    view = scenarios[(scenarios["lead_time"] == default[0]) & (scenarios["service_level"] == default[1])]
    print(f"Simulación — lead time {default[0]} días, nivel de servicio objetivo {default[1]:.0%}:")
    print(
        view[["policy", "group", "fill_rate", "skus_meeting_target", "avg_inv_value", "days_of_supply"]]
        .round(3).to_string(index=False)
    )
    need = inventory_at_fill_rate(frontier, config.DEFAULT_SERVICE_LEVEL)
    piv = need[need["lead_time"] == default[0]].pivot(
        index="group", columns="policy", values="avg_inv_value"
    )
    print(f"\nInventario promedio (USD a costo) para ALCANZAR {default[1]:.0%} de fill rate:")
    print(piv.round(0).to_string())


if __name__ == "__main__":
    main()
