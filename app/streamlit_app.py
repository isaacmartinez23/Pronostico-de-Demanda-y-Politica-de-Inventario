"""App de Streamlit: pronóstico de 4 semanas e inventario sugerido por SKU.

Uso: streamlit run app/streamlit_app.py   (o `make app`)

Solo lee los parquet de app/data/ (generados con `make app-data`): no entrena ni necesita
los datos de M5. La política se recalcula en vivo con las fórmulas de src/inventory/policy.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.inventory import policy as pol  # noqa: E402
from src.utils import palette as c  # noqa: E402

DATA = ROOT / "app" / "data"
MODEL = "LightGBM"
CLASSIC, NAIVE, LGBM = pol.POLICIES
POLICY_COLORS = {CLASSIC: c.ORANGE, NAIVE: c.AQUA, LGBM: c.BLUE}
METHODS = {"sqrt": "Fórmula de libro · z·σ·√L", "acumulado": "σ del error acumulado · z·σ(L)"}

st.set_page_config(page_title="Pronóstico de demanda e inventario", page_icon="📦", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def styled(fig: go.Figure, height: int = 360, unified: bool = True) -> go.Figure:
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor=c.SURFACE,
        plot_bgcolor=c.SURFACE,
        font={"family": "Segoe UI, system-ui, sans-serif", "color": c.INK, "size": 13},
        margin={"l": 55, "r": 20, "t": 40, "b": 45},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "traceorder": "normal"},
        hovermode="x unified" if unified else "closest",
    )
    fig.update_xaxes(showgrid=False, linecolor=c.AXIS, tickfont={"color": c.INK_SECONDARY}, automargin=True)
    fig.update_yaxes(gridcolor=c.GRID, zeroline=False, tickfont={"color": c.INK_SECONDARY}, automargin=True)
    return fig


def policy_table(skus: pd.DataFrame, lead_time: int, service_level: float, method: str) -> pd.DataFrame:
    """Safety stock, punto de reorden y nivel objetivo de cada SKU para los parámetros elegidos."""
    z = pol.z_score(service_level)
    if method == "sqrt":
        sigma_l = skus["sigma_error"] * np.sqrt(lead_time)
    else:
        sigma_l = skus[f"sigma_lead_{lead_time}"]
    out = skus.copy()
    out["demand_lead"] = skus[f"demand_lead_{lead_time}"]
    out["sigma_l"] = sigma_l
    out["safety_stock"] = np.ceil(np.maximum(z * sigma_l, 0.0))
    out["rop"] = np.ceil(pol.reorder_point(out["demand_lead"], out["safety_stock"]))
    out["order_up_to"] = out["rop"] + out["lot"]
    out["ss_value"] = out["safety_stock"] * out["price"] * config.UNIT_COST_RATIO
    return out


# ------------------------------------------------------------------------------- datos

skus = load("skus")
history, backtest, forecast = load("history"), load("backtest"), load("forecast")

# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Política de inventario")
    lead_time = st.radio("Lead time (días)", config.LEAD_TIMES, index=1, horizontal=True)
    service_pct = st.slider("Nivel de servicio objetivo (%)", 80.0, 99.5, 95.0, step=0.5)
    method = st.selectbox("Cálculo del safety stock", list(METHODS), format_func=METHODS.get)
    service_level = service_pct / 100

    st.header("Filtros")
    f_abc = st.multiselect("Clase ABC (ingreso)", ["A", "B", "C"], default=["A", "B", "C"])
    f_xyz = st.multiselect("Clase XYZ (variabilidad)", ["X", "Y", "Z"], default=["X", "Y", "Z"])
    f_dept = st.multiselect("Departamento", sorted(skus["dept_id"].unique()),
                            default=sorted(skus["dept_id"].unique()))
    st.caption(
        "Datos: M5 Forecasting (Walmart), categoría FOODS en la tienda CA_1. "
        "Pronóstico del 2016-05-23 al 2016-06-19."
    )

table = policy_table(skus, lead_time, service_level, method)
view = table[table["abc"].isin(f_abc) & table["xyz"].isin(f_xyz) & table["dept_id"].isin(f_dept)]

st.title("Pronóstico de demanda y política de inventario")
st.caption(
    "¿Cuánto vamos a vender de cada producto las próximas 4 semanas y cuánto inventario "
    "necesitamos para cumplir el nivel de servicio sin sobre-stock?"
)
if view.empty:
    st.warning("Ningún SKU cumple los filtros seleccionados.")
    st.stop()

tab_sku, tab_port, tab_res, tab_met = st.tabs(["SKU", "Portafolio", "Resultados", "Metodología"])

# --------------------------------------------------------------------------------- SKU

with tab_sku:
    labels = {
        r.unique_id: f"{r.item_id} · {r.segment} · {r.pattern}" for r in view.itertuples()
    }
    left, right = st.columns([3, 2])
    uid = left.selectbox("SKU (ordenados por ingreso)", list(labels), format_func=labels.get)
    row = view.set_index("unique_id").loc[uid]
    on_hand = right.number_input(
        "Inventario disponible hoy (unidades)", min_value=0, value=int(row["rop"]), step=1,
        key=f"onhand-{uid}-{lead_time}-{service_pct}-{method}",
        help="Parte del punto de reorden; cámbialo para ver qué pedido sugiere la política.",
    )
    order = float(pol.order_quantity(on_hand, row["rop"], row["lot"]))

    k = st.columns(5)
    k[0].metric("Pronóstico 4 semanas", f"{row['forecast_28d']:,.0f} u")
    k[1].metric(f"Demanda en {lead_time} días", f"{row['demand_lead']:,.0f} u")
    k[2].metric("Safety stock", f"{row['safety_stock']:,.0f} u")
    k[3].metric("Punto de reorden", f"{row['rop']:,.0f} u")
    k[4].metric("Pedido sugerido hoy", f"{order:,.0f} u")
    if order > 0:
        st.success(
            f"Pedir **{order:,.0f} unidades** hoy: el inventario ({on_hand:,} u) está en o por "
            f"debajo del punto de reorden ({row['rop']:,.0f} u) y se repone hasta "
            f"{row['order_up_to']:,.0f} u."
        )
    else:
        st.info(
            f"No pedir hoy: el inventario ({on_hand:,} u) está por encima del punto de reorden "
            f"({row['rop']:,.0f} u)."
        )

    hist = history[history["unique_id"] == uid].tail(config.N_WINDOWS * config.HORIZON)
    back = backtest[backtest["unique_id"] == uid]
    fut = forecast[forecast["unique_id"] == uid]

    st.subheader("Venta real, pronóstico de backtesting y próximas 4 semanas")
    level = st.radio("Intervalo de predicción", [80, 95], horizontal=True, format_func=lambda v: f"{v}%")
    fig = go.Figure()
    fig.add_scatter(x=hist["ds"], y=hist["y"], name="Venta real", line={"color": c.MUTED, "width": 1.5})
    fig.add_scatter(x=back["ds"], y=back[MODEL], name="Pronóstico (backtesting)",
                    line={"color": c.BLUE, "width": 2})
    fig.add_scatter(x=fut["ds"], y=fut[f"{MODEL}-hi-{level}"], line={"width": 0}, showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=fut["ds"], y=fut[f"{MODEL}-lo-{level}"], name=f"Intervalo {level}%",
                    line={"width": 0}, fill="tonexty", fillcolor="rgba(235,104,52,0.18)",
                    hoverinfo="skip")
    fig.add_scatter(x=fut["ds"], y=fut[MODEL], name="Pronóstico 4 semanas",
                    line={"color": c.ORANGE, "width": 2})
    fig.update_yaxes(title_text="Unidades por día", rangemode="tozero")
    st.plotly_chart(styled(fig), width="stretch", theme=None)

    st.subheader("Proyección del inventario si la demanda sigue el pronóstico")
    demand = fut[MODEL].to_numpy()[:, None]
    flat = np.ones_like(demand)
    sim = pol.simulate_policy(demand, flat * row["rop"], flat * row["lot"], lead_time,
                              initial_on_hand=np.array([float(on_hand)]))
    ordered = sim.orders[:, 0] > 0
    fig = go.Figure()
    fig.add_scatter(x=fut["ds"], y=sim.on_hand[:, 0], name="Inventario al cierre",
                    line={"color": c.BLUE, "width": 2})
    fig.add_scatter(x=fut["ds"], y=flat[:, 0] * row["rop"], name="Punto de reorden",
                    line={"color": c.INK_SECONDARY, "width": 1, "dash": "dash"})
    fig.add_scatter(x=fut["ds"][ordered], y=sim.on_hand[ordered, 0], name="Se coloca un pedido",
                    mode="markers", customdata=sim.orders[ordered, 0],
                    hovertemplate="Pedido: %{customdata:,.0f} u<extra></extra>",
                    marker={"color": c.ORANGE, "size": 10, "line": {"color": c.SURFACE, "width": 2}})
    fig.update_yaxes(title_text="Unidades", rangemode="tozero")
    st.plotly_chart(styled(fig, height=300), width="stretch", theme=None)
    lost = sim.lost.sum()
    st.caption(
        f"Con {lead_time} días de lead time, cada pedido llega {lead_time} días después de colocarse. "
        + (f"Con este inventario inicial se perderían ~{lost:,.0f} unidades de venta."
           if lost > 0.5 else "Con este inventario inicial no se proyectan quiebres.")
    )

    with st.expander("Cómo se calcula"):
        sigma_txt = (
            f"σ diario {row['sigma_error']:.2f} × √{lead_time}" if method == "sqrt"
            else f"σ del error acumulado en {lead_time} días"
        )
        st.markdown(
            f"""
| Paso | Cálculo | Resultado |
| --- | --- | --- |
| Demanda esperada en el lead time | suma del pronóstico de los próximos {lead_time} días | {row['demand_lead']:,.1f} u |
| Incertidumbre en el lead time | {sigma_txt} | {row['sigma_l']:,.1f} u |
| Factor de seguridad | z para {service_pct:.1f}% | {pol.z_score(service_level):.3f} |
| Safety stock | z × incertidumbre, redondeado hacia arriba | {row['safety_stock']:,.0f} u |
| Punto de reorden | demanda en lead time + safety stock | {row['rop']:,.0f} u |
| Lote | EOQ acotado a {config.MAX_COVER_DAYS} días de demanda | {row['lot']:,.0f} u |

La incertidumbre se mide con los errores del modelo en 6 ventanas de backtesting de este SKU.
"""
        )
    p = st.columns(3)
    p[0].metric("WAPE en backtesting", f"{row['wape_backtest']:.0%}" if pd.notna(row["wape_backtest"]) else "—",
                help="Error absoluto del pronóstico sobre la demanda total, en las 12 semanas de evaluación.")
    p[1].metric("Sesgo en backtesting", f"{row['bias_backtest']:+.0%}" if pd.notna(row["bias_backtest"]) else "—",
                help="Positivo: el modelo sobre-pronostica este SKU.")
    p[2].metric("Fill rate en la simulación", f"{row['fill_rate_sim']:.1%}" if pd.notna(row["fill_rate_sim"]) else "—",
                help="Con lead time de 7 días, objetivo 95% y la fórmula de libro.")

# -------------------------------------------------------------------------- portafolio

with tab_port:
    k = st.columns(4)
    k[0].metric("SKUs", f"{len(view):,}")
    k[1].metric("Pronóstico 4 semanas", f"{view['forecast_28d'].sum():,.0f} u")
    k[2].metric("Safety stock", f"{view['safety_stock'].sum():,.0f} u")
    k[3].metric("Valor del safety stock", f"USD {view['ss_value'].sum():,.0f}",
                help=f"A costo ({config.UNIT_COST_RATIO:.0%} del precio de venta).")

    st.subheader("Safety stock por segmento ABC/XYZ")
    by_seg = view.groupby("segment").agg(valor=("ss_value", "sum"), skus=("unique_id", "size"))
    fig = go.Figure(
        go.Bar(x=by_seg.index, y=by_seg["valor"], marker_color=c.BLUE, customdata=by_seg["skus"],
               hovertemplate="%{x}: USD %{y:,.0f} · %{customdata} SKUs<extra></extra>")
    )
    fig.update_yaxes(title_text="USD a costo")
    st.plotly_chart(styled(fig, height=280, unified=False), width="stretch", theme=None)

    st.subheader("Política por SKU")
    cols = {
        "item_id": "SKU", "segment": "Segmento", "pattern": "Patrón", "forecast_28d": "Pronóstico 28d",
        "demand_lead": "Demanda en lead time", "safety_stock": "Safety stock", "rop": "Punto de reorden",
        "lot": "Lote", "order_up_to": "Reponer hasta", "wape_backtest": "WAPE",
    }
    shown = view[list(cols)].rename(columns=cols)
    st.dataframe(
        shown, width="stretch", hide_index=True, height=420,
        column_config={
            "Pronóstico 28d": st.column_config.NumberColumn(format="%.0f"),
            "Demanda en lead time": st.column_config.NumberColumn(format="%.1f"),
            "WAPE": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.download_button(
        "Descargar CSV", shown.to_csv(index=False).encode("utf-8"),
        file_name=f"politica_L{lead_time}_NS{service_pct:.1f}.csv", mime="text/csv",
    )

# -------------------------------------------------------------------------- resultados

with tab_res:
    wr = load("metrics_wrmsse").query("level == 'WRMSSE'").groupby("model")["value"].mean().sort_values()
    frontier, scenarios = load("sim_frontier"), load("sim_scenarios")
    need = pol.inventory_at_fill_rate(frontier.query("ss_method == 'sqrt' and group == 'Total'"), 0.95)
    need = need.query("lead_time == 7").set_index("policy")["avg_inv_value"]
    best = wr.drop(MODEL).idxmin()
    fill_a = scenarios.query(
        "policy == @LGBM and ss_method == 'sqrt' and lead_time == 7 and service_level == 0.95 and group == 'A'"
    )["fill_rate"].iloc[0]

    st.caption("Backtesting en 3 ventanas de 28 días (2016-02-29 → 2016-05-22) y simulación de la "
               "política sobre esas mismas 12 semanas.")
    k = st.columns(4)
    k[0].metric("WRMSSE vs. SeasonalNaive", f"{wr[MODEL] / wr['SeasonalNaive'] - 1:+.0%}")
    k[1].metric(f"WRMSSE vs. {best}", f"{wr[MODEL] / wr[best] - 1:+.0%}", help="El mejor baseline.")
    k[2].metric("Inventario a 95% de fill rate", f"{need[LGBM] / need[CLASSIC] - 1:+.0%}",
                help="LightGBM vs. política clásica (media móvil), lead time de 7 días.")
    k[3].metric("Fill rate clase A", f"{fill_a:.1%}", help="Lead time 7 días, objetivo 95%.")

    left, right = st.columns(2)
    with left:
        st.subheader("WRMSSE por modelo")
        colors = [c.BLUE if m == MODEL else c.AXIS for m in wr.index]
        fig = go.Figure(go.Bar(x=wr.to_numpy(), y=wr.index, orientation="h", marker_color=colors,
                               text=[f"{v:.3f}" for v in wr], textposition="outside",
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig = styled(fig, height=340, unified=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=c.GRID, range=[0, wr.max() * 1.15],
                         title_text="Menor es mejor")
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.subheader("Curva servicio–inventario")
        a, b = st.columns(2)
        lt_sel = a.radio("Lead time", config.LEAD_TIMES, index=1, horizontal=True, key="res-lt")
        grp = b.radio("Grupo", ["Total", "A", "B", "C"], horizontal=True, key="res-grp")
        fig = go.Figure()
        sub = frontier.query("ss_method == 'sqrt' and group == @grp and lead_time == @lt_sel")
        for policy, g in sub.groupby("policy", sort=False):
            g = g.sort_values("avg_inv_value").query("fill_rate >= 0.85")
            fig.add_scatter(x=g["avg_inv_value"], y=g["fill_rate"] * 100, name=policy, mode="lines",
                            line={"color": POLICY_COLORS[policy], "width": 2},
                            hovertemplate="USD %{x:,.0f} · %{y:.1f}%<extra>" + policy + "</extra>")
        fig.add_hline(y=95, line={"color": c.INK_SECONDARY, "width": 1, "dash": "dash"})
        fig.update_xaxes(title_text="Inventario promedio (USD a costo)")
        fig.update_yaxes(title_text="Fill rate (%)")
        st.plotly_chart(styled(fig, height=340, unified=False), width="stretch", theme=None)
        n = pol.inventory_at_fill_rate(sub, 0.95).set_index("policy")["avg_inv_value"]
        if n[[CLASSIC, LGBM]].notna().all():
            diff = n[LGBM] / n[CLASSIC] - 1
            st.caption(f"Para 95% de fill rate: USD {n[CLASSIC]:,.0f} con la política clásica y "
                       f"USD {n[LGBM]:,.0f} con LightGBM: {abs(diff):.0%} {'menos' if diff < 0 else 'más'}.")
        else:
            st.caption("Alguna política no alcanza 95% de fill rate en este escenario.")

    st.subheader("¿Dónde gana el modelo? WAPE por segmento")
    seg = load("metrics_segment").query("grouping == 'segment'")
    wape = seg.pivot(index="group", columns="model", values="wape")
    base = wape.drop(columns=MODEL)
    comp = pd.DataFrame({
        "SKUs": seg.groupby("group")["n_skus"].first(), "WAPE LightGBM": wape[MODEL],
        "Mejor baseline": base.idxmin(axis=1), "WAPE baseline": base.min(axis=1),
    })
    comp["Diferencia"] = comp["WAPE LightGBM"] / comp["WAPE baseline"] - 1
    st.dataframe(
        comp.reset_index(names="Segmento"), width="stretch", hide_index=True,
        column_config={
            "WAPE LightGBM": st.column_config.NumberColumn(format="%.3f"),
            "WAPE baseline": st.column_config.NumberColumn(format="%.3f"),
            "Diferencia": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption("Diferencia negativa: LightGBM es mejor. Gana en los segmentos de mayor ingreso (AX, AY) "
               "y pierde en la cola de baja rotación, donde una media móvil es suficiente.")

# ------------------------------------------------------------------------- metodología

with tab_met:
    st.markdown(
        f"""
### Datos
M5 Forecasting (Walmart): ventas diarias, precios y calendario de eventos. Se usa la categoría
**FOODS en la tienda CA_1**: {len(skus):,} SKUs con al menos un año de historia.

### Pronóstico
Un modelo global **LightGBM** (objetivo Tweedie) entrenado con `mlforecast`: lags 7/14/28 y medias
móviles, día de la semana, SNAP, eventos y precio relativo. Se compara contra Naive, SeasonalNaive,
media móvil, AutoETS y AutoARIMA con backtesting en 3 ventanas de 28 días.

### Política de inventario
Política (s, S) con revisión diaria y ventas perdidas:

- **Safety stock** = z · σ(error del pronóstico) · √(lead time)
- **Punto de reorden** = demanda esperada en el lead time + safety stock
- **Lote** = EOQ, acotado entre 1 y {config.MAX_COVER_DAYS} días de demanda

La política se **simula** contra la demanda real del periodo de evaluación y se compara con una
política clásica (media y desviación de los últimos 56 días) *a igual fill rate*.

### Limitaciones
- **Ventas, no demanda:** M5 registra lo vendido; un quiebre real aparece como venta cero.
- **Costos supuestos:** costo unitario = {config.UNIT_COST_RATIO:.0%} del precio, mantener = {config.HOLDING_RATE_ANNUAL:.0%} anual,
  USD {config.ORDER_COST:.0f} por pedido. Solo afectan el tamaño del lote.
- **Lead time fijo** y sin restricciones de capacidad, mínimos de compra ni caducidad.
- **z no es el fill rate:** el nivel de servicio objetivo fija la probabilidad de no quebrar en un
  ciclo; en la simulación el servicio de ciclo logrado queda por debajo del nominal.
- Los intervalos de predicción cubren menos de lo nominal (≈74% el de 80%, ≈89% el de 95%).
"""
    )
