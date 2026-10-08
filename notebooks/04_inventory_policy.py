# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 · Del pronóstico a la decisión de inventario
#
# Un pronóstico no vale por su error sino por la decisión que permite tomar. Aquí cada
# pronóstico se convierte en una política de inventario y se **simula** contra la demanda
# real de las 12 semanas de evaluación.
#
# **Política (s, S) con revisión diaria y ventas perdidas**
#
# | Elemento | Cálculo |
# | --- | --- |
# | Safety stock | `SS = z(nivel de servicio) · σ(error) · √L` |
# | Punto de reorden | `ROP = demanda esperada en el lead time + SS` |
# | Lote | EOQ, acotado entre 1 y 14 días de demanda (perecederos) |
# | Regla | Si la posición de inventario ≤ ROP, pedir hasta `ROP + lote` |
#
# **Tres políticas, misma mecánica y misma cadencia de actualización (cada 28 días):**
#
# - **Clásica:** media y desviación de la demanda de los últimos 56 días (lo que haría una
#   hoja de cálculo).
# - **SeasonalNaive** y **LightGBM:** demanda esperada = pronóstico; σ = error del pronóstico
#   medido en ventanas *anteriores* a la que se simula.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.inventory import policy as pol
from src.utils import plotting as pl

pl.use_style()
LT, SL = config.DEFAULT_LEAD_TIME, config.DEFAULT_SERVICE_LEVEL
CLASSIC, NAIVE, LGBM = pol.POLICIES
COLORS = {CLASSIC: pl.ORANGE, NAIVE: pl.AQUA, LGBM: pl.BLUE}

scenarios = pd.read_parquet(config.SIM_SCENARIOS)
frontier = pd.read_parquet(config.SIM_FRONTIER)
by_sku = pd.read_parquet(config.SIM_BY_SKU)
costs = pd.read_parquet(config.SIM_COSTS)
table = pd.read_parquet(config.POLICY_TABLE)

pd.DataFrame(
    {
        "Supuesto": ["Costo unitario", "Costo de mantener", "Costo fijo por pedido", "Cobertura máxima del lote"],
        "Valor": [
            f"{config.UNIT_COST_RATIO:.0%} del precio de venta",
            f"{config.HOLDING_RATE_ANNUAL:.0%} anual del costo",
            f"USD {config.ORDER_COST:.2f}",
            f"{config.MAX_COVER_DAYS} días",
        ],
    }
)

# %% [markdown]
# M5 no trae costos: los de arriba son **supuestos** y solo afectan el tamaño del lote. Más
# abajo se revisa cuánto dependen las conclusiones de ellos.

# %% [markdown]
# ## 1. Resultado con la fórmula de libro
#
# Lead time de 7 días y nivel de servicio objetivo de 95% (`z = 1.645`).

# %%
cols = ["fill_rate", "cycle_service", "skus_meeting_target", "avg_inv_value", "days_of_supply"]
at_target = scenarios.query("lead_time == @LT and service_level == @SL and ss_method == 'sqrt'")
view = at_target.set_index(["group", "policy"])[cols].loc[["Total", "A", "B", "C"]]
view.round(3)

# %% [markdown]
# Tres lecturas:
#
# 1. **La meta del proyecto se cumple:** con LightGBM la clase A alcanza 96.8% de fill rate
#    (objetivo ≥ 95%) y 87% de sus SKUs lo cumplen individualmente. La política clásica
#    también llega a 95.4% en clase A, con un inventario parecido.
# 2. **Fill rate y nivel de servicio de ciclo no son lo mismo.** `z` fija la probabilidad de
#    *no quebrar en un ciclo*; el fill rate mide el % de la demanda atendida. Con `z` de 95%,
#    el servicio de ciclo realmente logrado es 82% (clásica) y 90% (LightGBM): la fórmula de
#    libro promete más de lo que entrega.
# 3. **Esta tabla no sirve para comparar políticas**, porque cada una termina en un punto
#    distinto: una da más servicio con más inventario, otra menos con menos. La comparación
#    válida es *a igual servicio*.

# %% [markdown]
# ## 2. La comparación justa: curva servicio–inventario
#
# Para cada política se barre el factor de seguridad y se simula. El resultado es una curva:
# cuánto inventario promedio cuesta cada punto de fill rate.

# %%
def curve(policy: str, group: str = "Total", lead_time: int = LT, method: str = "sqrt") -> pd.DataFrame:
    g = frontier.query(
        "policy == @policy and group == @group and lead_time == @lead_time and ss_method == @method"
    )
    return g.sort_values("avg_inv_value")


need = pol.inventory_at_fill_rate(frontier, SL)
need_sqrt = need.query("ss_method == 'sqrt'")


def needed(policy: str, group: str = "Total", lead_time: int = LT) -> float:
    row = need_sqrt.query("policy == @policy and group == @group and lead_time == @lead_time")
    return float(row["avg_inv_value"].iloc[0])


saving = needed(LGBM) / needed(CLASSIC) - 1
saving_a = needed(LGBM, "A") / needed(CLASSIC, "A") - 1

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for policy in pol.POLICIES:
    c = curve(policy).query("fill_rate >= 0.86")
    ax.plot(c["avg_inv_value"] / 1000, c["fill_rate"] * 100, color=COLORS[policy], label=policy)
ax.axhline(SL * 100, color=pl.INK_SECONDARY, linewidth=1, linestyle="--")
for policy in (CLASSIC, LGBM):
    x = needed(policy) / 1000
    ax.plot([x], [SL * 100], marker="o", markersize=8, color=COLORS[policy],
            markeredgecolor=pl.SURFACE, markeredgewidth=2)
    left = policy == LGBM
    ax.annotate(f"USD {x:.1f}k", (x, SL * 100), xytext=(-8 if left else 8, -16),
                textcoords="offset points", ha="right" if left else "left", color=pl.INK_SECONDARY)
ax.set_title(
    f"A igual fill rate ({SL:.0%}), la política con LightGBM necesita {abs(saving):.0%} menos inventario",
    pad=30,
)
ax.set_xlabel("Inventario promedio (miles de USD a costo)")
ax.set_ylabel("Fill rate (% de la demanda atendida, en valor)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, borderaxespad=0.2)
pl.save(fig, "16_frontera_servicio_inventario")
plt.show()

print(f"Total  : {needed(CLASSIC):,.0f} → {needed(LGBM):,.0f} USD ({saving:+.1%})")
print(f"Clase A: {needed(CLASSIC, 'A'):,.0f} → {needed(LGBM, 'A'):,.0f} USD ({saving_a:+.1%})")

# %% [markdown]
# **Resultado principal.** Para atender el 95% de la demanda con 7 días de lead time, la
# política basada en LightGBM necesita USD 63.7k de inventario promedio contra USD 73.8k de
# la política clásica: **14% menos** (11% menos en la clase A). SeasonalNaive no aporta nada
# sobre la política clásica: un pronóstico solo ayuda al inventario si es mejor que el
# promedio.

# %% [markdown]
# ## 3. ¿Se sostiene en otros escenarios?
#
# Inventario necesario para alcanzar cada fill rate objetivo, por lead time. `NaN` significa
# que la política no alcanza ese servicio con ningún factor de seguridad del barrido.

# %%
rows = []
for target in (0.90, 0.95, 0.98):
    n = pol.inventory_at_fill_rate(frontier, target).query("ss_method == 'sqrt' and group == 'Total'")
    piv = n.pivot(index="lead_time", columns="policy", values="avg_inv_value")
    piv["objetivo"] = target
    rows.append(piv.reset_index())
sens = pd.concat(rows).set_index(["objetivo", "lead_time"])[list(pol.POLICIES)]
sens["LightGBM vs clásica"] = sens[LGBM] / sens[CLASSIC] - 1
sens.round({CLASSIC: 0, NAIVE: 0, LGBM: 0, "LightGBM vs clásica": 3})

# %%
red = sens.xs(SL, level="objetivo")["LightGBM vs clásica"] * 100

fig, ax = plt.subplots(figsize=(6, 3.4))
bars = ax.bar([f"{lt} días" for lt in red.index], red.to_numpy(), width=0.5, color=pl.BLUE)
ax.bar_label(bars, labels=[f"{v:.0f}%" for v in red], padding=3, color=pl.INK_SECONDARY)
ax.axhline(0, color=pl.AXIS, linewidth=1)
ax.set_title(f"El ahorro crece con el lead time (fill rate {SL:.0%})")
ax.set_xlabel("Lead time")
ax.set_ylabel("Inventario vs. política clásica (%)")
ax.set_ylim(red.min() * 1.25, 0)
pl.save(fig, "17_ahorro_por_lead_time")
plt.show()

# %% [markdown]
# - **El ahorro crece con el lead time:** 7% con 3 días, 14% con 7 y 22% con 14. Cuanto más
#   lejos hay que mirar, más vale un pronóstico que conoce el calendario.
# - **Y con la exigencia de servicio:** a 90% de fill rate la diferencia es de solo 2–4%
#   (casi cualquier política llega); a 98% la política clásica ya no llega con ningún factor
#   de seguridad del barrido.
# - Con 14 días de lead time, ninguna política alcanza 98%.

# %% [markdown]
# ## 4. Escenarios: lead time × nivel de servicio (clase A, LightGBM)

# %%
grid = scenarios.query("policy == @LGBM and ss_method == 'sqrt' and group == 'A'")
grid.pivot(index="lead_time", columns="service_level",
           values=["fill_rate", "cycle_service", "avg_inv_value"]).round(3)

# %% [markdown]
# En la clase A el fill rate supera 95% en ocho de los nueve escenarios; solo queda corto con
# 14 días de lead time y `z` de 90%. Subir el nivel nominal de 90% a 99% cuesta entre 14% y
# 19% más inventario.

# %% [markdown]
# ## 5. ¿Importa cómo se calcula el safety stock?
#
# La fórmula de libro supone errores diarios independientes y normales. Se comparan tres
# formas de medir la incertidumbre en el lead time:
#
# | Método | Safety stock |
# | --- | --- |
# | `sqrt` | `z · σ(error diario) · √L` — la fórmula de libro |
# | `acumulado` | `z · σ(error acumulado en L días)` — mide la correlación entre días |
# | `empirico` | cuantil empírico del error acumulado en L días — sin supuesto de normalidad |

# %%
methods = scenarios.query("lead_time == @LT and service_level == @SL and group == 'Total'")
calib = methods.pivot(index="ss_method", columns="policy", values="cycle_service")[list(pol.POLICIES)]
eff = need.query("lead_time == @LT and group == 'Total'").pivot(
    index="ss_method", columns="policy", values="avg_inv_value"
)[list(pol.POLICIES)]
pd.concat({"Servicio de ciclo logrado (nominal 95%)": calib.round(3),
           "Inventario para 95% de fill rate (USD)": eff.round(0)}, axis=1)

# %%
nominal = scenarios.query("lead_time == @LT and group == 'Total' and policy == @LGBM")
piv = nominal.pivot(index="service_level", columns="ss_method", values="cycle_service")

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.plot([88, 100], [88, 100], color=pl.AXIS, linewidth=1, linestyle="--")
ax.annotate("logrado = nominal", (92.6, 92.9), ha="center", va="bottom", rotation=25,
            color=pl.MUTED, fontsize=8.5)
for method, color in zip(pol.SS_METHODS, pl.CATEGORICAL, strict=False):
    ax.plot(piv.index * 100, piv[method] * 100, marker="o", color=color,
            label=f"{method} ({pol.SS_METHODS[method]})")
ax.set_title("Servicio de ciclo: nominal vs. logrado (LightGBM)")
ax.set_xlabel("Nivel de servicio nominal (%)")
ax.set_ylabel("Nivel de servicio de ciclo logrado (%)")
ax.set_xticks([90, 95, 99])
ax.set_xlim(88, 100)
ax.set_ylim(80, 100)
ax.legend(loc="lower right", fontsize=9)
pl.save(fig, "18_servicio_nominal_vs_logrado")
plt.show()

# %% [markdown]
# Un resultado que no esperaba: **los métodos más sofisticados no hacen la política más
# eficiente.**
#
# - **Eficiencia (inventario para 95% de fill rate):** con LightGBM, la fórmula de libro y el
#   σ acumulado necesitan lo mismo (USD 63.7k); el cuantil empírico necesita 4% *más*, porque
#   estimar un cuantil extremo por SKU con tan pocas muestras mete ruido.
# - **Calibración (¿se cumple lo que promete `z`?):** aquí sí hay diferencia. Con la fórmula
#   de libro el servicio de ciclo queda en 90% cuando se pide 95%; con σ acumulado sube a
#   92%. Ninguno llega a lo nominal.
#
# Es decir: el método de safety stock mueve *en qué punto* de la curva cae la política, no la
# curva. Lo que mueve la curva es la calidad del pronóstico. En la práctica conviene fijar
# el factor de seguridad **por simulación** contra el fill rate deseado, en vez de confiar en
# la lectura nominal de `z`.

# %% [markdown]
# ## 6. Por clase ABC

# %%
by_class = need_sqrt.query("lead_time == @LT").pivot(index="group", columns="policy", values="avg_inv_value")
by_class = by_class.loc[["A", "B", "C", "Total"], list(pol.POLICIES)]
by_class["LightGBM vs clásica"] = by_class[LGBM] / by_class[CLASSIC] - 1
by_class.round({CLASSIC: 0, NAIVE: 0, LGBM: 0, "LightGBM vs clásica": 3})

# %% [markdown]
# El ahorro relativo es mayor en la clase B (25%) que en la A (11%), y en la clase C la
# política clásica no alcanza 95% con ningún factor de seguridad. En dinero, la mayor parte
# sigue viniendo de la clase A (USD 5.4k de los 10.1k).

# %% [markdown]
# ## 7. ¿Qué SKUs no llegan al objetivo?

# %%
classes = pd.read_parquet(config.ABC_XYZ)[["unique_id", "pattern"]]
sku = by_sku.query("policy == @LGBM and ss_method == 'sqrt'").merge(classes, on="unique_id")
sku = sku[sku["demand"] > 0].assign(cumple=lambda d: d["fill_rate"] >= SL)
miss = sku.groupby("pattern").agg(
    skus=("unique_id", "size"), cumplen=("cumple", "mean"), fill_rate_mediano=("fill_rate", "median"),
    demanda_diaria=("daily_demand", "median"),
)
miss.round(3)

# %%
sku.groupby("abc").agg(skus=("unique_id", "size"), cumplen=("cumple", "mean"),
                       fill_rate_p10=("fill_rate", lambda s: s.quantile(0.10))).round(3)

# %% [markdown]
# La mediana de fill rate por SKU es 100% en todos los patrones: la mayoría de los SKUs no
# quiebra nunca en las 12 semanas, y el incumplimiento se concentra en una minoría. El
# patrón más difícil es el **errático** (ventas frecuentes de tamaño muy variable): solo 73%
# de esos SKUs cumple. En la clase A cumple el 87%, y el 10% peor queda en 93% de fill rate
# o menos.

# %% [markdown]
# ## 8. Sensibilidad a los supuestos de costo
#
# El costo por pedido y el costo de mantener solo entran al tamaño del lote (EOQ). ¿Cambia
# la conclusión si los supuestos son otros?

# %%
costs.pivot(index="holding_rate_annual", columns="order_cost", values="reduction").round(3)

# %% [markdown]
# Duplicar o reducir a la mitad el costo por pedido, o mover el costo de mantener entre 15% y
# 35% anual, deja el ahorro entre 12.8% y 13.7%. La conclusión no depende de los supuestos de
# costo, porque afectan por igual el lote de todas las políticas.

# %% [markdown]
# ## 9. La tabla que recibe el planeador
#
# Política sugerida hoy para cada SKU, con el pronóstico de las próximas 4 semanas
# (`data/processed/policy_table.parquet`; es lo que muestra la app).

# %%
show = ["unique_id", "segment", "pattern", "forecast_28d", "demand_lead_time", "safety_stock",
        "rop", "lot", "order_up_to"]
table.sort_values("forecast_28d", ascending=False)[show].head(10).round(1)
