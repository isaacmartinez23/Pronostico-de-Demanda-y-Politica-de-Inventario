# Pronóstico de demanda y política de inventario

[![CI](https://github.com/isaacmartinez23/Pronostico-de-Demanda-y-Politica-de-Inventario/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacmartinez23/Pronostico-de-Demanda-y-Politica-de-Inventario/actions/workflows/ci.yml)
[![Dashboard](https://img.shields.io/badge/dashboard-en%20vivo-3d3a9e)](https://isaacmartinez.space/Pronostico-de-Demanda-y-Politica-de-Inventario/)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

> **Pregunta de negocio:** ¿cuánto vamos a vender de cada producto las próximas 4 semanas y
> cuánto inventario debemos tener para cumplir un nivel de servicio del 95% sin sobre-stock?

Proyecto de supply chain que va **del pronóstico a la decisión**: un modelo de demanda por
SKU-tienda se traduce en safety stock, punto de reorden y cantidad a pedir, y la política
resultante se valida simulándola contra demanda real que el modelo no vio.

**Datos:** [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
(Walmart) — categoría FOODS en la tienda CA_1: 1,429 SKUs, 2011–2016.

## Resultado

**A igual nivel de servicio (95% de fill rate, lead time de 7 días), la política basada en
el modelo necesita 14% menos inventario promedio que una política clásica de media móvil.**

![Curva servicio–inventario](reports/figures/16_frontera_servicio_inventario.png)

| Métrica de éxito | Resultado |
| --- | --- |
| El modelo supera a SeasonalNaive en WRMSSE | **0.528 vs 0.785 (−33%)**, en las 3 ventanas. Contra el mejor baseline (AutoETS, 0.608): −13% |
| Fill rate ≥ 95% en SKUs clase A | **96.8%** en la simulación (87% de los SKUs A lo cumple individualmente) |
| Inventario a igual servicio vs. política clásica | **−14%** total (USD 63.7k vs 73.8k), −11% clase A, −25% clase B |
| Reproducible con `make all` | Sí; CI corre el pipeline completo sobre datos sintéticos |

Resumen de una página: [reports/executive_summary.md](reports/executive_summary.md).

## Lo que encontré

1. **El modelo gana donde hay volumen y pierde en la cola.** Mejora el WAPE en los
   segmentos AX y AY (74% del ingreso) y empeora hasta 16% en CZ, donde una media móvil es
   mejor. La recomendación no es "ML para todo" sino segmentar.
2. **El ahorro de inventario crece con el lead time:** −7% con 3 días, −14% con 7 y −22% con
   14. A 90% de fill rate la diferencia es de solo 2–4%.
3. **`z` no entrega lo que promete.** Con el factor de seguridad para 95%, el nivel de
   servicio de ciclo realmente logrado fue 90% (y 82% con la política clásica). Fill rate y
   servicio de ciclo son métricas distintas y conviene reportar ambas.
4. **Sofisticar el safety stock no lo hizo más eficiente.** Probé σ del error acumulado en el
   lead time y cuantiles empíricos del error en lugar de `z·σ·√L`: mejor calibración, mismo
   inventario para el mismo servicio. Lo que mueve la curva es el pronóstico.
5. **Los intervalos de predicción quedan cortos:** 74% de cobertura real para el intervalo de
   80% y 89% para el de 95%, medidos fuera de muestra. No cumple el objetivo de ±5 puntos
   que me había fijado.
6. **Una variable "obvia" empeoraba el modelo.** `item_id` como categórica (1,429 niveles)
   subía el WRMSSE 5.6% en las ventanas de calibración; quedó fuera.

Detalle en los notebooks:
[01 EDA](notebooks/01_eda.ipynb) ·
[02 Baselines](notebooks/02_baselines.ipynb) ·
[03 LightGBM](notebooks/03_lightgbm.ipynb) ·
[04 Política de inventario](notebooks/04_inventory_policy.ipynb)

## Dashboard

**En vivo: <https://isaacmartinez.space/Pronostico-de-Demanda-y-Politica-de-Inventario/>**

[![Resumen del dashboard](docs/img/dashboard_resumen.png)](https://isaacmartinez.space/Pronostico-de-Demanda-y-Politica-de-Inventario/)

Cinco páginas, cada una abre con la pregunta que responde y una lectura en una frase calculada
con los filtros vigentes:

| Página | Qué responde |
| --- | --- |
| Resumen | Cuánto inventario hace falta para el 95% de servicio y cuánto ahorra el modelo; curva servicio–inventario |
| Pronóstico | Error contra los baselines, venta real contra pronóstico por departamento, dónde gana y dónde pierde |
| Inventario | Inventario por política a igual fill rate, escenarios de lead time y nivel de servicio, servicio nominal contra logrado |
| Producto | Pronóstico diario con intervalo, safety stock, punto de reorden, pedido sugerido y proyección del inventario |
| Portafolio | Política de todos los productos, valor del colchón por segmento y descarga en CSV |

Filtros de lead time, nivel de servicio, método de safety stock, clase ABC/XYZ y departamento.

![Vista por producto](docs/img/dashboard_producto.png)

Es un solo HTML sin servidor ni librerías externas, con el mismo diseño que el
[dashboard de precios de PROFECO](https://isaacmartinez.space/analisis_profeco/) del portafolio.
`make dashboard` extrae los resultados derivados a `reports/dashboard/datos.json` (versionado)
y genera `reports/dashboard/index.html`; GitHub Actions lo publica en GitHub Pages con cada
cambio. Para abrirlo en local:

```bash
python -m src.dashboard.generar
```

```bash
python -m http.server 8766 --directory reports/dashboard
```

**Datos que publica.** Las reglas de M5 permiten usar los datos con fines no comerciales y de
investigación, pero no redistribuirlos. El dashboard solo incluye resultados derivados:
pronósticos, márgenes de los intervalos, métricas, resultados de la simulación y la venta
sumada por departamento. Ninguna venta diaria ni precio de un producto llega a la página, y
`src/dashboard/datos.py` falla si aparece una columna con ese contenido.

## Arquitectura

```mermaid
flowchart LR
    A[M5: ventas, precios,<br>calendario] --> B[make_dataset<br>formato largo + DuckDB]
    B --> C[abc_xyz<br>ABC/XYZ y patrón]
    B --> D[baselines<br>statsforecast]
    B --> E[ml_model<br>LightGBM + mlforecast]
    D --> F[backtest<br>WRMSSE, WAPE, cobertura]
    E --> F
    D --> G[policy<br>SS, ROP, EOQ + simulación]
    E --> G
    C --> G
    F --> H[notebooks y reportes]
    G --> H
    G --> I[dashboard HTML<br>GitHub Pages]
    E --> I
    F --> I
```

## Metodología

**Evaluación.** 6 ventanas móviles de 28 días con reentrenamiento. Las 3 últimas
(2016-02-29 → 2016-05-22) son de **evaluación** y son las únicas que se reportan. Las 3
anteriores son de **calibración**: ahí se decidieron las variables del modelo y se estimó la
incertidumbre del pronóstico, para no tomar ninguna decisión mirando el periodo reportado.

**Pronóstico.** LightGBM global (objetivo Tweedie) con lags 7/14/28, medias móviles, día de
la semana, SNAP, eventos, días hasta el próximo evento y precio relativo. Baselines: Naive,
SeasonalNaive, media móvil de 28 días, AutoETS y AutoARIMA.

**Inventario.** Política (s, S) con revisión diaria y ventas perdidas:

- Safety stock = `z · σ(error del pronóstico) · √L`
- Punto de reorden = demanda esperada en el lead time + safety stock
- Lote = EOQ, acotado entre 1 y 14 días de demanda

Cada política se simula día a día contra la demanda real de las 12 semanas de evaluación.
Como cada una termina con un nivel de servicio distinto, la comparación se hace sobre la
**curva servicio–inventario**: cuánto inventario necesita cada política para alcanzar el
mismo fill rate.

## Cómo reproducirlo

Requisitos: [uv](https://docs.astral.sh/uv/) y `make`.

```bash
uv sync
```

Descarga los datos de M5 (requiere cuenta de Kaggle y aceptar las reglas de la competencia).
Con la CLI autenticada (`kaggle auth login`) lo hace `make data`; si no, deja
`calendar.csv`, `sales_train_evaluation.csv` y `sell_prices.csv` en `data/raw/`.

```bash
make all
```

`make all` corre datos → clasificación → baselines → modelo → ablaciones → evaluación →
política → dashboard. En una máquina de 32 núcleos tarda ~35 minutos, casi todo en
AutoETS y AutoARIMA; con `DFI_SKIP_ARIMA=1` se omite AutoARIMA, que no alimenta la política.

```bash
make notebooks
```

Para probar el pipeline sin M5, con un dataset sintético del mismo formato (es lo que hace CI):

```bash
DFI_DATA_DIR=/tmp/dfi-data make synthetic all
```

La tienda y la categoría se eligen con las variables de entorno `M5_STORE` y `M5_CAT`
(por defecto `CA_1` y `FOODS`).

## Estructura

```
├── src/
│   ├── data/          # descarga, formato largo y M5 sintético
│   ├── features/      # calendario, eventos y precio relativo
│   ├── models/        # baselines, LightGBM, intervalos, backtesting, ablaciones
│   ├── inventory/     # ABC/XYZ, safety stock, ROP, EOQ y simulación
│   ├── dashboard/     # extracción de datos derivados, plantilla HTML y generador
│   └── utils/         # métricas (WRMSSE) y estilo de figuras
├── notebooks/         # 01–04, fuente .py (jupytext) + .ipynb ejecutado
├── reports/           # figuras, resumen ejecutivo y datos del dashboard
├── docs/img/          # capturas del dashboard
├── tests/             # política, simulación, métricas, fuga de datos
└── .github/workflows/ # CI (lint, pruebas, pipeline sintético) y despliegue del dashboard
```

## Limitaciones

- **Ventas, no demanda.** M5 registra lo vendido: un quiebre real aparece como venta cero y
  la simulación lo trata como ausencia de demanda.
- **Costos supuestos.** M5 no trae costos; se asume costo unitario de 70% del precio, 25%
  anual de costo de mantener y USD 5 por pedido. Al variarlos, el ahorro se mantuvo entre
  12.8% y 13.7%.
- **Lead time fijo**, sin mínimos de compra, capacidad de almacén ni caducidad.
- **Una tienda y una categoría.** El WRMSSE usa la jerarquía de este subconjunto y no es
  comparable con el leaderboard de Kaggle.
- **Las políticas se actualizan cada 28 días.** No se evaluó una actualización más frecuente.

## Stack

Python 3.12 · pandas · DuckDB · statsforecast · mlforecast · LightGBM · SciPy · HTML/SVG ·
uv · ruff · pytest · GitHub Actions · GitHub Pages

---

Isaac Martínez · [isaacmartinez.space](https://isaacmartinez.space)
