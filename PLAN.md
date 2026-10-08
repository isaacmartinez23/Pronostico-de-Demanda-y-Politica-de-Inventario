# Plan de desarrollo — Pronóstico de Demanda y Política de Inventario

Plan anclado al estado real del repo al 2026-10-07. El brief original describe el proyecto
desde cero; aquí la lógica de `src/` ya está escrita, así que el trabajo que falta es
**ejecutar, verificar, endurecer la metodología y empaquetar**.

## 0. Estado al cierre de la primera iteración

Hecho todo lo que no requiere publicar nada fuera de esta máquina. Resultados en el periodo
de evaluación (3 ventanas de 28 días, 2016-02-29 → 2016-05-22):

| Criterio | Resultado |
| --- | --- |
| WRMSSE de LightGBM < SeasonalNaive en las 3 ventanas | Cumplido: 0.528 vs 0.785 (−33%); vs AutoETS 0.608 (−13%) |
| Fill rate ≥ 95% en clase A (L=7, objetivo 95%) | Cumplido: 96.8%; 87% de los SKUs A lo cumple |
| Inventario a igual fill rate (95%) vs. política clásica | −14% total, −11% clase A |
| Cobertura de intervalos a ±5 puntos de la nominal | **No cumplido:** 73.6% y 88.7% (≈6 puntos por debajo) |
| `make all` regenera todo; CI con datos sintéticos | Cumplido localmente; CI aún sin correr en GitHub |

Pendiente (requiere tu confirmación porque publica contenido): crear el repo en GitHub,
desplegar la app en Streamlit Community Cloud, grabar el GIF de la demo, tarjeta en el sitio
y post en LinkedIn.

Decisiones abiertas:
- `item_id` quedó fuera del modelo por la ablación en calibración; en evaluación el efecto
  fue neutro en WRMSSE y redujo el ahorro de inventario de −16% a −14%. Ver notebook 03.
- Método de safety stock por defecto de la tabla de recomendación: `sqrt` (fórmula de libro).
  `acumulado` calibra mejor el servicio de ciclo con la misma eficiencia.
- Los intervalos quedan cortos; mejorarlos (más ventanas de calibración, márgenes por
  horizonte) es el siguiente paso técnico natural.

## 1. Dónde estábamos al empezar (2026-10-07)

Subconjunto ya procesado: **FOODS en CA_1 — 1,429 series**, 2011-01-29 → 2016-05-22,
sin huecos de fecha, 46.8% de días en cero. Matriz ABC/XYZ: A=603, B=458, C=368 ·
X=401, Y=715, Z=313. Patrón de demanda: **925 intermitentes, 254 irregulares**, 213 suaves,
37 erráticas (el 82% de los SKUs no tiene demanda "suave"; esto condiciona la Fase 4).

| Pieza | Estado |
| --- | --- |
| Entorno (`pyproject.toml`, `uv.lock`, `.venv` 3.12) | Listo |
| `data/raw/` (CSV de M5) | Presentes (descarga manual; no hay credencial de Kaggle configurada) |
| `src/data/make_dataset.py`, `src/inventory/abc_xyz.py` | Escritos **y ejecutados** (hay parquet) |
| `src/models/baselines.py`, `ml_model.py`, `intervals.py` | Escritos, **sin ejecutar** (no hay `cv_*.parquet`) |
| `src/inventory/policy.py` (fórmulas + simulación + frontera) | Escrito, **sin ejecutar** |
| `src/utils/metrics.py` (WRMSSE jerárquico, MAE, WAPE, sesgo) | Escrito, nadie lo llama todavía |
| `src/models/backtest.py` (tabla de métricas por segmento) | **No existe** |
| `tests/`, `notebooks/`, `app/`, `reports/`, `.github/workflows/` | Carpetas vacías |
| `Makefile`, `README.md`, `.pre-commit-config.yaml`, repo git | No existen |

## 2. Decisiones que conviene fijar antes de seguir

1. **Protocolo de evaluación congelado.** 6 ventanas de 28 días: 3 de calibración + 3 de
   evaluación (2016-02-29 → 2016-05-22). Cualquier ajuste de hiperparámetros, features o
   σ se decide solo con las de calibración; las de evaluación se reportan, no se optimizan.
2. **La vara real no es SeasonalNaive.** En demanda intermitente, `MediaMovil28` y AutoETS
   suelen ser más difíciles de vencer. El criterio "supera a SeasonalNaive" se mantiene,
   pero el reporte compara contra el **mejor baseline por segmento**.
3. **Nivel de servicio: dos definiciones distintas.** `z` fija el nivel de servicio de
   *ciclo* (probabilidad de no quebrar); la simulación mide *fill rate* (% de demanda
   atendida). No son intercambiables. Se reportan ambos y el titular se calcula
   **a igual fill rate** sobre la curva servicio–inventario (`inventory_at_fill_rate` ya lo
   hace), nunca comparando políticas que alcanzan servicios distintos.
4. **σ·√L subestima el riesgo.** Los errores diarios de un pronóstico recursivo están
   correlacionados, así que `σ_diario·√L` ([policy.py:47](src/inventory/policy.py:47))
   queda corto. Se conserva como variante "de libro" y se agrega la variante que mide
   directamente **σ del error acumulado en L días** con los residuos del backtest.
5. **La normal no describe demanda intermitente.** Tercera variante: ROP = pronóstico del
   lead time + **cuantil empírico (conformal) del error acumulado**. Es el puente natural
   entre los intervalos de la Fase 3 y la decisión de la Fase 4, y un diferenciador del
   portafolio.
6. **Ventas ≠ demanda.** M5 registra ventas; un quiebre real aparece como cero. Se
   documenta como limitación (la simulación trata la venta observada como demanda).
7. **Costos del EOQ son supuestos** (`config.py`: costo 70% del precio, holding 25% anual,
   USD 5 por pedido, tope 14 días). Se declaran y se les hace sensibilidad, no se esconden.
8. **Fuera de alcance:** Favorita y polars. La validación de robustez se hace re-corriendo
   el pipeline en otro subconjunto con `M5_STORE` / `M5_CAT` (ya soportado en config).

## 3. Fases

### Fase 0 — Cimientos (½ día)

- [x] `git init` y commits locales
- [ ] Repo público `demand-forecasting-inventory` en GitHub (pendiente de tu visto bueno)
- [x] `Makefile` con `data | classify | baselines | train | policy | report | app | test | all`.
      Recetas de una línea (`uv run python -m src...`) para que funcionen igual en
      `cmd`/`sh` de Windows y en Linux
- [x] `.pre-commit-config.yaml` (ruff lint + format, fin de línea, tamaño de archivos)
- [x] Rutas de datos sobreescribibles por variable de entorno (p. ej. `DFI_DATA_DIR`) en
      `src/config.py` — lo necesita el smoke test de CI

**Salida:** `make data classify` reproduce los tres parquet actuales desde `data/raw/`.

### Fase 1 — Datos y EDA (1 día)

- [x] Subconjunto, formato largo, calendario + precios, ABC/XYZ
- [x] `notebooks/01_eda` (jupytext `.py` pareado, como ya prevé `pyproject.toml`):
      estacionalidad semanal/anual, efecto SNAP y eventos (incluido el efecto *previo* al
      feriado), ceros por departamento, mapa ADI–CV², curva de Pareto
- [x] Guardar 4–5 figuras en `reports/figures/` (se reutilizan en README y resumen)

**Salida:** cada feature de `build_features.py` queda justificada por una figura del EDA.

### Fase 2 — Baselines y evaluación (1½ días)

- [x] Ejecutar `src.models.baselines` y **medir el tiempo** de AutoARIMA (1,429 series × 3
      ventanas). Si pasa de ~30 min: bajar `STATS_INPUT_SIZE` a 365 o limitar AutoARIMA a
      clases A+B, y decirlo en el README
- [x] Crear `src/models/backtest.py`: une `cv_baselines` + `cv_lgbm`, calcula WRMSSE por
      ventana y nivel, y MAE / WAPE / sesgo / RMSSE **por segmento ABC-XYZ y por patrón**;
      escribe `metrics_*.parquet`
- [x] `notebooks/02_baselines`: tabla comparativa y mejor baseline por segmento

**Salida:** una tabla modelo × segmento reproducible con `make baselines`.

### Fase 3 — Modelo de ML (2 días)

- [x] Ejecutar `src.models.ml_model` (6 ventanas con reentrenamiento + pronóstico final)
- [x] Comparar contra baselines: dónde gana y dónde pierde, sesgo por segmento, error por
      paso del horizonte (¿se degrada la recursión hacia el día 28?)
- [x] Ablaciones acotadas, decididas en ventanas de calibración: sin `item_id` categórico
      (1,429 niveles, riesgo de sobreajuste), sin features de precio, Tweedie vs Poisson
- [x] **Intervalos fuera de muestra:** hoy los cuantiles se calibran con las 6 ventanas
      ([ml_model.py:103](src/models/ml_model.py:103)) y `intervals.coverage` no se usa.
      Calibrar con las 3 primeras, medir cobertura 80/95 en las 3 de evaluación y
      reportarla por segmento; para el pronóstico final sí se usan las 6
- [x] `notebooks/03_lightgbm`: importancia de variables, ejemplos buenos y malos

**Salida:** WRMSSE de LightGBM < SeasonalNaive en las 3 ventanas; cobertura empírica a
±5 puntos de la nominal. Si el modelo no gana en algún segmento, se reporta tal cual.

### Fase 4 — De pronóstico a decisión (2 días)

- [x] Ejecutar `src.inventory.policy` tal como está: primeros números de la simulación
- [x] Agregar las variantes de safety stock de las decisiones 4 y 5 (σ del error acumulado
      en L días; cuantil empírico) y compararlas en la frontera servicio–inventario
- [x] Reportar fill rate **y** nivel de servicio de ciclo; resultados por clase ABC
- [x] Escenarios L ∈ {3, 7, 14} × servicio ∈ {90, 95, 99%} (grid ya implementado)
- [x] Sensibilidad a los supuestos de costo del EOQ
- [x] `notebooks/04_inventory_policy`: frontera, escenarios, SKUs que no cumplen y por qué

**Salida:** la frase titular con su número real — inventario promedio que necesita cada
política para alcanzar 95% de fill rate en clase A — y fill rate ≥ 95% en clase A.

### Fase 5 — Calidad y producto de portafolio (3 días)

**Tests y CI** (se pueden adelantar en paralelo desde la Fase 2)

- [x] Generador de un mini-M5 sintético en formato crudo (≈30 SKUs × 500 días): CI no puede
      bajar M5 (licencia y credenciales)
- [x] `test_policy.py`: z(0.95)=1.645, SS monótono en L y servicio, EOQ contra valor
      conocido, y propiedades de la simulación (ventas + perdidas = demanda, inventario
      nunca negativo, el pedido llega exactamente a los L días)
- [x] `test_features.py`: **sin fuga** (alterar `y` futuro no cambia features pasadas),
      `days_to_event`, precio relativo
- [x] `test_metrics.py`, `test_abc_xyz.py`, `test_intervals.py` (pronóstico perfecto → 0,
      cortes 80/95, cobertura en datos sintéticos)
- [x] `.github/workflows/ci.yml`: uv + ruff + pytest + `make all` sobre el mini-M5

**App Streamlit**

- [x] `make app` exporta a `app/data/` solo lo necesario (historia reciente, pronóstico con
      intervalos, tabla de política, escenarios, frontera, métricas); objetivo < 25 MB
- [x] `app/streamlit_app.py`: selector de SKU con filtro ABC/XYZ, pronóstico de 28 días con
      intervalo, tarjetas SS / ROP / cantidad a pedir, sliders de lead time, nivel de
      servicio e inventario actual (recalcula en vivo con las fórmulas de `policy.py`), y
      una pestaña de resultados globales
- [x] `app/requirements.txt` mínimo (streamlit, pandas, pyarrow, plotly, scipy): la app
      **no entrena nada**, así Streamlit Cloud no instala statsforecast ni lightgbm
- [ ] Desplegar en Streamlit Community Cloud con Python 3.12

**Documentación y difusión**

- [x] README: problema, resultados clave, diagrama de arquitectura, capturas de la demo, cómo
      reproducir (incluida la descarga de M5 y `kaggle auth login`), limitaciones
- [x] `reports/executive_summary.md` de 1 página, escrito para un gerente de supply chain
- [ ] Tarjeta en isaacmartinez.site (Análisis de Datos + Inteligencia Artificial) y post
      en LinkedIn

## 4. Orden recomendado

1. Fase 0, y enseguida **correr todo el pipeline tal cual** (baselines → LightGBM →
   política). Es medio día y despeja la incógnita que define el relato: si el modelo gana
   y por cuánto.
2. `backtest.py` + tests + CI, antes de tocar la metodología (para refinar sobre red).
3. Refinamientos de Fase 3 y 4 (intervalos fuera de muestra, variantes de safety stock).
4. Notebooks y figuras, ya con números definitivos.
5. App, README, resumen ejecutivo, publicación.

Esfuerzo total estimado: **≈ 10 días efectivos** (3–4 semanas a medio tiempo).

## 5. Riesgos

| Riesgo | Mitigación |
| --- | --- |
| LightGBM no supera a los baselines en SKUs intermitentes | Reportar por segmento; el relato puede ser "ML para AX/AY, regla simple para CZ", que es una conclusión de negocio válida |
| La política con modelo no reduce inventario a igual servicio | El titular sale de la frontera, no de un punto; un resultado nulo honesto vale más que uno inflado |
| AutoARIMA demasiado lento | Historia acotada o solo clases A+B |
| `make all` depende de credenciales de Kaggle | Documentar la descarga manual; evaluar el espejo de M5 de Nixtla (`datasetsforecast`) como alternativa sin credenciales |
| Sobreajuste al periodo de evaluación | Decisión 1: ventanas de evaluación congeladas |
| Límite de recursos en Streamlit Cloud | App de solo lectura sobre parquet precalculado |

## 6. Definición de terminado

- WRMSSE de LightGBM menor que SeasonalNaive en las 3 ventanas de evaluación, con la
  comparación contra el mejor baseline por segmento a la vista
- Simulación (L=7, objetivo 95%): fill rate ≥ 95% en clase A y % de SKUs A que lo cumplen
- Cobertura de los intervalos 80/95 medida fuera de muestra y reportada
- `make all` desde un clon limpio (con los CSV en `data/raw/`) regenera todos los artefactos
- CI en verde: ruff + pytest + pipeline completo sobre datos sintéticos
- App pública, README con demo, resumen ejecutivo y tarjeta en el sitio
