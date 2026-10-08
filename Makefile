# Pipeline reproducible. Recetas de una sola línea: corren igual en cmd/sh (Windows) y Linux.
#   make all   -> datos, clasificación, baselines, modelo, ablaciones, evaluación, política y datos de la app
#   make notebooks -> ejecuta los notebooks y regenera las figuras
#   make app   -> abre la app de Streamlit
#   make test  -> pruebas unitarias
#   make synthetic -> mini-M5 sintético (usar con DFI_DATA_DIR apuntando a otra carpeta; lo usa CI)

PY := uv run python

# Los scripts imprimen caracteres no ASCII (→, σ); sin esto fallan en Windows al redirigir la salida.
export PYTHONUTF8 := 1

.PHONY: all data classify baselines train ablation evaluate policy app-data app notebooks synthetic test lint

all: data classify baselines train ablation evaluate policy app-data

data:
	$(PY) -m src.data.download
	$(PY) -m src.data.make_dataset

classify:
	$(PY) -m src.inventory.abc_xyz

baselines:
	$(PY) -m src.models.baselines

train:
	$(PY) -m src.models.ml_model

ablation:
	$(PY) -m src.models.ablation

evaluate:
	$(PY) -m src.models.backtest

policy:
	$(PY) -m src.inventory.policy

app-data:
	$(PY) -m src.data.export_app

app:
	uv run streamlit run app/streamlit_app.py

# Ejecuta los notebooks (fuente .py con jupytext) y guarda el .ipynb con resultados y figuras.
notebooks:
	uv run jupytext --to ipynb --execute notebooks/01_eda.py notebooks/02_baselines.py notebooks/03_lightgbm.py notebooks/04_inventory_policy.py

synthetic:
	$(PY) -m src.data.synthetic

test:
	uv run pytest

lint:
	uv run ruff check .
