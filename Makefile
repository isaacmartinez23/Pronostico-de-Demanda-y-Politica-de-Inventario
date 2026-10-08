# Pipeline reproducible. Recetas de una sola línea: corren igual en cmd/sh (Windows) y Linux.
#   make all   -> datos, clasificación, baselines, modelo, evaluación y política
#   make test  -> pruebas unitarias
#   make synthetic -> mini-M5 sintético (usar con DFI_DATA_DIR apuntando a otra carpeta; lo usa CI)

PY := uv run python

# Los scripts imprimen caracteres no ASCII (→, σ); sin esto fallan en Windows al redirigir la salida.
export PYTHONUTF8 := 1

.PHONY: all data classify baselines train evaluate policy notebooks synthetic test lint

all: data classify baselines train evaluate policy

data:
	$(PY) -m src.data.download
	$(PY) -m src.data.make_dataset

classify:
	$(PY) -m src.inventory.abc_xyz

baselines:
	$(PY) -m src.models.baselines

train:
	$(PY) -m src.models.ml_model

evaluate:
	$(PY) -m src.models.backtest

policy:
	$(PY) -m src.inventory.policy

# Ejecuta los notebooks (fuente .py con jupytext) y guarda el .ipynb con resultados y figuras.
notebooks:
	uv run jupytext --to ipynb --execute notebooks/01_eda.py

synthetic:
	$(PY) -m src.data.synthetic

test:
	uv run pytest

lint:
	uv run ruff check .
