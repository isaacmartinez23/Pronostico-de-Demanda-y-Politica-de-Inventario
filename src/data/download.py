"""Verifica (o descarga) los archivos crudos de M5 en data/raw/.

Uso: python -m src.data.download

Si los CSV ya están en data/raw/ no hace nada. Si faltan, intenta bajarlos con la
API de Kaggle (requiere ~/.kaggle/kaggle.json y haber aceptado las reglas de la competencia).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

from src import config

COMPETITION = "m5-forecasting-accuracy"


def missing_files() -> list[str]:
    return [f for f in config.RAW_FILES if not (config.RAW_DIR / f).exists()]


def download() -> None:
    config.ensure_dirs()
    missing = missing_files()
    if not missing:
        print(f"OK: datos de M5 ya presentes en {config.RAW_DIR}")
        return

    print(f"Faltan {missing}. Intentando descargar con la API de Kaggle...")
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION,
             "-p", str(config.RAW_DIR)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "No se pudo descargar M5 automáticamente.\n"
            f"  1) Acepta las reglas en https://www.kaggle.com/competitions/{COMPETITION}\n"
            "  2) Autentica la CLI con `kaggle auth login`, o descarga el zip manualmente\n"
            f"  3) Deja {', '.join(config.RAW_FILES)} en {config.RAW_DIR}"
        ) from exc

    for zpath in config.RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(config.RAW_DIR)
        zpath.unlink()

    if missing_files():
        raise SystemExit(f"Descarga incompleta, siguen faltando: {missing_files()}")
    print(f"OK: datos descargados en {config.RAW_DIR}")


if __name__ == "__main__":
    download()
