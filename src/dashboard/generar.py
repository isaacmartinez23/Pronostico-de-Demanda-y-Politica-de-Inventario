"""Genera el dashboard HTML a partir de reports/dashboard/datos.json.

Uso: python -m src.dashboard.generar   (escribe reports/dashboard/index.html)

El resultado es un único archivo sin dependencias de servidor: la plantilla (``plantilla.html``)
con los datos incrustados como JSON. Se abre con doble clic y se publica en GitHub Pages. Como el
JSON se versiona, CI puede generar y desplegar la página sin los datos de M5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src import config

PLANTILLA = Path(__file__).with_name("plantilla.html")
MARCA_DATOS = "__DATOS_DASHBOARD__"
MARCA_CUERPO = "<!-- cuerpo -->"


def renderizar(datos: dict[str, Any], plantilla: str | None = None) -> str:
    """Inserta los datos en la plantilla y devuelve el documento completo."""
    plantilla = plantilla if plantilla is not None else PLANTILLA.read_text(encoding="utf-8")
    for marca in (MARCA_DATOS, MARCA_CUERPO):
        if plantilla.count(marca) != 1:
            raise ValueError(f"[dashboard] la plantilla debe contener {marca!r} exactamente una vez")
    # `<` escapado: ningún texto de los datos puede cerrar la etiqueta <script> que los contiene.
    carga = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    cabeza, cuerpo = plantilla.replace(MARCA_DATOS, carga).split(MARCA_CUERPO)
    return (
        '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{cabeza}</head>\n<body>\n{cuerpo}</body>\n</html>\n"
    )


def run(origen: Path | None = None, destino: Path | None = None) -> Path:
    origen = origen or config.DASHBOARD_DATA
    destino = destino or config.DASHBOARD_HTML
    datos = json.loads(origen.read_text(encoding="utf-8"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(renderizar(datos), encoding="utf-8")
    print(f"[dashboard] {destino} ({destino.stat().st_size / 1e6:.2f} MB)")
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, help="JSON de entrada (predeterminado: reports/dashboard/datos.json)")
    parser.add_argument("--destino", type=Path, help="HTML de salida (predeterminado: reports/dashboard/index.html)")
    args = parser.parse_args(argv)
    run(args.datos, args.destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
