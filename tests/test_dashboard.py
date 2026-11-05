import json
import re
from html.parser import HTMLParser

import pytest

from src import config
from src.dashboard import datos as dd
from src.dashboard import generar

DATOS_VERSIONADOS = config.ROOT / "reports" / "dashboard" / "datos.json"
PLANTILLA = generar.PLANTILLA.read_text(encoding="utf-8")


def _minimos(**cambios):
    """Conjunto mínimo válido para probar `validar` y `renderizar` sin correr el pipeline."""
    base = {
        "meta": {"escenarios": [{"lead_time": 7, "nivel": 0.95, "metodo": "sqrt"}], "pronostico_fechas": ["2016-05-23"]},
        "skus": {"id": ["A_1"], "ss": [[3]], "rop": [[9]], "pronostico": [[1.5]]},
        **{k: {"x": [1]} for k in ("valor_ss", "real_vs_pronostico", "error_horizonte", "wrmsse", "segmentos",
                                    "cobertura", "importancia", "ablaciones", "frontera", "necesario",
                                    "escenarios", "costos")},
    }
    base.update(cambios)
    return base


def test_validar_acepta_un_conjunto_completo():
    dd.validar(_minimos())


@pytest.mark.parametrize("columna", ["y", "sell_price", "price", "revenue", "demand"])
def test_validar_rechaza_datos_originales_de_m5(columna):
    with pytest.raises(ValueError, match="originales"):
        dd.validar(_minimos(real_vs_pronostico={"ds": ["2016-01-01"], columna: [3]}))


def test_validar_rechaza_conjuntos_incompletos_o_desalineados():
    datos = _minimos()
    del datos["frontera"]
    with pytest.raises(ValueError, match="faltan"):
        dd.validar(datos)
    with pytest.raises(ValueError, match="largo"):
        dd.validar(_minimos(costos={"a": [1, 2], "b": [1]}))
    with pytest.raises(ValueError, match="escenario"):
        dd.validar(_minimos(skus={"id": ["A_1"], "ss": [[3, 4]], "rop": [[9]], "pronostico": [[1.5]]}))


class _Etiquetas(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self._dentro = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._dentro = dict(attrs).get("id")

    def handle_endtag(self, tag):
        if tag == "script":
            self._dentro = False

    def handle_data(self, data):
        if self._dentro:
            self.scripts.append((self._dentro, data))


def test_renderizar_incrusta_los_datos_sin_romper_el_script():
    datos = _minimos(meta={"nota": "</script><script>alert(1)</script>", "escenarios": [], "pronostico_fechas": []})
    html = generar.renderizar(datos)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "</script><script>alert" not in html

    parser = _Etiquetas()
    parser.feed(html)
    carga = [d for ident, d in parser.scripts if ident == "datos-dashboard"]
    assert len(carga) == 1 and json.loads(carga[0]) == datos


def test_la_plantilla_tiene_las_marcas_una_sola_vez():
    with pytest.raises(ValueError):
        generar.renderizar({}, plantilla="sin marcas")
    assert PLANTILLA.count(generar.MARCA_DATOS) == 1 and PLANTILLA.count(generar.MARCA_CUERPO) == 1


@pytest.mark.skipif(not DATOS_VERSIONADOS.exists(), reason="sin reports/dashboard/datos.json")
def test_los_datos_versionados_son_validos_y_livianos():
    datos = json.loads(DATOS_VERSIONADOS.read_text(encoding="utf-8"))
    dd.validar(datos)
    assert DATOS_VERSIONADOS.stat().st_size < 2_000_000


@pytest.mark.skipif(not DATOS_VERSIONADOS.exists(), reason="sin reports/dashboard/datos.json")
def test_contrato_entre_la_plantilla_y_los_datos():
    """Cada conjunto y columna que lee el JavaScript existe en los datos versionados."""
    datos = json.loads(DATOS_VERSIONADOS.read_text(encoding="utf-8"))
    script = PLANTILLA.split("<script>")[-1]

    conjuntos = set(re.findall(r"\bD\.(\w+)", script))
    assert conjuntos <= set(datos), conjuntos - set(datos)

    columnas_sku = set(re.findall(r"\bS\.(\w+)", script))
    columnas_sku |= {f"dem_lt{lt}" for lt in datos["meta"]["lead_times"]}
    columnas_sku |= {f"q_{lado}{b}" for lado in ("lo", "hi") for b in (80, 95)}
    assert columnas_sku <= set(datos["skus"]), columnas_sku - set(datos["skus"])

    meta = set(re.findall(r"\bM\.(\w+)", script))
    assert meta <= set(datos["meta"]), meta - set(datos["meta"])

    # Columnas de los conjuntos tabulares que la plantilla lee por alias (X.col, Fr.col, ...).
    alias = dict(re.findall(r"const (\w+) = D\.(\w+)", script))
    alias.update(dict(re.findall(r"\b(\w+) = D\.(\w+),", script)))
    for nombre, conjunto in alias.items():
        leidas = set(re.findall(rf"\b{nombre}\.(\w+)\[", script))
        assert leidas <= set(datos[conjunto]), (conjunto, leidas - set(datos[conjunto]))
