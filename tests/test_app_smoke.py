"""Test de humo de la interfaz.

Streamlit se ejecuta como script normal, asi que un error de importacion o de
configuracion inicial pasaria desapercibido hasta desplegar. Este test ejecuta
el modulo en un contexto simulado para detectar eso pronto.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).parent.parent / "app.py"


def test_app_existe():
    assert APP.exists(), "no se encuentra app.py"


def test_app_compila():
    """Un error de sintaxis impediria que la app arrancase."""
    compile(APP.read_text(encoding="utf-8"), str(APP), "exec")


def test_app_importa_el_nucleo():
    source = APP.read_text(encoding="utf-8")
    assert "import mlp" in source, "app.py debe usar mlp.py como fuente unica"


def test_app_no_reimplementa_la_matematica():
    """El nucleo no debe duplicarse dentro de la interfaz.

    Busca cuerpos de funcion que contengan la formula de un delta o la regla de
    actualizacion. Si aparece aqui, la matematica esta duplicada y divergira.
    """
    source = APP.read_text(encoding="utf-8")
    arbol = ast.parse(source)

    marcadores = (
        "a2[k] - y", "a2 - y", "outer(", "grad_w1 =", "grad_w2 =",
        "losses = 0.5", "0.5 *", "* (2.0 * h)",
    )

    oficiales = {"numeric_gradient", "gradient_report"}
    ejecutable = Path(__file__).parent.parent / "mlp.py"

    for archivo in (APP,):
        for nodo in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
            if not isinstance(nodo, ast.FunctionDef) or nodo.name in oficiales:
                continue
            cuerpo = ast.get_source_segment(archivo.read_text(encoding="utf-8"), nodo) or ""
            for marcador in marcadores:
                assert marcador not in cuerpo, (
                    f"app.py::{nodo.name} contiene '{marcador}': "
                    "la matematica debe vivir solo en mlp.py"
                )
    assert ejecutable.exists()


def test_app_no_usa_matplotlib_ni_plotly():
    """Se.proto: la app usa los gráficos nativos para arrancar rápido."""
    source = APP.read_text(encoding="utf-8")
    for libreria in ("matplotlib", "plotly", "bokeh"):
        assert f"import {libreria}" not in source, f"{libreria} añade peso innecesario"


def test_app_declara_el_titulo():
    source = APP.read_text(encoding="utf-8")
    assert "set_page_config" in source


def test_app_tiene_ocho_preguntas_de_quiz():
    """El cuestionario debe tener 8 preguntas con una unica respuesta correcta."""
    source = APP.read_text(encoding="utf-8")
    arbol = ast.parse(source)

    asignacion = None
    for node in ast.walk(arbol):
        if isinstance(node, ast.Assign):
            for objetivo in node.targets:
                if isinstance(objetivo, ast.Name) and objetivo.id == "QUIZ":
                    asignacion = node.value
    assert asignacion is not None, "no se encuentra la constante QUIZ"

    preguntas = ast.literal_eval(asignacion)
    assert len(preguntas) == 8, f"se esperaban 8 preguntas, hay {len(preguntas)}"

    for numero, pregunta in enumerate(preguntas, start=1):
        assert {"pregunta", "opciones", "correcta", "explicacion"} <= set(pregunta), numero
        assert pregunta["correcta"] in pregunta["opciones"], numero
        assert len(pregunta["opciones"]) == 3, numero
        assert pregunta["explicacion"].strip(), numero


@pytest.mark.parametrize("termino", ["sigma", "eta", "delta", "z_j", "partial"])
def test_las_preguntas_de_math_usan_notacion_latex(termino):
    """Las preguntas con matematicas deben llevar LaTeX, no texto plano."""
    source = APP.read_text(encoding="utf-8")
    preguntas = ast.literal_eval(
        next(
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            for objetivo in node.targets
            if isinstance(objetivo, ast.Name) and objetivo.id == "QUIZ"
        )
    )
    con_math = [p for p in preguntas if any(simbolo in p["pregunta"] for simbolo in ("\\sigma", "\\eta", "\\delta"))]
    assert con_math, "deberia haber preguntas con notacion matematica"
    for pregunta in con_math:
        assert "$" in pregunta["pregunta"], pregunta["pregunta"]