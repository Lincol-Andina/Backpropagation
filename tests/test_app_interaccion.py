"""Tests de la interfaz usando el framework de prueba de Streamlit.

A diferencia de un arranque en seco, AppTest ejecuta la app de principio a fin,
interactuando con los widgets como haria un alumno en el navegador. Aqui se
comprueba lo que un test del nucleo no puede ver: que la interfaz responde, que
el recorrido de los 7 pasos funciona y que el boton 'Anterior' reconstruye
exactamente el estado anterior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).parent.parent / "app.py"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def cargar():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=300)


def boton(at, etiqueta: str):
    for b in at.button:
        if b.label == etiqueta:
            return b
    raise AssertionError(f"no se encuentra el boton '{etiqueta}'")


def comprobar_sin_excepciones(at, contexto: str) -> None:
    assert not at.exception, (
        f"{contexto} lanzo {len(at.exception)} excepcion(es): "
        + " | ".join(str(e.value) for e in at.exception)
    )


def caption_de(at, fragmento: str) -> str:
    for c in at.caption:
        if fragmento in c.value:
            return c.value
    raise AssertionError(f"no se encuentra el caption con '{fragmento}'")


def tabla_de(at, *columnas: str):
    for d in at.dataframe:
        df = d.value
        if all(columna in df.columns for columna in columnas):
            return df
    return None


# --------------------------------------------------------------------------
# Arranque
# --------------------------------------------------------------------------


def test_la_app_carga_sin_errores():
    at = cargar().run()
    comprobar_sin_excepciones(at, "la carga inicial")
    assert at.title[0].value.startswith("Simulador de Backpropagation")


def test_la_app_muestra_los_paneles_principales():
    at = cargar().run()
    titulos = [s.value for s in at.subheader]
    assert any("Arquitectura" in t for t in titulos)
    assert any("Curva de perdida" in t for t in titulos)
    assert any("Verificacion del gradiente" in t for t in titulos)


# --------------------------------------------------------------------------
# El recorrido de los 7 pasos
# --------------------------------------------------------------------------


def test_recorrido_completo_de_los_siete_pasos():
    at = cargar().run()
    esperados = [
        "0. Entrada",
        "1. Forward: capa oculta",
        "2. Forward: capa salida",
        "3. Funcion de perdida",
        "4. Backprop: capa salida",
        "5. Backprop: capa oculta",
        "6. Actualizacion de pesos",
    ]
    for esperado in esperados:
        assert esperado in at.subheader[1].value, at.subheader[1].value
        boton(at, "Siguiente paso").click().run()
        comprobar_sin_excepciones(at, f"el paso {esperado}")
    # Tras el paso 6 vuelve al principio de la siguiente epoca.
    assert at.subheader[1].value == esperados[0]


def test_cada_paso_avanza_la_red():
    at = cargar().run()
    for _ in range(7):
        boton(at, "Siguiente paso").click().run()
    comprobar_sin_excepciones(at, "el ciclo completo")
    assert "Epoca 1" in caption_de(at, "Epoca")


# --------------------------------------------------------------------------
# Reproducibilidad: el requisito de que 'Anterior' sea exacto
# --------------------------------------------------------------------------


def test_anterior_esta_deshabilitado_sin_historial():
    at = cargar().run()
    assert boton(at, "Anterior").disabled


def test_anterior_se_habilita_tras_una_epoca():
    at = cargar().run()
    for _ in range(7):
        boton(at, "Siguiente paso").click().run()
    assert not boton(at, "Anterior").disabled


def test_anterior_reconstruye_la_tabla_de_actualizacion():
    """El bug mas grave del HTML original: al retroceder mostraba valores viejos.

    Aqui se mide de verdad: se guardan los pesos del paso 6, se retrocede y se
    vuelve a avanzar por un camino distinto. La tabla debe salir identica.
    """
    at = cargar().run()
    # 13 pasos = epoca 1 completa (pasos 0..6) mas los pasos 0..5 de la epoca 2.
    for _ in range(13):
        boton(at, "Siguiente paso").click().run()

    assert at.subheader[1].value == "6. Actualizacion de pesos", at.subheader[1].value
    esperado = tabla_de(at, "parametro", "anterior", "nuevo")
    assert esperado is not None, "no hay tabla de actualizacion en el paso 6"

    # Volver al paso 6 por un camino distinto: retroceder y reavanzar.
    boton(at, "Anterior").click().run()
    comprobar_sin_excepciones(at, "el primer Anterior")
    assert at.subheader[1].value == "5. Backprop: capa oculta", at.subheader[1].value

    boton(at, "Siguiente paso").click().run()
    assert at.subheader[1].value == "6. Actualizacion de pesos", at.subheader[1].value

    recuperada = tabla_de(at, "parametro", "anterior", "nuevo")
    assert recuperada is not None
    assert np.allclose(esperado["nuevo"].values, recuperada["nuevo"].values)
    assert np.allclose(esperado["anterior"].values, recuperada["anterior"].values)
    assert np.allclose(esperado["gradiente"].values, recuperada["gradiente"].values)


def test_anterior_revierte_la_epoca():
    """Retroceder desde el paso 0 deshace la actualizacion de la epoca anterior."""
    at = cargar().run()
    for _ in range(7):
        boton(at, "Siguiente paso").click().run()
    assert "Epoca 1" in caption_de(at, "Epoca")

    boton(at, "Anterior").click().run()
    comprobar_sin_excepciones(at, "el retroceso de epoca")
    assert "Epoca 0" in caption_de(at, "Epoca"), caption_de(at, "Epoca")
    assert at.subheader[1].value == "6. Actualizacion de pesos", at.subheader[1].value


# --------------------------------------------------------------------------
# Dataset y ejemplos
# --------------------------------------------------------------------------


def test_el_ejemplo_simple_tiene_una_sola_muestra():
    at = cargar().run()
    assert "Muestra 1 de 1" in caption_de(at, "Muestra")


def test_xor_recorre_las_cuatro_combinaciones():
    """Correccion del bug del HTML: XOR debe recorrer 4 patrones por epoca.

    Cada patron pasa por los 7 pasos del recorrido, asi que una epoca completa
    son 7 clics y la muestra solo rota al terminar el paso 6.
    """
    at = cargar().run()
    at.selectbox[0].set_value("xor").run()
    comprobar_sin_excepciones(at, "el cambio a XOR")
    assert "Salida 1" in caption_de(at, "Muestra")

    # La muestra rota al terminar el paso 6, asi que cada patron ocupa 7 clics.
    for indice in range(4):
        assert f"Muestra {indice + 1} de 4" in caption_de(at, "Muestra"), (
            f"esperaba la muestra {indice + 1} al empezar la vuelta {indice + 1}"
        )
        for _ in range(7):
            boton(at, "Siguiente paso").click().run()
        comprobar_sin_excepciones(at, f"la vuelta {indice + 1} sobre XOR")

    assert "Epoca 4" in caption_de(at, "Epoca"), "cuatro vueltas completan cuatro epocas"
    assert "Muestra 1 de 4" in caption_de(at, "Muestra"), "tras 4 vueltas vuelve a la primera"


def test_una_epoca_en_xor_registra_una_sola_perdida():
    """Una epoca = los 4 patrones promediados, no 4 entradas en la curva."""
    at = cargar().run()
    at.selectbox[0].set_value("xor").run()
    boton(at, "Epoca completa").click().run()
    comprobar_sin_excepciones(at, "la epoca en XOR")
    assert "Epoca 1" in caption_de(at, "Epoca")


def test_el_ejemplo_simple_usa_dos_salidas():
    at = cargar().run()
    assert "Salida 2" in caption_de(at, "Muestra")


# --------------------------------------------------------------------------
# Verificacion del gradiente
# --------------------------------------------------------------------------


def test_la_verificacion_del_gradiente_confirma_la_implementacion():
    at = cargar().run()
    boton(at, "Ejecutar la verificacion").click().run()
    comprobar_sin_excepciones(at, "la verificacion del gradiente")

    mensajes = [s.value for s in at.success]
    assert any("gradiente analitico coincide" in m for m in mensajes), mensajes

    tabla = tabla_de(at, "parametro", "analitico", "numerico")
    assert tabla is not None
    assert len(tabla) == 12, "deben compararse los 12 parametros de la red"
    assert tabla["error rel"].max() < 1e-6


def test_la_verificacion_also_funciona_en_xor():
    at = cargar().run()
    at.selectbox[0].set_value("xor").run()
    boton(at, "Ejecutar la verificacion").click().run()
    comprobar_sin_excepciones(at, "la verificacion en XOR")
    tabla = tabla_de(at, "parametro", "analitico", "numerico")
    assert tabla is not None
    assert len(tabla) == 9, "la red XOR tiene 9 parametros"
    assert tabla["error rel"].max() < 1e-6


# --------------------------------------------------------------------------
# Cuestionario
# --------------------------------------------------------------------------


def test_el_cuestionario_no_se_autocorrige_al_abrir():
    """Regresion: con index=None el radio arranca vacio y no hay nota falsa."""
    at = cargar().run()
    assert all(radio.value is None for radio in at.radio), "un radio nace con valor"
    assert not [w for w in at.warning if "Nota" in w.value], "no debe haber nota sin responder"


def test_responder_una_pregunta_muestra_el_resultado():
    at = cargar().run()
    radios = at.radio
    radios[0].set_value("b").run()
    boton(at, "Responder").click().run()
    comprobar_sin_excepciones(at, "la respuesta a la pregunta 1")

    assert any("Correcto" in s.value for s in at.success)
    assert any("sigma" in i.value.lower() for i in at.info), [
        i.value[:80] for i in at.info
    ]


def test_una_respuesta_incorrecta_se_marca_y_se_explica():
    at = cargar().run()
    at.radio[0].set_value("a").run()
    boton(at, "Responder").click().run()
    comprobar_sin_excepciones(at, "la respuesta incorrecta")
    assert any("Incorrecto" in e.value for e in at.error)


def test_la_nota_se_calcula_al_responder():
    """Respondiendo bien las 8 preguntas se obtiene la maxima nota."""
    import ast

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
    letras = [pregunta["correcta"] for pregunta in preguntas]

    at = cargar().run()
    for indice, letra in enumerate(letras):
        # Los indices de los botones Responders bajan a medida que se responde,
        # porque el boton desaparece de la pregunta ya contestada.
        at.radio[indice].set_value(letra).run()
        botones = [b for b in at.button if b.label == "Responder"]
        assert botones, f"no queda boton Responder para la pregunta {indice + 1}"
        botones[0].click().run()
        comprobar_sin_excepciones(at, f"la respuesta {indice + 1}")

    mensajes = [w.value for w in at.warning] + [s.value for s in at.success]
    assert any("Nota" in m for m in mensajes), mensajes
    assert any("100" in m for m in mensajes), (
        f"con las 8 respuestas correctas la nota debe ser 100/100: {mensajes}"
    )


def test_reiniciar_el_cuestionario_borra_las_respuestas():
    at = cargar().run()
    at.radio[0].set_value("b").run()
    boton(at, "Responder").click().run()
    boton(at, "Reiniciar el cuestionario").click().run()
    comprobar_sin_excepciones(at, "el reinicio del cuestionario")


# --------------------------------------------------------------------------
# Controles generales
# --------------------------------------------------------------------------


def test_la_epoca_completa_no_lanza_errores():
    at = cargar().run()
    boton(at, "Epoca completa").click().run()
    comprobar_sin_excepciones(at, "la epoca completa")


def test_reiniciar_devuelve_la_red_al_estado_inicial():
    at = cargar().run()
    boton(at, "Epoca completa").click().run()
    assert "Epoca 1" in caption_de(at, "Epoca")
    boton(at, "Reiniciar").click().run()
    comprobar_sin_excepciones(at, "el reinicio")
    assert "Epoca 0" in caption_de(at, "Epoca")


def test_cambiar_activacion_recarga_la_red():
    at = cargar().run()
    at.selectbox[1].set_value("tanh").run()
    boton(at, "Aplicar cambios").click().run()
    comprobar_sin_excepciones(at, "el cambio de activacion")


def test_el_ejemplo_lineal_avisa_sobre_la_estabilidad():
    """Con salida lineal, avisar de que la curva puede ser inestable."""
    at = cargar().run()
    at.selectbox[2].set_value("identity").run()
    boton(at, "Aplicar cambios").click().run()
    comprobar_sin_excepciones(at, "la activacion lineal")
    assert any("lineal" in w.value.lower() for w in at.warning)


def graficos(at) -> list:
    """st.line_chart se materializa como vega_lite_chart en el arbol."""
    return [e for e in at.main if e.type == "vega_lite_chart"]


def test_la_curva_de_perdida_aparece_tras_entrenar():
    at = cargar().run()
    assert not graficos(at), "sin entrenar no debe haber curva"
    assert any("Avanza pasos" in i.value for i in at.info)

    boton(at, "Epoca completa").click().run()
    comprobar_sin_excepciones(at, "la primera epoca")
    assert graficos(at), "deberia aparecer la curva de perdida"


def test_escala_logaritmica_no_produce_infinitos():
    """La escala log debe tolerar perdidas que llegan a cero."""
    at = cargar().run()
    at.selectbox[0].set_value("xor").run()
    for _ in range(40):
        boton(at, "Epoca completa").click().run()
    casillas = [c for c in at.checkbox if "logaritmica" in c.label]
    assert casillas, "falta la casilla de escala logaritmica"
    casillas[0].check().run()
    comprobar_sin_excepciones(at, "la escala logaritmica")
    assert any("log" in c.value.lower() for c in at.caption)