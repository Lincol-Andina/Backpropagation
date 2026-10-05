"""Tests del nucleo matematico.

La red de seguridad del proyecto: si la backpropagation esta mal, estos tests
lo detectan. Ejecutar con `pytest -q` desde la raiz del repositorio.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlp import (
    DATASET_SIMPLE,
    DATASET_XOR,
    MLP,
    Pattern,
    Weights,
    activation_target_warnings,
    dataset_for,
    default_weights,
    gradient_report,
    numeric_gradient,
    num_outputs_for,
)

ACTIVATIONS = ["sigmoid", "tanh", "relu", "identity"]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def finite_difference_check(
    mlp: MLP, x: np.ndarray, y: np.ndarray, tol: float = 1e-6
) -> float:
    """Comprueba analitico vs numerico y devuelve el mayor error relativo."""
    trace = mlp.forward(x, y)
    w = mlp.weights
    n_in, n_hidden = w.w1.shape
    n_out = w.w2.shape[1]

    analytic = np.concatenate(
        [trace.grad_w1.ravel(), trace.grad_b1.ravel(), trace.grad_w2.ravel(), trace.grad_b2.ravel()]
    )
    numeric = numeric_gradient(mlp, x, y, weights=w, h=1e-5)

    worst = 0.0
    for a, n in zip(analytic, numeric):
        worst = max(worst, abs(a - n) / max(abs(a), abs(n), 1e-12))
    assert worst < tol, f"error relativo maximo {worst:.3e} supera {tol:.1e}"
    return worst


# --------------------------------------------------------------------------
# El punto critico: el gradiente analitico debe coincidir con el numerico
# --------------------------------------------------------------------------


def test_gradiente_analitico_coincide_con_diferencias_finitas_2_2_2():
    mlp = MLP.from_preset("simple")
    finite_difference_check(mlp, np.array([0.05, 0.10]), np.array([0.01, 0.99]))


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_gradiente_correcto_en_todas_las_activaciones_ocultas(activation):
    mlp = MLP.from_preset("simple", hidden_activation=activation, output_activation="identity")
    finite_difference_check(mlp, np.array([0.35, 0.80]), np.array([0.20, 0.60]))


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_gradiente_correcto_en_todas_las_activaciones_de_salida(activation):
    mlp = MLP.from_preset("simple", hidden_activation="tanh", output_activation=activation)
    finite_difference_check(mlp, np.array([-0.40, 0.15]), np.array([0.30, 0.70]))


def test_gradiente_correcto_en_xor():
    mlp = MLP.from_preset("xor")
    finite_difference_check(mlp, np.array([0.0, 1.0]), np.array([1.0]))


def test_gradiente_es_casi_cero_en_el_minimo():
    """En un minimo de la perdida el gradiente debe desvanecerse."""
    mlp = MLP.from_preset("xor", learning_rate=0.5)
    for _ in range(4000):
        mlp.train_epoch(DATASET_XOR)
    trace = mlp.forward(np.array([0.0, 1.0]), np.array([1.0]))
    worst = max(
        float(np.abs(trace.grad_w1).max()),
        float(np.abs(trace.grad_b1).max()),
        float(np.abs(trace.grad_w2).max()),
        float(np.abs(trace.grad_b2).max()),
    )
    assert worst < 1e-2, f"gradiente en el minimo demasiado grande: {worst:.3e}"


def test_informe_de_gradiente_devuelve_una_fila_por_parametro():
    mlp = MLP.from_preset("simple")
    rows = gradient_report(mlp, np.array([0.05, 0.10]), np.array([0.01, 0.99]))
    # 4 pesos W1 + 2 sesgos b1 + 4 pesos W2 + 2 sesgos b2 = 12
    assert len(rows) == 12
    assert all(row["error rel"] < 1e-6 for row in rows)
    assert {"parametro", "analitico", "numerico", "error abs", "error rel"} <= set(rows[0])


# --------------------------------------------------------------------------
# Entrenamiento: el objetivo pedagogico es que el error baje y XOR se aprenda
# --------------------------------------------------------------------------


def test_xor_se_aprende():
    """Las cuatro combinaciones deben aprenderse hasta error despreciable."""
    mlp = MLP.from_preset("xor", learning_rate=0.5)
    for _ in range(5000):
        loss, diverged = mlp.train_epoch(DATASET_XOR)
        assert not diverged, "el entrenamiento de XOR diverge"
    assert loss < 1e-3, f"XOR no converge: perdida final {loss:.6f}"

    for pattern in DATASET_XOR:
        prediction = mlp.forward(np.array(pattern.x), np.array(pattern.y)).a2
        assert abs(float(prediction[0]) - pattern.y[0]) < 0.05


def test_xor_es_imposible_sin_capa_oculta():
    """La red debe clasificar XOR de verdad, no por casualidad."""
    mlp = MLP.from_preset("xor", learning_rate=0.5)
    for _ in range(5000):
        mlp.train_epoch(DATASET_XOR)

    outputs = {}
    for pattern in DATASET_XOR:
        outputs[pattern.label] = float(mlp.forward(np.array(pattern.x), np.array(pattern.y)).a2[0])

    # AND (0,0)->0 y (1,1)->0 deben dar lo mismo; (0,1)->1 y (1,0)->1 tambien.
    assert abs(outputs["(0, 0) -> (0)"] - outputs["(1, 1) -> (0)"]) < 0.05
    assert abs(outputs["(0, 1) -> (1)"] - outputs["(1, 0) -> (1)"]) < 0.05
    assert outputs["(0, 1) -> (1)"] > outputs["(0, 0) -> (0)"]


def test_epoca_reduce_la_perdida_en_el_ejemplo_simple():
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    initial = mlp.forward(np.array([0.05, 0.10]), np.array([0.01, 0.99])).loss_total
    for _ in range(200):
        loss, _ = mlp.train_epoch(DATASET_SIMPLE)
    assert loss < initial * 0.01


def test_la_perdida_registrada_es_la_antes_de_actualizar():
    """La epoca promedia la perdida previa a cada actualizacion."""
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    loss, _ = mlp.train_epoch(DATASET_SIMPLE)
    mlp.weights = default_weights("simple")
    before = mlp.forward(np.array([0.05, 0.10]), np.array([0.01, 0.99])).loss_total
    assert loss == pytest.approx(before)


def test_actualizar_reduce_el_error_en_la_muestra():
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    x, y = np.array([0.05, 0.10]), np.array([0.01, 0.99])
    before = mlp.forward(x, y).loss_total
    mlp.train_epoch(DATASET_SIMPLE)
    after = mlp.forward(x, y).loss_total
    assert after < before


def test_valor_esperado_del_ejemplo_clasico():
    """Regression test: la salida o2 debe acercarse a 0.99 tras entrenar.

    Con este dataset de una sola muestra la convergencia es lenta: hacen falta
    del orden de 5000 epocas para llegar a ~0.986 con eta=0.5.
    """
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    for _ in range(5000):
        mlp.train_epoch(DATASET_SIMPLE)
    output = mlp.forward(np.array([0.05, 0.10]), np.array([0.01, 0.99])).a2
    assert float(output[1]) == pytest.approx(0.99, abs=0.01)
    # La salida que ya era correcta no debe empeorarse.
    assert float(output[0]) < 0.05


# --------------------------------------------------------------------------
# Divergencia: una tasa de aprendizaje absurda debe detectarse
# --------------------------------------------------------------------------


def test_tasa_de_aprendizaje_alta_divergE():
    """Con salida lineal, eta=10 hace explotar la perdida y hay que avisarlo.

    Nota sobre por que la salida importa: con sigmoide en la salida, z se recorta
    y la perdida esta acotada, asi que una eta enorme produce oscilacion pero no
    divergencia numerica. La divergencia explosionista necesita salida no acotada.
    """
    mlp = MLP.from_preset("simple", output_activation="identity", learning_rate=10.0)
    diverged = False
    for _ in range(500):
        _, diverged = mlp.train_epoch(DATASET_SIMPLE)
        if diverged:
            break
    assert diverged, "con eta=10 y salida lineal la perdida deberia divergir y avisar"


def test_tasa_alta_con_sigmoide_no_diverge_pero_inestabiliza():
    """Con sigmoide la perdida no puede explotar, pero eta=10 la hace oscilar.

    Documenta por que la deteccion de divergencia no se dispara aqui.
    """
    mlp = MLP.from_preset("simple", output_activation="sigmoid", learning_rate=10.0)
    losses = []
    for _ in range(60):
        loss, diverged = mlp.train_epoch(DATASET_SIMPLE)
        assert not diverged
        losses.append(loss)

    assert all(np.isfinite(value) for value in losses)
    assert max(losses[-10:]) > min(losses[-10:]), "eta=10 deberia producir oscilacion"


def test_deteccion_de_divergencia_acepta_perdidas_sanas():
    assert not MLP.is_diverged(0.5)
    assert not MLP.is_diverged(1e-9)
    assert MLP.is_diverged(float("nan"))
    assert MLP.is_diverged(float("inf"))
    assert MLP.is_diverged(1e7)


def test_la_perdida_a_lo_largo_del_entrenamiento_no_debe_crecer_de_repente():
    """Con eta razonable la perdida no debe dar saltos absurdos al principio."""
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    history = []
    for _ in range(50):
        loss, _ = mlp.train_epoch(DATASET_SIMPLE)
        history.append(loss)
    assert history[-1] < history[0]
    assert all(np.isfinite(value) for value in history)


# --------------------------------------------------------------------------
# Compatibilidad activacion / objetivo
# --------------------------------------------------------------------------


def test_relu_salida_aprende_el_ejemplo_simple():
    """ReLU alcanza un objetivo de 0.01 porque 0.01 >= 0. No es un problema."""
    mlp = MLP.from_preset("simple", output_activation="relu", learning_rate=0.5)
    for _ in range(2000):
        mlp.train_epoch(DATASET_SIMPLE)
    loss, _ = mlp.train_epoch(DATASET_SIMPLE)
    assert loss < 1e-3, f"ReLU deberia aprender el ejemplo simple, perdida {loss:.3e}"


def test_relu_no_puede_producir_objetivos_negativos():
    """La limitacion real de ReLU: nunca da salida negativa."""
    mlp = MLP.from_preset("simple", output_activation="relu", learning_rate=0.5)
    objetivo_negativo = (Pattern(x=(0.05, 0.10), y=(-0.5, 0.5)),)
    for _ in range(2000):
        mlp.train_epoch(objetivo_negativo)
    loss, _ = mlp.train_epoch(objetivo_negativo)

    # El minimo teorico alcanzable es 0.5 * (0.5 - 0)^2 = 0.125
    assert loss >= 0.125 - 1e-6, f"ReLU no deberia poder bajar de 0.125, obtuvo {loss:.4f}"
    assert np.all(mlp.forward(np.array([0.05, 0.10]), np.array([-0.5, 0.5])).a2 >= 0.0)


def test_aviso_relu_con_objetivos_negativos():
    negativos = (Pattern(x=(0.05, 0.10), y=(-0.5, 0.5)),)
    warnings = activation_target_warnings("relu", negativos)
    assert any("negativos" in text for text in warnings), warnings


def test_aviso_relu_advertencia_de_diseno_no_de_error():
    """Con objetivos no negativos, ReLU da un aviso de diseño pero no de imposibilidad."""
    warnings = activation_target_warnings("relu", DATASET_SIMPLE)
    assert all("inalcanzables" not in text for text in warnings), warnings


def test_sin_avisos_con_sigmoide_y_el_ejemplo_simple():
    assert activation_target_warnings("sigmoid", DATASET_SIMPLE) == []
    assert activation_target_warnings("sigmoid", DATASET_XOR) == []


def test_aviso_para_salida_lineal():
    assert activation_target_warnings("identity", DATASET_SIMPLE)


def test_aviso_tanh_con_objetivos_fuera_de_rango():
    fuera = (Pattern(x=(0.05, 0.10), y=(3.0, 0.5)),)
    warnings = activation_target_warnings("tanh", fuera)
    assert any("inalcanzables" in text for text in warnings), warnings


def test_xor_con_relu_si_es_aprendible():
    """XOR usa objetivos 0 y 1, ambos alcanzables con ReLU."""
    mlp = MLP.from_preset("xor", learning_rate=0.2, output_activation="relu")
    for _ in range(6000):
        loss, diverged = mlp.train_epoch(DATASET_XOR)
        assert not diverged
    assert loss < 1e-2, f"XOR con ReLU no converge: {loss:.6f}"


# --------------------------------------------------------------------------
# Guardas numericas
# --------------------------------------------------------------------------


def test_sigmoide_no_desborda_con_entradas_enormes():
    mlp = MLP.from_preset("simple", hidden_activation="sigmoid", output_activation="sigmoid")
    weights = Weights(
        w1=np.array([[1000.0, 1000.0], [1000.0, 1000.0]]),
        b1=np.array([1000.0, 1000.0]),
        w2=np.array([[1000.0, 1000.0], [1000.0, 1000.0]]),
        b2=np.array([1000.0, 1000.0]),
    )
    trace = mlp.forward(np.array([1.0, 1.0]), np.array([0.0, 1.0]), weights=weights)
    assert np.all(np.isfinite(trace.a1)), "la activacion oculta desbordo"
    assert np.all(np.isfinite(trace.a2)), "la activacion de salida desbordo"
    assert np.isfinite(trace.loss_total)


def test_sigmoide_es_exacta_en_el_rango_razonable():
    mlp = MLP.from_preset("simple")
    trace = mlp.forward(np.array([0.05, 0.10]), np.array([0.01, 0.99]))
    assert np.allclose(trace.a1, 1.0 / (1.0 + np.exp(-trace.z1)))
    assert np.allclose(trace.a2, 1.0 / (1.0 + np.exp(-trace.z2)))


def test_relu_deriva_cero_para_z_negativo():
    mlp = MLP.from_preset("simple", hidden_activation="relu", output_activation="relu")
    trace = mlp.forward(np.array([-5.0, -5.0]), np.array([0.0, 0.0]))
    assert np.allclose(trace.sigma1_prime, 0.0)


# --------------------------------------------------------------------------
# Instantaneas: el requisito de que el stepper sea reproducible
# --------------------------------------------------------------------------


def test_instantanea_restaura_el_estado_exacto():
    mlp = MLP.from_preset("simple")
    before_weights = mlp.weights.flat().copy()
    before_trace = mlp.trace.clone() if mlp.trace else None

    mlp.train_epoch(DATASET_SIMPLE)
    assert not np.allclose(mlp.weights.flat(), before_weights), "el entrenamiento no cambio nada"

    mlp.rebuild(Weights.from_flat(before_weights, mlp.n_in, mlp.n_hidden, mlp.n_out))
    assert np.allclose(mlp.weights.flat(), before_weights)


def test_snapshot_copia_los_pesores_y_la_traza():
    mlp = MLP.from_preset("simple")
    mlp.train_epoch(DATASET_SIMPLE)
    snapshot = mlp.snapshot()

    original = mlp.weights.flat().copy()
    mlp.train_epoch(DATASET_SIMPLE)

    assert np.allclose(snapshot.weights.flat(), original), "el snapshot no quedó congelado"
    assert not np.allclose(mlp.weights.flat(), original)


def test_snapshot_preserva_los_ajustes():
    mlp = MLP.from_preset("xor", learning_rate=0.25, hidden_activation="tanh")
    snapshot = mlp.snapshot()
    assert snapshot.learning_rate == 0.25
    assert snapshot.hidden_activation == "tanh"
    assert snapshot.output_activation == "tanh"


def test_weights_va_y_vuelta_por_vector_plano():
    weights = default_weights("simple")
    vector = weights.flat()
    assert vector.shape == (12,)
    restored = Weights.from_flat(vector, 2, 2, 2)
    assert np.allclose(restored.w1, weights.w1)
    assert np.allclose(restored.b1, weights.b1)
    assert np.allclose(restored.w2, weights.w2)
    assert np.allclose(restored.b2, weights.b2)


def test_rejilla_de_pesos_conserva_el_orden():
    """El orden de flat() debe coincidir con el de la rejilla que ve el alumno."""
    weights = default_weights("simple")
    rows = weights.as_dict()["params"]
    valor = [row["valor"] for row in rows]
    esperado = list(weights.w1.ravel()) + list(weights.b1) + list(weights.w2.ravel()) + list(weights.b2)
    assert np.allclose(valor, esperado)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def test_xor_tiene_las_cuatro_combinaciones():
    assert len(DATASET_XOR) == 4
    assert len(dataset_for("xor")) == 4
    assert num_outputs_for("xor") == 1


def test_xor_es_la_funcion_logica_excluyente():
    """Cada patron debe cumplir y = x1 XOR x2."""
    for pattern in DATASET_XOR:
        x1, x2 = pattern.x
        esperado = 1.0 if x1 != x2 else 0.0
        assert pattern.y[0] == esperado


def test_dataset_simple_conserva_el_ejemplo_de_la_teoria():
    assert DATASET_SIMPLE[0].x == (0.05, 0.10)
    assert DATASET_SIMPLE[0].y == (0.01, 0.99)
    assert num_outputs_for("simple") == 2


def test_una_epoca_actualiza_una_vez_por_patron():
    """Una epoca debe pasar por los 4 patrones de XOR, no solo por uno.

    Con un dataset de un solo patron, una epoca y un patron son indistinguibles.
    Aqui se comprueba que la longitud del dataset marca el numero de
    actualizaciones, y que la perdida que devuelve es la media de todas.
    """
    mlp = MLP.from_preset("xor", learning_rate=0.1)
    vecinos = []
    for _ in range(50):
        # Guardamos la perdida de cada patron tras la epoca completa.
        for pattern in DATASET_XOR:
            vecinos.append(mlp.forward(np.array(pattern.x), np.array(pattern.y)).loss_total)
        loss, _ = mlp.train_epoch(DATASET_XOR)
    assert len(vecinos) == 50 * len(DATASET_XOR)
    assert loss >= 0.0


def test_la_perdida_de_la_epoca_es_la_media_antes_de_cada_actualizacion():
    """train_epoch promedia la perdida de los patrones en el estado inicial."""
    mlp = MLP.from_preset("simple", learning_rate=0.5)
    esperado = sum(
        mlp.forward(np.array(p.x), np.array(p.y)).loss_total for p in DATASET_SIMPLE
    ) / len(DATASET_SIMPLE)
    loss, _ = mlp.train_epoch(DATASET_SIMPLE)
    assert loss == pytest.approx(esperado)


def test_tamanos_de_la_red_segun_el_preset():
    assert MLP.from_preset("simple").n_out == 2
    assert MLP.from_preset("xor").n_out == 1
    assert MLP.from_preset("simple").n_in == 2
    assert MLP.from_preset("simple").n_hidden == 2


def test_perdida_sobre_dataset_no_modifica_los_pesos():
    mlp = MLP.from_preset("simple")
    antes = mlp.weights.flat().copy()
    mlp.dataset_loss(DATASET_SIMPLE)
    assert np.allclose(mlp.weights.flat(), antes)


# --------------------------------------------------------------------------
# Perder el ritmo: recordar el criterio de(loss, diverged)
# --------------------------------------------------------------------------


def test_perdida_sobre_dataset_es_la_media_de_las_muestras():
    mlp = MLP.from_preset("xor")
    total = sum(mlp.forward(np.array(p.x), np.array(p.y)).loss_total for p in DATASET_XOR)
    assert mlp.dataset_loss(DATASET_XOR) == pytest.approx(total / 4)