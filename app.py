"""Simulador didactico de Backpropagation en un perceptron multicapa.

Interfaz Streamlit. Toda la matematica vive en mlp.py; aqui solo se presenta
y se gestiona el estado de la sesion.

Para ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import mlp
from mlp import DATASETS, MLP, Pattern, Weights, dataset_for, num_outputs_for

st.set_page_config(
    page_title="Simulador de Backpropagation",
    page_icon="ðŸ§ ",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOTAL_STEPS = 7
ACTIVATIONS = ["sigmoid", "tanh", "relu", "identity"]

STEP_TITLES = [
    "0. Entrada",
    "1. Forward: capa oculta",
    "2. Forward: capa salida",
    "3. Funcion de perdida",
    "4. Backprop: capa salida",
    "5. Backprop: capa oculta",
    "6. Actualizacion de pesos",
]

STEP_COLORS = [
    "gray", "green", "green", "orange", "red", "red", "indigo",
]


# --------------------------------------------------------------------------
# Estado de la sesion
# --------------------------------------------------------------------------


def init_state(preset: str, learning_rate: float, hidden: str, output: str) -> None:
    dataset = dataset_for(preset)
    st.session_state.net = MLP.from_preset(
        preset, learning_rate=learning_rate, hidden_activation=hidden, output_activation=output
    )
    st.session_state.preset = preset
    st.session_state.dataset = dataset
    st.session_state.sample_index = 0
    st.session_state.step = 0
    st.session_state.epoch = 0
    st.session_state.loss_history: list[float] = []
    st.session_state.history: list[dict] = []
    st.session_state.diverged = False
    st.session_state.auto = False
    st.session_state.quiz_answers: dict[int, str] = {}


def snapshot_current() -> None:
    """Guarda el estado completo antes de avanzar un paso.

    Se apila en cada paso, no solo en la frontera de epoca, para que 'Anterior'
    retroceda paso a paso y cada pantalla muestre los valores que se vio en su
    momento, con los pesos que habia entonces. Es lo que el HTML original hacia
    mal: al retroceder mostraba z, a y delta calculados con pesos ya actualizados.
    """
    st.session_state.history.append(
        {
            "net": st.session_state.net.snapshot(),
            "step": st.session_state.step,
            "sample_index": st.session_state.sample_index,
            "epoch": st.session_state.epoch,
            "losses": list(st.session_state.loss_history),
            "diverged": st.session_state.diverged,
        }
    )
    if len(st.session_state.history) > 400:
        st.session_state.history.pop(0)


def current_pattern() -> Pattern:
    dataset = st.session_state.dataset
    return dataset[st.session_state.sample_index % len(dataset)]


def trace_for_current() -> mlp.Trace:
    pattern = current_pattern()
    return st.session_state.net.forward(np.array(pattern.x), np.array(pattern.y))


def fmt(value: float, decimals: int = 5) -> str:
    return f"{value:.{decimals}f}"


def reset_for_preset(preset: str, learning_rate: float, hidden: str, output: str) -> None:
    init_state(preset, learning_rate, hidden, output)


# --------------------------------------------------------------------------
# Controles
# --------------------------------------------------------------------------


def sidebar() -> tuple[str, float, str, str]:
    with st.sidebar:
        st.header("Configuracion")

        preset_label = st.selectbox(
            "Ejemplo didactico",
            options=list(DATASETS.keys()),
            format_func=lambda key: DATASETS[key][0],
            help="XOR usa las cuatro combinaciones y una sola neurona de salida.",
        )

        if "preset_previo" in st.session_state and st.session_state.preset_previo != preset_label:
            reset_for_preset(
                preset_label,
                st.session_state.get("learning_rate", 0.50),
                st.session_state.get("hidden_activation", "sigmoid"),
                st.session_state.get("output_activation", "sigmoid"),
            )
        st.session_state.preset_previo = preset_label

        if "net" not in st.session_state:
            reset_for_preset(preset_label, 0.50, "sigmoid", "sigmoid")

        learning_rate = st.slider(
            "Tasa de aprendizaje (eta)",
            min_value=0.01,
            max_value=2.0,
            value=float(st.session_state.net.learning_rate),
            step=0.01,
            help="Demasiado alta: los pesos oscilan y la perdida diverge.",
        )

        hidden = st.selectbox(
            "Activacion de la capa oculta",
            ACTIVATIONS,
            index=ACTIVATIONS.index(st.session_state.net.hidden_activation),
        )

        output = st.selectbox(
            "Activacion de la capa de salida",
            ACTIVATIONS,
            index=ACTIVATIONS.index(st.session_state.net.output_activation),
        )

        if st.button("Aplicar cambios"):
            net = st.session_state.net
            reset_for_preset(st.session_state.preset, learning_rate, hidden, output)

        st.divider()

        velocidad = st.select_slider(
            "Velocidad del auto-entrenamiento",
            options=[1, 5, 20, 100, 400],
            value=20,
            format_func=lambda n: f"{n} {'epoca' if n == 1 else 'epocas'} por segundo",
        )
        st.session_state.velocidad = velocidad

        objetivo = st.number_input(
            "Parar cuando la perdida baje de",
            min_value=0.0,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.5f",
        )
        st.session_state.objetivo = objetivo

        if st.button("Reiniciar la red", type="secondary"):
            reset_for_preset(
                st.session_state.preset, learning_rate, hidden, output
            )

    return preset_label, learning_rate, hidden, output


# --------------------------------------------------------------------------
# Panel de red
# --------------------------------------------------------------------------


def render_network() -> None:
    net = st.session_state.net
    trace = trace_for_current()
    pattern = current_pattern()

    st.subheader("Arquitectura y flujo de senal")
    st.caption(
        f"Entrada {len(net.n_in * [0])} -> Oculta {net.n_hidden} -> Salida {net.n_out}"
        f"   |   Muestra {st.session_state.sample_index % len(st.session_state.dataset) + 1}"
        f" de {len(st.session_state.dataset)}"
    )

    left, mid, right = st.columns(3)

    with left:
        st.markdown("**Entrada y objetivo**")
        st.latex(rf"x = ({fmt(pattern.x[0], 3)},\,{fmt(pattern.x[1], 3)})")
        targets = ", ".join(fmt(v, 3) for v in pattern.y)
        st.latex(rf"y = ({targets})")

    with mid:
        st.markdown("**Capa oculta**")
        st.markdown(
            f"- `z1 = [{', '.join(fmt(v, 4) for v in trace.z1)}]`\n"
            f"- `a1 = [{', '.join(fmt(v, 4) for v in trace.a1)}]`\n"
            f"- `delta1 = [{', '.join(fmt(v, 4) for v in trace.delta1)}]`"
        )

    with right:
        st.markdown("**Capa de salida**")
        st.markdown(
            f"- `z2 = [{', '.join(fmt(v, 4) for v in trace.z2)}]`\n"
            f"- `a2 = [{', '.join(fmt(v, 4) for v in trace.a2)}]`  <- prediccion\n"
            f"- `delta2 = [{', '.join(fmt(v, 4) for v in trace.delta2)}]`"
        )

    with st.expander("Pesos y sesgos actuales (editables)"):
        weights_df = pd.DataFrame(net.weights.as_dict()["params"])
        edited = st.data_editor(
            weights_df,
            disabled=["capa", "origen", "destino", "tipo"],
            column_config={
                "valor": st.column_config.NumberColumn("Valor", format="%.4f", step=0.01)
            },
            hide_index=True,
            width="stretch",
            key=f"editor_{st.session_state.preset}_{net.n_out}",
        )
        st.caption(
            "Editar un valor y pulsar Enter recalcula la perdida, los deltas y los "
            "gradientes sobre los pesos nuevos."
        )

        if not edited.equals(weights_df):
            aplicar_pesos_editados(net, edited)


def aplicar_pesos_editados(net: MLP, edited: pd.DataFrame) -> None:
    w1 = np.array(
        [
            [float(edited[(edited.capa == 1) & (edited.origen == f"x{i + 1}") & (edited.destino == f"h{j + 1}")]["valor"].iloc[0])
             for j in range(net.n_hidden)]
            for i in range(net.n_in)
        ]
    )
    b1 = np.array(
        [float(edited[(edited.capa == 1) & (edited.destino == f"h{j + 1}") & (edited.tipo == "sesgo")]["valor"].iloc[0])
         for j in range(net.n_hidden)]
    )
    w2 = np.array(
        [
            [float(edited[(edited.capa == 2) & (edited.origen == f"h{j + 1}") & (edited.destino == f"o{k + 1}")]["valor"].iloc[0])
             for k in range(net.n_out)]
            for j in range(net.n_hidden)
        ]
    )
    b2 = np.array(
        [float(edited[(edited.capa == 2) & (edited.destino == f"o{k + 1}") & (edited.tipo == "sesgo")]["valor"].iloc[0])
         for k in range(net.n_out)]
    )
    net.rebuild(Weights(w1=w1, b1=b1, w2=w2, b2=b2))


# --------------------------------------------------------------------------
# Desglose del paso
# --------------------------------------------------------------------------


def render_step() -> None:
    step = st.session_state.step
    net = st.session_state.net
    trace = trace_for_current()
    pattern = current_pattern()

    st.subheader(STEP_TITLES[step])

    eta = net.learning_rate
    activ_hidden = mlp.activation_formula(net.hidden_activation)
    activ_output = mlp.activation_formula(net.output_activation)
    deriv_hidden = mlp.derivative_formula(net.hidden_activation)
    deriv_output = mlp.derivative_formula(net.output_activation)

    if step == 0:
        st.markdown("**Datos de entrada**")
        entradas = r" \begin{bmatrix} " + r" \\ ".join(fmt(v, 3) for v in pattern.x) + r" \end{bmatrix}"
        objetivos = r" \begin{bmatrix} " + r" \\ ".join(fmt(v, 3) for v in pattern.y) + r" \end{bmatrix}"
        st.latex(rf"x = {entradas}")
        st.latex(rf"y = {objetivos}")
        st.info(
            f"Estos son los valores que la red debe aprender a asociar. "
            f"En el dataset actual hay {len(st.session_state.dataset)} patron(es)."
        )

    elif step == 1:
        st.markdown("**Combinacion lineal de la capa oculta**")
        st.latex(activ_hidden)
        for j in range(net.n_hidden):
            st.markdown(f"Neurona oculta $h_{{{j + 1}}}$:")
            st.latex(
                rf"z_{{h{j + 1}}} = x_1 w_{{{j + 1}1}} + x_2 w_{{{j + 1}2}} + b^{{(1)}}_{{h{j + 1}}}"
            )
            st.latex(
                rf"= {fmt(pattern.x[0], 3)} \cdot {fmt(net.weights.w1[0, j], 3)}"
                rf" + {fmt(pattern.x[1], 3)} \cdot {fmt(net.weights.w1[1, j], 3)}"
                rf" + {fmt(net.weights.b1[j], 3)} = {fmt(trace.z1[j])}"
            )
            st.latex(
                rf"a_{{h{j + 1}}} = \sigma({fmt(trace.z1[j])}) = {fmt(trace.a1[j])}"
            )
        st.info("Estas activaciones son la entrada de la capa de salida.")

    elif step == 2:
        st.markdown("**Combinacion lineal de la capa de salida**")
        st.latex(activ_output)
        for k in range(net.n_out):
            st.markdown(f"Neurona de salida $o_{{{k + 1}}}$:")
            st.latex(
                rf"z_{{o{k + 1}}} = a_{{h1}} w_{{11}} + a_{{h2}} w_{{21}} + b^{{(2)}}_{{o{k + 1}}}"
            )
            st.latex(
                rf"= {fmt(trace.a1[0], 4)} \cdot {fmt(net.weights.w2[0, k], 4)}"
                rf" + {fmt(trace.a1[1], 4)} \cdot {fmt(net.weights.w2[1, k], 4)}"
                rf" + {fmt(net.weights.b2[k], 4)} = {fmt(trace.z2[k])}"
            )
            st.latex(
                rf"a_{{o{k + 1}}} = \sigma({fmt(trace.z2[k])}) = {fmt(trace.a2[k])}"
            )
        st.success(
            f"Prediccion de la red: "
            f"`[{', '.join(fmt(v, 4) for v in trace.a2)}]`"
        )

    elif step == 3:
        st.markdown("**Funcion de perdida**")
        st.latex(r"E = \frac{1}{2}\sum_{k}\left(y_k - a_k^{(2)}\right)^2")
        for k in range(net.n_out):
            st.latex(
                rf"E_{{o{k + 1}}} = \tfrac{{1}}{{2}}\left({fmt(pattern.y[k], 4)}"
                rf" - {fmt(trace.a2[k], 4)}\right)^2 = {fmt(trace.losses[k])}"
            )
        st.latex(rf"E_{{total}} = {fmt(trace.loss_total)}")
        st.warning(
            "El objetivo es reducir $E_{total}$ hasta 0. Bajar entre N patrones da "
            "el error cuadratico medio, que es lo que se suele reportar como MSE."
        )

    elif step == 4:
        st.markdown("**Deltas de la capa de salida**")
        st.latex(r"\delta_k^{(2)} = \frac{\partial E}{\partial z_k^{(2)}} = (a_k^{(2)} - y_k)\,\sigma'(z_k^{(2)})")
        st.latex(deriv_output)
        for k in range(net.n_out):
            st.latex(
                rf"\delta_{{o{k + 1}}} = ({fmt(trace.a2[k], 5)} - {fmt(pattern.y[k], 5)})"
                rf" \cdot {fmt(trace.sigma2_prime[k], 5)} = {fmt(trace.delta2[k])}"
            )
        st.markdown("**Gradientes de los pesos de salida**")
        for j in range(net.n_hidden):
            for k in range(net.n_out):
                st.latex(
                    rf"\frac{{\partial E}}{{\partial w^{{(2)}}_{{h{j + 1}o{k + 1}}}}}"
                    rf" = a_{{h{j + 1}}} \cdot \delta_{{o{k + 1}}}"
                    rf" = {fmt(trace.a1[j], 5)} \cdot {fmt(trace.delta2[k], 5)}"
                    rf" = {fmt(trace.grad_w2[j, k])}"
                )

    elif step == 5:
        st.markdown("**Propagacion del error hacia la capa oculta**")
        st.latex(
            r"\delta_j^{(1)} = \left(\sum_{k} \delta_k^{(2)}\,w_{jk}^{(2)}\right)\sigma'(z_j^{(1)})"
        )
        st.latex(deriv_hidden)
        for j in range(net.n_hidden):
            suma = " + ".join(
                rf"\delta_{{o{k + 1}}} \cdot w^{{(2)}}_{{h{j + 1}o{k + 1}}}"
                for k in range(net.n_out)
            )
            st.latex(rf"\delta_{{h{j + 1}}} = \left({suma}\right) \cdot {fmt(trace.sigma1_prime[j], 5)}")
            st.latex(rf"= {fmt(trace.d_hidden_pre[j], 5)} \cdot {fmt(trace.sigma1_prime[j], 5)} = {fmt(trace.delta1[j])}")
        st.markdown("**Gradientes de los pesos ocultos**")
        for i in range(net.n_in):
            for j in range(net.n_hidden):
                st.latex(
                    rf"\frac{{\partial E}}{{\partial w^{{(1)}}_{{x{i + 1}h{j + 1}}}}}"
                    rf" = x_{{{i + 1}}} \cdot \delta_{{h{j + 1}}}"
                    rf" = {fmt(pattern.x[i], 5)} \cdot {fmt(trace.delta1[j], 5)}"
                    rf" = {fmt(trace.grad_w1[i, j])}"
                )

    else:
        st.markdown("**Descenso de gradiente**")
        st.latex(r"w^{(nuevo)} = w^{(viejo)} - \eta\,\frac{\partial E}{\partial w}")
        st.latex(rf"\eta = {eta}")

        rows = []
        nuevos = net.updated_weights(trace)
        for j in range(net.n_hidden):
            for k in range(net.n_out):
                rows.append(
                    {
                        "parametro": rf"w^{{(2)}}_{{h{j + 1}o{k + 1}}}",
                        "anterior": float(net.weights.w2[j, k]),
                        "gradiente": float(trace.grad_w2[j, k]),
                        "nuevo": float(nuevos.w2[j, k]),
                    }
                )
        for i in range(net.n_in):
            for j in range(net.n_hidden):
                rows.append(
                    {
                        "parametro": rf"w^{{(1)}}_{{x{i + 1}h{j + 1}}}",
                        "anterior": float(net.weights.w1[i, j]),
                        "gradiente": float(trace.grad_w1[i, j]),
                        "nuevo": float(nuevos.w1[i, j]),
                    }
                )
        for j in range(net.n_hidden):
            rows.append(
                {
                    "parametro": rf"b^{{(1)}}_{{h{j + 1}}}",
                    "anterior": float(net.weights.b1[j]),
                    "gradiente": float(trace.grad_b1[j]),
                    "nuevo": float(nuevos.b1[j]),
                }
            )
        for k in range(net.n_out):
            rows.append(
                {
                    "parametro": rf"b^{{(2)}}_{{o{k + 1}}}",
                    "anterior": float(net.weights.b2[k]),
                    "gradiente": float(trace.grad_b2[k]),
                    "nuevo": float(nuevos.b2[k]),
                }
            )

        tabla = pd.DataFrame(rows)
        st.dataframe(
            tabla.style.format(
                {"anterior": "{:+.5f}", "gradiente": "{:+.6f}", "nuevo": "{:+.5f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.success(
            "Los pesos se ajustan en la direccion que reduce el error. Pulsa "
            "'Siguiente paso' para continuar con la siguiente epoca."
        )


# --------------------------------------------------------------------------
# Navegacion
# --------------------------------------------------------------------------


def next_step() -> None:
    step = st.session_state.step
    net = st.session_state.net
    dataset = st.session_state.dataset

    if step < TOTAL_STEPS - 1:
        snapshot_current()
        st.session_state.step = step + 1
        return

    # Paso 6: aplicar la actualizacion y pasar a la siguiente muestra.
    snapshot_current()
    pattern = current_pattern()
    trace = net.forward(np.array(pattern.x), np.array(pattern.y))
    net.apply_update(trace)
    st.session_state.epoch += 1
    st.session_state.loss_history.append(trace.loss_total)
    st.session_state.sample_index = (st.session_state.sample_index + 1) % len(dataset)
    st.session_state.step = 0


def prev_step() -> None:
    if not st.session_state.history:
        return
    estado = st.session_state.history.pop()
    st.session_state.net = estado["net"]
    st.session_state.step = estado["step"]
    st.session_state.sample_index = estado["sample_index"]
    st.session_state.epoch = estado["epoch"]
    st.session_state.loss_history = estado["losses"]
    st.session_state.diverged = estado["diverged"]


def run_epoch() -> None:
    net = st.session_state.net
    dataset = st.session_state.dataset
    loss, diverged = net.train_epoch(dataset)
    st.session_state.epoch += 1
    st.session_state.loss_history.append(loss)
    st.session_state.step = TOTAL_STEPS - 1
    if diverged:
        st.session_state.diverged = True
        st.session_state.auto = False


def render_controls() -> None:
    col1, col2, col3, col4, col5 = st.columns([1, 1.2, 1, 1, 1])

    with col1:
        # on_click y no una llamada dentro del render: si se ejecutara aqui, el
        # estado cambiaria despues de que la columna izquierda ya ha pintado el
        # subtitulo, y la pantalla mostraria un paso de desfase.
        st.button(
            "Anterior",
            disabled=not st.session_state.history,
            on_click=prev_step,
        )

    with col2:
        st.button("Siguiente paso", type="primary", on_click=next_step)

    with col3:
        st.button("Epoca completa", on_click=run_epoch)

    with col4:
        if st.session_state.auto:
            st.button("Pausar", on_click=detener_auto)
        else:
            st.button("Auto-entrenar", on_click=iniciar_auto)

    with col5:
        st.button("Reiniciar", on_click=lambda: reset_actual())

    st.caption(
        f"Epoca {st.session_state.epoch}  |  Paso {st.session_state.step}/{TOTAL_STEPS - 1}"
        f"  |  Muestra {(st.session_state.sample_index % len(st.session_state.dataset)) + 1}"
        f" de {len(st.session_state.dataset)}"
    )


def reset_actual() -> None:
    reset_for_preset(
        st.session_state.preset,
        st.session_state.net.learning_rate,
        st.session_state.net.hidden_activation,
        st.session_state.net.output_activation,
    )


def iniciar_auto() -> None:
    st.session_state.auto = True


def detener_auto() -> None:
    st.session_state.auto = False


def render_loss_chart() -> None:
    st.subheader("Curva de perdida")
    history = st.session_state.loss_history
    if not history:
        st.info("Avanza pasos o ejecuta epocas para ver la curva de perdida.")
        return

    escala_log = st.checkbox("Escala logaritmica", value=False)
    datos = pd.DataFrame(
        {
            "epoca": list(range(1, len(history) + 1)),
            "perdida": history,
        }
    )

    if escala_log:
        # log10 con suelo para no producir -inf cuando la perdida llega a 0.
        columna = np.log10(datos["perdida"].clip(lower=1e-12))
        st.line_chart(
            pd.DataFrame({"epoca": datos["epoca"], "log10 perdida": columna}).set_index("epoca"),
            height=260,
        )
        st.caption(
            "Escala logaritmica: cada unidad vertical del eje es un factor 10 de "
            "perdida. Asi se aprecia la convergencia aunque caiga varios ordenes "
            "de magnitud."
        )
    else:
        st.line_chart(datos.set_index("epoca"), y="perdida", height=260)

    st.caption(
        f"Perdida actual: **{fmt(history[-1], 6)}** | "
        f"Minima alcanzada: **{fmt(min(history), 6)}** en la epoca "
        f"{history.index(min(history)) + 1}"
    )


# --------------------------------------------------------------------------
# Verificacion numerica del gradiente
# --------------------------------------------------------------------------


def render_gradient_check() -> None:
    st.subheader("Verificacion del gradiente por diferencias finitas")
    st.markdown(
        "Backpropagation es un algoritmo de calculo diferencial. Para comprobar que "
        "los deltas que calcula son correctos, contrastamos el gradiente analitico "
        "con el numerico: se perturba cada peso un epsilon y se mide cuanto cambia "
        "la perdida."
    )
    st.latex(r"\frac{\partial E}{\partial w} \approx \frac{E(w+\varepsilon) - E(w-\varepsilon)}{2\varepsilon}")

    if st.button("Ejecutar la verificacion", type="secondary"):
        pattern = current_pattern()
        st.session_state.gradient_report = mlp.gradient_report(
            st.session_state.net, np.array(pattern.x), np.array(pattern.y)
        )

    filas = st.session_state.get("gradient_report")
    if not filas:
        return

    df = pd.DataFrame(filas)
    st.dataframe(
            df.style.format(
                {"analitico": "{:+.8f}", "numerico": "{:+.8f}", "error abs": "{:.2e}", "error rel": "{:.2e}"}
            ),
            hide_index=True,
            width="stretch",
        )

    peor = df["error rel"].max()
    if peor < 1e-6:
        st.success(
            f"El gradiente analitico coincide con el numerico (error relativo maximo "
            f"{peor:.2e}). **La backpropagation esta implementada correctamente.**"
        )
    else:
        st.error(
            f"Discrepancia detectada: error relativo maximo {peor:.2e}. "
            "Revisar la implementacion de los deltas."
        )


# --------------------------------------------------------------------------
# Teoria y autoevaluacion
# --------------------------------------------------------------------------


def render_theory() -> None:
    st.subheader("Demostracion: la regla de la cadena")
    st.markdown(
        "Backpropagation aplica la regla de la cadena de forma iterativa para obtener "
        "el gradiente del error respecto a cada peso."
    )

    pestana1, pestana2 = st.tabs(["Capa de salida", "Capa oculta"])

    with pestana1:
        st.latex(r"E_k = \frac{1}{2}\left(y_k - a_k^{(2)}\right)^2")
        st.latex(
            r"\frac{\partial E}{\partial w_{jk}^{(2)}} = "
            r"\frac{\partial E}{\partial a_k^{(2)}} \cdot "
            r"\frac{\partial a_k^{(2)}}{\partial z_k^{(2)}} \cdot "
            r"\frac{\partial z_k^{(2)}}{\partial w_{jk}^{(2)}}"
        )
        st.latex(r"\delta_k^{(2)} = \left(a_k^{(2)} - y_k\right)\sigma'(z_k^{(2)})")
        st.latex(r"\frac{\partial E}{\partial w_{jk}^{(2)}} = a_j^{(1)}\cdot\delta_k^{(2)}")

    with pestana2:
        st.latex(
            r"\frac{\partial E}{\partial w_{ij}^{(1)}} = "
            r"\left(\sum_k \frac{\partial E}{\partial z_k^{(2)}}"
            r"\frac{\partial z_k^{(2)}}{\partial a_j^{(1)}}\right)"
            r"\frac{\partial a_j^{(1)}}{\partial z_j^{(1)}}"
            r"\frac{\partial z_j^{(1)}}{\partial w_{ij}^{(1)}}"
        )
        st.latex(
            r"\delta_j^{(1)} = \left(\sum_k \delta_k^{(2)}\,w_{jk}^{(2)}\right)\sigma'(z_j^{(1)})"
        )
        st.markdown(
"El error se reparte entre las neuronas de salida **en proporcion a los pesos** "
            "que las conectan: un peso grande significa que esa neurona oculta tuvo "
            "mucha responsabilidad en el error."
        )


QUIZ = [
    {
        "pregunta": "Por que es necesaria la derivada $\\sigma'$ de la activacion durante el backpropagation?",
        "opciones": {
            "a": "Para calcular la tasa de aprendizaje $\\eta$.",
            "b": "Por la regla de la cadena, al derivar la activacion respecto a su suma ponderada $z$.",
            "c": "Para convertir pesos negativos en positivos.",
        },
        "correcta": "b",
        "explicacion": "Al derivar respecto a $z$ aparece el factor $\\partial a/\\partial z = \\sigma'(z)$, que es lo que escala el error local.",
    },
    {
        "pregunta": "Si la tasa de aprendizaje es demasiado alta ($\\eta = 10$), que suele ocurrir?",
        "opciones": {
            "a": "El algoritmo converge mas rapido y de forma estable.",
            "b": "Los pesos oscilan y la perdida puede diverger.",
            "c": "Los deltas se vuelven exactamente cero.",
        },
        "correcta": "b",
        "explicacion": "Pasos demasiado grandes hacen que el error oscile alrededor del minimo y, con salida no acotada, que la perdida crezca sin control.",
    },
    {
        "pregunta": "Por que el gradiente de la salida es $(a - y)$ y no $(y - a)$?",
        "opciones": {
            "a": "Por convencion, para que el signo sea positivo al actualizar.",
            "b": "Es un error tipico de la notacion.",
            "c": "Depende de como se defina la funcion de perdida.",
        },
        "correcta": "a",
        "explicacion": "Al derivar $\\frac{1}{2}(y-a)^2$ sale $(y-a)\\cdot(-1) = a-y$. Como se resta el gradiente, este signo hace que el peso **suba** cuando la salida es mayor que el objetivo.",
    },
    {
        "pregunta": "Para que sirve el sesgo $b$?",
        "opciones": {
            "a": "Para desplazar la activacion y permitir que la neurona aprenda un umbral.",
            "b": "Para penalizar pesos grandes.",
            "c": "No cambia nada, es opcional.",
        },
        "correcta": "a",
        "explicacion": "Sin sesgo, la activacion pasa siempre por el origen. El sesgo permite desplazar la frontera de decision, que es imprescindible en XOR.",
    },
    {
        "pregunta": "Que efecto tiene usar ReLU en la capa de salida cuando el objetivo es un valor negativo, como $-0.5$?",
        "opciones": {
            "a": "Nada, ReLU se ajusta automaticamente.",
            "b": "Es inalcanzable: ReLU solo produce valores mayores o iguales que cero.",
            "c": "El error se reduce mas rapido.",
        },
        "correcta": "b",
        "explicacion": "ReLU nunca devuelve negativos. El mejor error posible con objetivo $-0.5$ es $\\frac{1}{2}(0.5)^2 = 0.125$, y nunca llega a cero.",
    },
    {
        "pregunta": "Que ocurre si $\\eta$ es demasiado pequena, como $0.001$?",
        "opciones": {
            "a": "El entrenamiento se detiene de inmediato.",
            "b": "Los pesos apenas cambian y el aprendizaje es muy lento, aunque noæŠ¥é”™.",
            "c": "La perdida se reduce mas rapido.",
        },
        "correcta": "b",
        "explicacion": "Los pasos son tan cortos que hacen falta muchisimas epocas. No falla: simplemente converge muy despacio.",
    },
    {
        "pregunta": "Por que la backpropagation necesita una capa oculta para resolver XOR?",
        "opciones": {
            "a": "Porque dos entradas exigen dos capas.",
            "b": "XOR no es linealmente separable: hace falta combinar rasgos en un espacio nuevo.",
            "c": "Por eficiencia de calculo.",
        },
        "correcta": "b",
        "explicacion": "Un solo perceptron produce una sola frontera lineal. La capa oculta transforma las entradas para que XOR se vuelva linealmente separable en ese espacio.",
    },
    {
        "pregunta": "Que indica que la curva de perdida ha dejado de bajar?",
        "opciones": {
            "a": "Que la red ha alcanzado un minimo local o que la tasa de aprendizaje es inadequate.",
            "b": "Que hay un error en el codigo.",
            "c": "Que hay que reiniciar siempre.",
        },
        "correcta": "a",
        "explicacion": "Puede ser un minimo local, unlearning rate demasiado pequeno o demasiado grande. La verificacion del gradiente ayuda a distinguir si el gradiente es cero de verdad.",
    },
]


def render_quiz() -> None:
    st.markdown(f"**{len(QUIZ)} preguntas.** Responde, revisa la explicacion y comprueba tu nota al final.")
    if st.button("Reiniciar el cuestionario"):
        st.session_state.quiz_answers = {}

    for index, pregunta in enumerate(QUIZ, start=1):
        with st.expander(f"Pregunta {index}: {pregunta['pregunta']}", expanded=False):
            respondida = index in st.session_state.quiz_answers
            letras = list(pregunta["opciones"].keys())

            eleccion = st.radio(
                f"pregunta_{index}",
                options=letras,
                index=None,
                format_func=lambda letra, q=pregunta: f"{letra.upper()}) {q['opciones'][letra]}",
                key=f"quiz_{index}",
                disabled=respondida,
                label_visibility="collapsed",
            )

            # index=None hace que el radio empiece vacio. Solo se registra la
            # respuesta cuando el alumno elige algo, nunca por el valor por defecto.
            if not respondida:
                # El boton se crea siempre: si se escribiera como
                # `eleccion is not None and st.button(...)`, el cortocircuito de
                # `and` impediria dibujarlo hasta haber elegido opcion.
                enviado = st.button("Responder", key=f"enviar_{index}")
                if enviado and eleccion is not None:
                    st.session_state.quiz_answers[index] = eleccion
                    st.rerun()
                continue

            elegida = st.session_state.quiz_answers[index]
            if elegida == pregunta["correcta"]:
                st.success("Correcto.")
            else:
                st.error(f"Incorrecto. La respuesta correcta es **{pregunta['correcta'].upper()}**.")
            st.info(pregunta["explicacion"])

    respondidas = st.session_state.quiz_answers
    if not respondidas:
        return

    aciertos = sum(
        1 for i, letra in respondidas.items() if letra == QUIZ[i - 1]["correcta"]
    )
    nota = 100.0 * aciertos / len(QUIZ)
    st.progress(min(nota / 100.0, 1.0))
    if nota >= 75:
        st.success(f"Nota: **{nota:.0f}/100** ({aciertos} de {len(QUIZ)}).")
    else:
        st.warning(f"Nota: **{nota:.0f}/100** ({aciertos} de {len(QUIZ)}). Repasa la teoria e intentalo otra vez.")


# --------------------------------------------------------------------------
# Layout principal
# --------------------------------------------------------------------------


def render_header() -> None:
    st.title("Simulador de Backpropagation en un perceptron multicapa")
    st.markdown(
        "Descompone numericamente cada fase del algoritmo: propagacion hacia delante, "
        "calculo de la perdida, retropropagacion de los deltas y actualizacion de los pesos."
    )
    st.session_state.setdefault("gradient_report", None)


def render_notices() -> None:
    net = st.session_state.net
    avisos = mlp.activation_target_warnings(net.output_activation, st.session_state.dataset)
    for aviso in avisos:
        st.warning(aviso)

    if st.session_state.diverged:
        st.error(
            "El entrenamiento diverge: la perdida crece sin control. "
            "Baja la tasa de aprendizaje (eta) y pulsa 'Reiniciar'."
        )


def render_offline_download() -> None:
    from pathlib import Path

    ruta = Path(__file__).parent / "static" / "simulador_offline.html"
    if not ruta.exists():
        return
    with open(ruta, "rb") as archivo:
        datos = archivo.read()
    st.download_button(
        "Descargar la version HTML offline",
        data=datos,
        file_name="simulador_offline.html",
        mime="text/html",
    )
    st.caption(
        "Version HTML autocontenida para usar sin conexion. Esta congelada: "
        "la version mantenida es esta aplicacion."
    )


def main() -> None:
    render_header()
    sidebar()

    if st.session_state.auto:
        ejecutar_auto()

    render_notices()

    izquierda, derecha = st.columns([3, 2])

    with izquierda:
        render_network()
        st.divider()
        render_step()

    with derecha:
        render_controls()
        render_loss_chart()
        st.divider()
        render_gradient_check()

    st.divider()
    teoria, evaluacion, descarga = st.tabs(["Teoria", "Autoevaluacion", "Descarga"])
    with teoria:
        render_theory()
    with evaluacion:
        render_quiz()
    with descarga:
        render_offline_download()


@st.fragment(run_every=None)
def ejecutar_auto() -> None:
    """Entrena automaticamente hasta alcanzar la perdida objetivo o diverger.

    Cada fragmento ejecuta una epoca y pide un rerun, de forma que el alumno ve
    la curva de perdida crecer en tiempo real.
    """
    velocidad = st.session_state.get("velocidad", 20)
    objetivo = st.session_state.get("objetivo", 0.001)

    epocas_por_tick = max(1, velocidad // 20)
    for _ in range(epocas_por_tick):
        run_epoch()
        if st.session_state.diverged:
            st.session_state.auto = False
            st.rerun()
            return

    perdida = st.session_state.loss_history[-1]
    if perdida < objetivo:
        st.session_state.auto = False
        st.success(f"Entrenamiento detenido: perdida {perdida:.6f} por debajo de {objetivo}.")
    else:
        st.rerun()


if __name__ == "__main__":
    main()