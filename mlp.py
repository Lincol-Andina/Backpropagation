"""Nucleo matematico del simulador de Backpropagation.

Fuente unica de verdad de la matematica del proyecto. No importa nada de
Streamlit ni de la interfaz: es codigo puro y testeable de forma aislada.

Red: perceptron multicapa de una capa oculta.

    x --W1,b1--> z1 --sigma--> a1 --W2,b2--> z2 --g--> a2

Convencion de indices (igual que en el material de clase):

    W1[i][j] = peso que va de la entrada i a la neurona oculta j
    W2[j][k] = peso que va de la neurona oculta j a la neurona de salida k

Por eso la activacion lineal es  z1 = W1.T @ x + b1  y  z2 = W2.T @ a1 + b2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import numpy as np

Activation = Literal["sigmoid", "tanh", "relu", "identity"]

# Numero tras el cual exp(-z) desborda en coma flotante. Recortar aqui mantiene
# el resultado exacto para |z| < 60 y evita inf/NaN en el resto.
Z_CLIP = 60.0

DIVERGENCE_THRESHOLD = 1e6


# --------------------------------------------------------------------------
# Funciones de activacion y sus derivadas
# --------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -Z_CLIP, Z_CLIP)))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def identity(z: np.ndarray) -> np.ndarray:
    return z


_ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str, str]] = {
    "sigmoid": (sigmoid, "Sigmoide", r"\sigma(z)=\frac{1}{1+e^{-z}}"),
    "tanh": (tanh, "Tangente hiperbolica", r"\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}"),
    "relu": (relu, "ReLU", r"\mathrm{ReLU}(z)=\max(0,z)"),
    "identity": (identity, "Lineal", r"g(z)=z"),
}

_DERIVATIVES: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "sigmoid": lambda a, z: a * (1.0 - a),
    "tanh": lambda a, z: 1.0 - a**2,
    "relu": lambda a, z: (z > 0).astype(float),
    "identity": lambda a, z: np.ones_like(a),
}


def activation_names() -> dict[str, str]:
    """Nombre legible de cada activacion, para las etiquetas de la interfaz."""
    return {key: value[1] for key, value in _ACTIVATIONS.items()}


def activation_formula(name: str) -> str:
    """Formula en LaTeX de la activacion, para mostrarla junto a la derivada."""
    return _ACTIVATIONS[name][2]


def derivative_formula(name: str) -> str:
    """Formula en LaTeX de sigma'(z), correspondiente a la activacion elegida."""
    return {
        "sigmoid": r"\sigma'(z)=\sigma(z)\left(1-\sigma(z)\right)",
        "tanh": r"\tanh'(z)=1-\tanh^{2}(z)",
        "relu": r"\mathrm{ReLU}'(z)=\begin{cases}1 & z>0\\ 0 & z\leq 0\end{cases}",
        "identity": r"g'(z)=1",
    }[name]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """Un par entrada/objetivo del dataset."""

    x: tuple[float, ...]
    y: tuple[float, ...]

    @property
    def label(self) -> str:
        return f"({', '.join(f'{v:g}' for v in self.x)}) -> ({', '.join(f'{v:g}' for v in self.y)})"


# El ejemplo numerico clasico de la teoria: una sola muestra, dos salidas.
DATASET_SIMPLE: tuple[Pattern, ...] = (
    Pattern(x=(0.05, 0.10), y=(0.01, 0.99)),
)

# XOR completo: las cuatro combinaciones. Una sola neurona de salida.
DATASET_XOR: tuple[Pattern, ...] = (
    Pattern(x=(0.0, 0.0), y=(0.0,)),
    Pattern(x=(0.0, 1.0), y=(1.0,)),
    Pattern(x=(1.0, 0.0), y=(1.0,)),
    Pattern(x=(1.0, 1.0), y=(0.0,)),
)

DATASETS: dict[str, tuple[str, tuple[Pattern, ...]]] = {
    "simple": ("Ejemplo numerico estandar (2-2-2)", DATASET_SIMPLE),
    "xor": ("Logica XOR completa (2-2-1, 4 patrones)", DATASET_XOR),
}


def dataset_for(preset: str) -> tuple[Pattern, ...]:
    return DATASETS.get(preset, DATASET_SIMPLE)[1]


def num_outputs_for(preset: str) -> int:
    return len(dataset_for(preset)[0].y)


# --------------------------------------------------------------------------
# Estado de la red
# --------------------------------------------------------------------------


@dataclass
class Weights:
    """Parametros de la red. Separado de las activaciones para poder clonar."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def copy(self) -> "Weights":
        return Weights(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def flat(self) -> np.ndarray:
        """Todos los parametros en un vector, en orden estable."""
        return np.concatenate([self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()])

    @classmethod
    def from_flat(cls, vector: np.ndarray, n_in: int, n_hidden: int, n_out: int) -> "Weights":
        """Reconstruye los parametros desde un vector plano producedo por flat()."""
        cursor = 0

        def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            chunk = vector[cursor : cursor + count].reshape(shape)
            cursor += count
            return chunk.copy()

        return cls(
            w1=take(n_in * n_hidden, (n_in, n_hidden)),
            b1=take(n_hidden, (n_hidden,)),
            w2=take(n_hidden * n_out, (n_hidden, n_out)),
            b2=take(n_out, (n_out,)),
        )

    def as_dict(self) -> dict[str, Any]:
        """Representacion plana, lista para st.data_editor."""
        rows: list[dict[str, Any]] = []
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                rows.append({"capa": 1, "origen": f"x{i + 1}", "destino": f"h{j + 1}", "tipo": "peso", "valor": float(self.w1[i, j])})
        for j in range(self.b1.shape[0]):
            rows.append({"capa": 1, "origen": "-", "destino": f"h{j + 1}", "tipo": "sesgo", "valor": float(self.b1[j])})
        for j in range(self.w2.shape[0]):
            for k in range(self.w2.shape[1]):
                rows.append({"capa": 2, "origen": f"h{j + 1}", "destino": f"o{k + 1}", "tipo": "peso", "valor": float(self.w2[j, k])})
        for k in range(self.b2.shape[0]):
            rows.append({"capa": 2, "origen": "-", "destino": f"o{k + 1}", "tipo": "sesgo", "valor": float(self.b2[k])})
        return {"params": rows}


def default_weights(preset: str = "simple") -> Weights:
    """Pesos iniciales, iguales a los del material original para no romper el ejemplo."""
    if preset == "xor":
        return Weights(
            w1=np.array([[0.50, -0.20], [0.30, 0.80]]),
            b1=np.array([-0.10, 0.20]),
            w2=np.array([[0.70], [-0.40]]),
            b2=np.array([0.10]),
        )
    return Weights(
        w1=np.array([[0.15, 0.25], [0.20, 0.30]]),
        b1=np.array([0.35, 0.35]),
        w2=np.array([[0.40, 0.50], [0.45, 0.55]]),
        b2=np.array([0.60, 0.60]),
    )


@dataclass
class Trace:
    """Todo lo que produjo un forward+backward, listo para pintar en pantalla."""

    x: np.ndarray
    y: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    d_out: np.ndarray
    d_hidden_pre: np.ndarray
    sigma1_prime: np.ndarray
    sigma2_prime: np.ndarray
    delta2: np.ndarray
    delta1: np.ndarray
    grad_w1: np.ndarray
    grad_b1: np.ndarray
    grad_w2: np.ndarray
    grad_b2: np.ndarray
    losses: np.ndarray
    loss_total: float

    def clone(self) -> "Trace":
        """Copia profunda. loss_total es un float escalar y se copia por valor."""
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            current = getattr(self, name)
            values[name] = current.copy() if isinstance(current, np.ndarray) else current
        return Trace(**values)


@dataclass
class MLP:
    """Perceptron multicapa de una capa oculta, con estado y activaciones a la vista."""

    weights: Weights
    learning_rate: float = 0.50
    hidden_activation: Activation = "sigmoid"
    output_activation: Activation = "sigmoid"
    trace: Trace | None = None

    # -- dimensiones -------------------------------------------------------

    @property
    def n_in(self) -> int:
        return self.weights.w1.shape[0]

    @property
    def n_hidden(self) -> int:
        return self.weights.b1.shape[0]

    @property
    def n_out(self) -> int:
        return self.weights.b2.shape[0]

    # -- fase forward ------------------------------------------------------

    def _forward_arrays(
        self, x: np.ndarray, weights: Weights
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = weights.w1.T @ x + weights.b1
        a1 = _ACTIVATIONS[self.hidden_activation][0](z1)
        z2 = weights.w2.T @ a1 + weights.b2
        a2 = _ACTIVATIONS[self.output_activation][0](z2)
        return z1, a1, z2, a2

    def forward(self, x: np.ndarray, y: np.ndarray, weights: Weights | None = None) -> Trace:
        """Propaga hacia delante y calcula la perdida.

        No toca los pesos. El parametro `weights` permite evaluar la red con otros
        parametros, que es lo que necesita la verificacion por diferencias finitas.
        """
        w = self.weights if weights is None else weights
        z1, a1, z2, a2 = self._forward_arrays(x, w)
        diff = y - a2
        losses = 0.5 * diff**2

        sigma1_prime = _DERIVATIVES[self.hidden_activation](a1, z1)
        sigma2_prime = _DERIVATIVES[self.output_activation](a2, z2)

        # dE/da2 = a2 - y. El signo sale de derivar (y-a)^2/2.
        d_out = a2 - y
        delta2 = d_out * sigma2_prime
        d_hidden_pre = w.w2 @ delta2
        delta1 = d_hidden_pre * sigma1_prime

        return Trace(
            x=x.copy(),
            y=y.copy(),
            z1=z1,
            a1=a1,
            z2=z2,
            a2=a2,
            d_out=d_out,
            d_hidden_pre=d_hidden_pre,
            sigma1_prime=sigma1_prime,
            sigma2_prime=sigma2_prime,
            delta2=delta2,
            delta1=delta1,
            grad_w1=np.outer(x, delta1),
            grad_b1=delta1,
            grad_w2=np.outer(a1, delta2),
            grad_b2=delta2,
            losses=losses,
            loss_total=float(losses.sum()),
        )

    # -- fase backward -----------------------------------------------------

    def backward(self, x: np.ndarray, y: np.ndarray) -> Trace:
        """Calcula deltas y gradientes sin modificar los pesos."""
        return self.forward(x, y)

    # -- actualizacion -----------------------------------------------------

    def updated_weights(self, trace: Trace, learning_rate: float | None = None) -> Weights:
        """Devuelve pesos nuevos aplicando descenso de gradiente. No muta la red."""
        eta = self.learning_rate if learning_rate is None else learning_rate
        w = self.weights
        return Weights(
            w1=w.w1 - eta * trace.grad_w1,
            b1=w.b1 - eta * trace.grad_b1,
            w2=w.w2 - eta * trace.grad_w2,
            b2=w.b2 - eta * trace.grad_b2,
        )

    def apply_update(self, trace: Trace, learning_rate: float | None = None) -> None:
        self.weights = self.updated_weights(trace, learning_rate)
        self.trace = trace

    # -- perdida sobre un dataset -----------------------------------------

    def dataset_loss(self, dataset: tuple[Pattern, ...]) -> float:
        """Perdida media sobre el dataset, sin actualizar pesos."""
        total = sum(self.forward(np.array(p.x), np.array(p.y)).loss_total for p in dataset)
        return total / len(dataset)

    def train_epoch(
        self, dataset: tuple[Pattern, ...], learning_rate: float | None = None
    ) -> tuple[float, bool]:
        """Una epoca: recorre todo el dataset actualizando tras cada patron.

        Devuelve (perdida media de la epoca, diverged). La perdida que se promedia
        es la de antes de cada actualizacion, que es la convencion habitual.
        """
        losses: list[float] = []
        for pattern in dataset:
            trace = self.forward(np.array(pattern.x), np.array(pattern.y))
            losses.append(trace.loss_total)
            self.apply_update(trace, learning_rate)
            if self.is_diverged(trace.loss_total):
                return float(np.mean(losses)), True
        return float(np.mean(losses)), False

    # -- divergencia -------------------------------------------------------

    @staticmethod
    def is_diverged(loss: float) -> bool:
        return not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD

    # -- instantaneas ------------------------------------------------------

    def snapshot(self) -> "MLP":
        """Copia profunda del estado, incluidos pesos, ajustes y traza."""
        return replace(
            self,
            weights=self.weights.copy(),
            trace=self.trace.clone() if self.trace is not None else None,
        )

    @classmethod
    def from_preset(
        cls,
        preset: str = "simple",
        learning_rate: float = 0.50,
        hidden_activation: Activation = "sigmoid",
        output_activation: Activation | None = None,
    ) -> "MLP":
        return cls(
            weights=default_weights(preset),
            learning_rate=learning_rate,
            hidden_activation=hidden_activation,
            output_activation=output_activation or hidden_activation,
        )

    def rebuild(self, weights: Weights) -> None:
        """Sustituye los pesos manteniendo el resto de ajustes."""
        self.weights = weights


# --------------------------------------------------------------------------
# Verificacion numerica del gradiente
# --------------------------------------------------------------------------


def numeric_gradient(
    mlp: MLP,
    x: np.ndarray,
    y: np.ndarray,
    weights: Weights | None = None,
    h: float = 1e-5,
) -> np.ndarray:
    """Gradiente por diferencias finitas centrales, para contrastar con el analitico.

    Solo se usa para verificar: es lento y solo sirve con perdida escalar.
    """
    w = weights if weights is not None else mlp.weights
    n_in, n_hidden = w.w1.shape
    n_out = w.w2.shape[1]
    grad = np.zeros(w.flat().shape[0])

    for index in range(grad.size):
        vector_plus = w.flat()
        vector_plus[index] += h
        vector_minus = w.flat()
        vector_minus[index] -= h

        loss_plus = mlp.forward(
            x, y, weights=Weights.from_flat(vector_plus, n_in, n_hidden, n_out)
        ).loss_total
        loss_minus = mlp.forward(
            x, y, weights=Weights.from_flat(vector_minus, n_in, n_hidden, n_out)
        ).loss_total

        grad[index] = (loss_plus - loss_minus) / (2.0 * h)

    return grad


def gradient_report(
    mlp: MLP, x: np.ndarray, y: np.ndarray, h: float = 1e-5
) -> list[dict[str, Any]]:
    """Compara gradiente analitico y numerico peso a peso."""
    trace = mlp.forward(x, y)
    w = mlp.weights
    n_in, n_hidden = w.w1.shape
    n_out = w.w2.shape[1]

    analytic_parts = [trace.grad_w1, trace.grad_b1, trace.grad_w2, trace.grad_b2]
    analytic = np.concatenate([part.ravel() for part in analytic_parts])
    numeric = numeric_gradient(mlp, x, y, weights=w, h=h)

    labels: list[str] = []
    for i in range(n_in):
        for j in range(n_hidden):
            labels.append(f"W1[{i},{j}]  w({'x'}{i + 1}, h{j + 1})")
    for j in range(n_hidden):
        labels.append(f"b1[{j}]  b(h{j + 1})")
    for j in range(n_hidden):
        for k in range(n_out):
            labels.append(f"W2[{j},{k}]  w(h{j + 1}, o{k + 1})")
    for k in range(n_out):
        labels.append(f"b2[{k}]  b(o{k + 1})")

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        a = float(analytic[index])
        n = float(numeric[index])
        rows.append(
            {
                "parametro": label,
                "analitico": a,
                "numerico": n,
                "error abs": abs(a - n),
                "error rel": abs(a - n) / max(abs(a), abs(n), 1e-12),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Comprobaciones de coherencia de la configuracion
# --------------------------------------------------------------------------


def activation_target_warnings(
    output_activation: str, dataset: tuple[Pattern, ...]
) -> list[str]:
    """Avisos sobre combinaciones activacion de salida / objetivos que no encajan.

    ReLU solo produce valores >= 0: alcanza sin problema un objetivo de 0.01,
    pero es incapaz de producir cualquiera negativo. El problema aparece con
    objetivos negativos o cuando se quiere que la salida pueda bajar de cero.
    """
    warnings: list[str] = []
    min_target = min(min(p.y) for p in dataset)
    max_target = max(max(p.y) for p in dataset)

    if output_activation == "relu" and min_target < 0.0:
        warnings.append(
            f"La activacion de salida ReLU solo genera valores >= 0, pero hay objetivos "
            f"negativos (minimo {min_target:g}). Esos objetivos son inalcanzables: "
            "usa sigmoide, tangente hiperbolica o lineal en la capa de salida."
        )
    if output_activation == "relu" and min_target >= 0.0 and max_target <= 1.0:
        warnings.append(
            "ReLU en la capa de salida funciona con estos objetivos, pero es una "
            "decision rara: lo habitual es reservar ReLU para la capa oculta y usar "
            "sigmoide en la salida. La razon es que ReLU nunca produce salidas negativas, "
            "asi que el umbral de decision queda siempre en 0."
        )
    if output_activation == "identity":
        warnings.append(
            "La salida es lineal y no acotada, asi que los pesos pueden crecer mucho: "
            "espera una curva de perdida inestable, especialmente con eta alta."
        )
    if output_activation == "tanh":
        warnings.append(
            "tanh en la salida solo alcanza el intervalo (-1, 1). Si tus objetivos estan "
            "fuera de ese rango, el error no podra llegar a cero."
        )
        if max_target > 1.0 or min_target < -1.0:
            warnings.append(
                f"Hay objetivos fuera de (-1, 1): minimo {min_target:g}, maximo {max_target:g}. "
                "Con tanh en la salida son inalcanzables."
            )
    return warnings