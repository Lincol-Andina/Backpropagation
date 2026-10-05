# Simulador de Backpropagation en un perceptrón multicapa

Aplicación didáctica que descompone numéricamente cada fase del algoritmo de
retropropagación: propagación hacia delante, cálculo de la pérdida, deltas,
gradientes y actualización de pesos.

Material del curso de Inteligencia Artificial.

## Qué incluye

| Pestaña / sección | Contenido |
|---|---|
| **Configuración** (lateral) | Ejemplo (numérico o XOR), tasa de aprendizaje, activación de la capa oculta y de la de salida, velocidad de auto-entrenamiento |
| **Arquitectura** | Valores `z`, `a` y `δ` de cada capa en tiempo real; pesos y sesgos editables |
| **Recorrido de 7 pasos** | Desglose matemático de cada fase, con los números concretos de la red |
| **Curva de pérdida** | Evolución por época, con escala logarítmica opcional |
| **Verificación del gradiente** | Contrasta el gradiente analítico con el numérico por diferencias finitas |
| **Teoría** | Demostración de la regla de la cadena |
| **Autoevaluación** | 8 preguntas con explicación y nota |

## Estructura

```
.
├── app.py                        Interfaz Streamlit
├── mlp.py                        Núcleo matemático (sin dependencias de UI)
├── tests/
│   ├── test_mlp.py               Verificación del gradiente y del entrenamiento
│   ├── test_app_smoke.py         Comprobaciones de estructura
│   └── test_app_interaccion.py   Pruebas de la interfaz ejecutándola de verdad
├── static/
│   └── simulador_offline.html    Versión HTML congelada (ver más abajo)
├── requirements.txt
└── .streamlit/config.toml
```

**Toda la matemática vive en `mlp.py`.** La interfaz solo presenta. Un test
comprueba que ninguna fórmula se duplique en `app.py`.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar los tests

```bash
pytest -q
```

83 tests. Los que importan de verdad:

- `test_gradiente_analitico_coincide_con_diferencias_finitas_*` — comprueba que
  los deltas de backpropagation coinciden con el gradiente numérico en las cuatro
  activaciones, con error relativo por debajo de `1e-6`.
- `test_anterior_reconstruye_la_tabla_de_actualizacion` — verifica que el botón
  *Anterior* reconstruye exactamente el estado anterior, includedo los pesos.
- `test_xor_recorre_las_cuatro_combinaciones` — comprueba que una época recorre
  los cuatro patrones de XOR.

## Notas sobre el modelo

**Pérdida.** Se calcula `E = ½ Σ (y − a)²`, que es el error cuadrático, no el
medio. La interfaz lo indica: al dividir entre el número de patrones se obtiene
el error cuadrático medio.

**Divergencia.** Con salida sigmoide la pérdida está acotada, así que una tasa
de aprendizaje enorme produce oscilación pero no divergencia numérica. La
divergencia explosiva requiere una salida no acotada (lineal o ReLU sin
restricción). El simulador detecta ambas situaciones.

**ReLU en la salida.** Alcanza sin problema un objetivo de `0.01`. Su
limitación real es que nunca produce salidas negativas, por lo que un objetivo
como `-0.5` es inalcanzable y el mejor error posible es `0.125`.

**Activaciones.** Sigmoide, tangente hiperbólica, ReLU y lineal. Se eligen por
separado para la capa oculta y la de salida.

## La versión HTML congelada

`static/simulador_offline.html` es una versión anterior de un solo archivo,
conservada para poder usarla sin conexión. **No se mantiene** y tiene errores
conocidos, señalados en un aviso dentro del propio archivo:

- El texto matemático se muestra con signos `$` sin renderizar.
- El ejemplo XOR solo entrena uno de los cuatro patrones.
- El botón *Anterior* no reconstruye los valores del paso.
- El paso 6 muestra la actualización siguiente, no la aplicada.

Use esta aplicación para la versión correcta.

## Desplegar en Streamlit Community Cloud

Requiere una cuenta de GitHub. La app es pública por defecto.

1. Sube el repositorio a GitHub (rama `main`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) y crea una cuenta
   conectando tu cuenta de GitHub.
3. Pulsa **Create app** → **I already have an app**.
4. Indica el repositorio, la rama `main` y el fichero de entrada `app.py`.
5. Elige un subdominio, por ejemplo `backprop-mi-apellido`.
6. **Deploy**. Tarda unos minutos la primera vez.

Cada `git push` a `main` actualiza la aplicación en segundos.

Los registros de la aplicación solo son visibles para quien tenga permiso de
escritura en el repositorio.

### Notas del despliegue

- El repositorio necesita `requirements.txt` en la raíz. Las versiones están
  fijadas para que el arranque sea reproducible.
- Solo puede haber un `.streamlit/config.toml` por repositorio, y debe estar en
  la raíz aunque el entrypoint esté en un subdirectorio.
- Si la aplicación deja de arrancar tras una actualización de Streamlit,
  fija la versión de Python desde *Advanced settings* al desplegar.

## Uso en clase

- **Demostrar los 7 pasos:** pulsa *Siguiente paso* y comenta qué cambia en cada
  pantalla. El botón *Anterior* reconstruye el estado exacto, así que puedes
  volver atrás sin perder la coherencia.
- **Contrastar activaciones:** cambia la de la capa de salida a lineal y observa
  la curva de pérdida; luego prueba con ReLU y observa el aviso.
- **Demostrar la divergencia:** pon la salida lineal y `η = 1.5`, y pulsa
  *Auto-entrenar*.
- **Verificar el cálculo:** el botón de verificación contrasta el gradiente
  analítico con el numérico. Es la forma de demostrar empíricamente que la
  backpropagation está bien implementada.

## Requisitos

Python 3.10 o superior. Probado con Python 3.14.3.