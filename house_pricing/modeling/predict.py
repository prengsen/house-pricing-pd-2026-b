"""Inferencia con el modelo ganador de house_pricing.

Carga el pipeline completo generado en los notebooks de investigación
(``models/modelo_ganador.joblib``: selección de features, feature
engineering, modelo ganador y reversión a dólares) y lo expone mediante
la función :func:`predecir`, pensada para ser consumida por una API
(FastAPI) o por línea de comandos.

Controles de calidad entre la entrada y la salida:

* Errores de estructura (detienen la ejecución con
  :class:`ErrorValidacionEntrada`): entrada vacía, columnas requeridas
  faltantes o valores no numéricos en columnas numéricas.
* Advertencias (se registran y la ejecución continúa): columnas
  desconocidas, valores faltantes que serán imputados y categorías no
  vistas durante el entrenamiento.
* Coherencia de la salida: cada predicción debe ser finita y estar en el
  rango de ``SalePrice`` de ``train.csv``. Las filas incoherentes se
  marcan con ``es_coherente=False`` y ``SalePrice`` nulo.

Todo se registra en ``house_pricing/modeling/logs/predict.log``.

Uso por línea de comandos (prueba con observaciones aleatorias de test)::

    python -m house_pricing.modeling.predict --n 10 --seed 42
"""

import argparse
from functools import lru_cache
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJ_ROOT / "models" / "modelo_ganador.joblib"
CONFIG_PATH = PROJ_ROOT / "pipeline_config.yaml"
TEST_DATA_PATH = PROJ_ROOT / "data" / "raw" / "test.csv"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "predict.log"

# Rango de SalePrice observado en data/raw/train.csv
PRECIO_MINIMO = 34_900.0
PRECIO_MAXIMO = 755_000.0

COLUMNA_ID = "Id"
COLUMNA_PREDICCION = "SalePrice"
COLUMNA_COHERENTE = "es_coherente"

LOGGER_NAME = "house_pricing.predict"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class ErrorValidacionEntrada(ValueError):
    """Error de estructura en los datos de entrada."""


def configurar_logger(log_path: Path = LOG_PATH) -> logging.Logger:
    """Configura el logger de inferencia para escribir en un archivo.

    Todas las ejecuciones se agregan al mismo archivo; cada línea lleva
    su fecha y hora. La función es idempotente: no duplica handlers si
    se llama varias veces con la misma ruta.

    Args:
        log_path: Ruta del archivo de log.

    Returns:
        Logger configurado.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    ruta = str(log_path.resolve())
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == ruta:
                return logger
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(ruta, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    return logger


def cargar_features(config_path: Path = CONFIG_PATH) -> list[str]:
    """Lee la lista de features del modelo desde ``pipeline_config.yaml``.

    Args:
        config_path: Ruta del archivo de configuración.

    Returns:
        Lista de nombres de las features requeridas (``MODEL_FEATURES``).
    """
    with open(config_path, encoding="utf-8") as archivo:
        return list(yaml.safe_load(archivo)["MODEL_FEATURES"])


@lru_cache(maxsize=1)
def cargar_modelo(
    model_path: Path = MODEL_PATH,
) -> TransformedTargetRegressor:
    """Carga el pipeline completo del modelo ganador.

    El resultado se guarda en caché para no leer el archivo en cada
    predicción (útil al servir el modelo desde una API).

    Args:
        model_path: Ruta del archivo ``.joblib`` del modelo.

    Returns:
        Pipeline ajustado que recibe datos crudos y devuelve dólares.

    Raises:
        FileNotFoundError: Si el archivo del modelo no existe.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    return joblib.load(model_path)


def _columnas_numericas(modelo: TransformedTargetRegressor) -> list[str]:
    """Obtiene las features numéricas a partir de los imputadores."""
    fe = modelo.regressor_.named_steps["fe"]
    return (
        list(fe.named_steps["imputar_mediana"].variables_)
        + list(fe.named_steps["imputar_aleatorio"].variables_)
    )


def _categorias_conocidas(
    modelo: TransformedTargetRegressor,
) -> dict[str, set[str]]:
    """Obtiene las categorías vistas en entrenamiento por cada encoder."""
    fe = modelo.regressor_.named_steps["fe"]
    ordinal = fe.named_steps["codificar_ordinales"]
    categorias = {
        variable: set(cats)
        for variable, cats in zip(
            ordinal.variables_, ordinal.transformer_.categories_
        )
    }
    onehot = fe.named_steps["codificar_nominales"]
    categorias.update({
        variable: set(cats)
        for variable, cats in onehot.encoder_dict_.items()
    })
    return categorias


def _features_usadas(
    modelo: TransformedTargetRegressor, features: list[str]
) -> list[str]:
    """Excluye las features que el pipeline descarta por faltantes."""
    fe = modelo.regressor_.named_steps["fe"]
    descartadas = fe.named_steps["descartar_faltantes"].features_to_drop
    return [c for c in features if c not in descartadas]


def preparar_entrada(
    datos: pd.DataFrame | list[dict[str, Any]],
    modelo: TransformedTargetRegressor,
    features: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Index]:
    """Valida los datos de entrada y los deja con el formato del pipeline.

    Args:
        datos: Observaciones crudas, como DataFrame o lista de dicts con
            los nombres de columna de ``train.csv``. Puede incluir ``Id``
            como columna o como índice.
        modelo: Pipeline ajustado del modelo ganador.
        features: Features del modelo (``MODEL_FEATURES``). Las que el
            pipeline descarta por exceso de faltantes no son requeridas.
        logger: Logger donde registrar las advertencias.

    Returns:
        Tupla ``(X, ids)``: DataFrame con las columnas exactas con las que
        se entrenó el pipeline, e identificadores de cada fila.

    Raises:
        ErrorValidacionEntrada: Si la entrada está vacía, le faltan
            columnas requeridas o tiene valores no numéricos en columnas
            numéricas.
    """
    if isinstance(datos, pd.DataFrame):
        df = datos.copy()
    elif isinstance(datos, list) and all(isinstance(d, dict) for d in datos):
        df = pd.DataFrame(datos)
    else:
        raise ErrorValidacionEntrada(
            "La entrada debe ser un DataFrame o una lista de diccionarios."
        )

    if df.empty:
        raise ErrorValidacionEntrada("La entrada no contiene observaciones.")

    if COLUMNA_ID in df.columns:
        df = df.set_index(COLUMNA_ID)
    elif df.index.name != COLUMNA_ID:
        df.index = pd.RangeIndex(len(df), name=COLUMNA_ID)
    ids = df.index

    features = _features_usadas(modelo, features)
    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        raise ErrorValidacionEntrada(
            f"Faltan columnas requeridas: {faltantes}"
        )

    columnas_modelo = list(modelo.regressor_.feature_names_in_)
    desconocidas = [c for c in df.columns if c not in columnas_modelo]
    if desconocidas:
        logger.warning("Columnas desconocidas ignoradas: %s", desconocidas)

    for columna in _columnas_numericas(modelo):
        convertida = pd.to_numeric(df[columna], errors="coerce")
        invalidos = convertida.isna() & df[columna].notna()
        if invalidos.any():
            raise ErrorValidacionEntrada(
                f"Valores no numéricos en '{columna}' para los Id "
                f"{ids[invalidos.to_numpy()].tolist()}"
            )
        df[columna] = convertida

    valores_faltantes = df[features].isna().sum()
    valores_faltantes = valores_faltantes[valores_faltantes > 0]
    if not valores_faltantes.empty:
        logger.warning(
            "Valores faltantes que serán imputados: %s",
            valores_faltantes.to_dict(),
        )

    for variable, conocidas in _categorias_conocidas(modelo).items():
        no_vistas = set(df[variable].dropna()) - conocidas
        if no_vistas:
            logger.warning(
                "Categorías no vistas en entrenamiento en '%s': %s",
                variable,
                sorted(map(str, no_vistas)),
            )

    # Las columnas crudas no usadas por el modelo se completan con nulos:
    # el primer paso del pipeline las descarta.
    return df.reindex(columns=columnas_modelo), ids


def evaluar_coherencia(predicciones: np.ndarray) -> np.ndarray:
    """Indica qué predicciones son coherentes.

    Una predicción es coherente si es finita y está dentro del rango de
    ``SalePrice`` observado en ``train.csv``.

    Args:
        predicciones: Precios predichos en dólares.

    Returns:
        Arreglo booleano, ``True`` en las predicciones coherentes.
    """
    return (
        np.isfinite(predicciones)
        & (predicciones >= PRECIO_MINIMO)
        & (predicciones <= PRECIO_MAXIMO)
    )


def predecir(
    datos: pd.DataFrame | list[dict[str, Any]],
    modelo: TransformedTargetRegressor | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Genera predicciones de ``SalePrice`` a partir de datos crudos.

    Args:
        datos: Observaciones crudas, como DataFrame o lista de dicts con
            los nombres de columna de ``train.csv`` (``Id`` opcional).
        modelo: Pipeline del modelo ganador. Si es ``None`` se carga
            desde :data:`MODEL_PATH`.
        logger: Logger a utilizar. Si es ``None`` se configura el logger
            por defecto en :data:`LOG_PATH`.

    Returns:
        DataFrame con una fila por observación de entrada, en el mismo
        orden, y las columnas ``Id``, ``SalePrice`` (dólares; nulo si la
        predicción es incoherente) y ``es_coherente``.

    Raises:
        ErrorValidacionEntrada: Si la entrada tiene errores de
            estructura.
        RuntimeError: Si el número de predicciones no coincide con el
            número de observaciones de entrada.
    """
    if logger is None:
        logger = configurar_logger()
    inicio = time.perf_counter()
    logger.info(
        "Inicio de predicción | filas recibidas: %d",
        len(datos) if hasattr(datos, "__len__") else -1,
    )

    try:
        if modelo is None:
            modelo = cargar_modelo()
        X, ids = preparar_entrada(datos, modelo, cargar_features(), logger)
    except (ErrorValidacionEntrada, FileNotFoundError) as error:
        logger.error("Entrada rechazada: %s", error)
        raise

    try:
        predicciones = np.asarray(modelo.predict(X), dtype=float)
    except Exception:
        logger.exception("Error del modelo al generar las predicciones")
        raise

    if len(predicciones) != len(X):
        mensaje = (
            f"Se recibieron {len(X)} filas pero se generaron "
            f"{len(predicciones)} predicciones"
        )
        logger.error(mensaje)
        raise RuntimeError(mensaje)

    coherentes = evaluar_coherencia(predicciones)
    resultado = pd.DataFrame({
        COLUMNA_ID: ids,
        COLUMNA_PREDICCION: np.where(coherentes, predicciones, np.nan),
        COLUMNA_COHERENTE: coherentes,
    })

    n_incoherentes = int((~coherentes).sum())
    if n_incoherentes:
        logger.warning(
            "Predicciones incoherentes (fuera de [%.0f, %.0f] o no "
            "finitas) para los Id %s: %s",
            PRECIO_MINIMO,
            PRECIO_MAXIMO,
            ids[~coherentes].tolist(),
            predicciones[~coherentes].round(2).tolist(),
        )

    logger.info(
        "Fin de predicción | filas recibidas: %d | coherentes: %d | "
        "incoherentes: %d | duración: %.3f s",
        len(X),
        len(X) - n_incoherentes,
        n_incoherentes,
        time.perf_counter() - inicio,
    )
    return resultado


def muestrear_test(
    n: int,
    seed: int | None = None,
    test_path: Path = TEST_DATA_PATH,
) -> pd.DataFrame:
    """Toma observaciones aleatorias de los datos crudos de test.

    Args:
        n: Número de observaciones a muestrear.
        seed: Semilla para la reproducibilidad del muestreo.
        test_path: Ruta del archivo ``test.csv``.

    Returns:
        DataFrame con ``n`` filas crudas de test (``Id`` como columna).
    """
    test = pd.read_csv(test_path)
    return test.sample(n=n, random_state=seed).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta una predicción de prueba con observaciones aleatorias.

    Args:
        argv: Argumentos de línea de comandos (``None`` usa sys.argv).

    Returns:
        Código de salida del proceso (0 si todo fue correcto).
    """
    parser = argparse.ArgumentParser(
        description="Predice SalePrice para observaciones aleatorias de "
        "test.csv con el modelo ganador."
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Número de observaciones."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla del muestreo."
    )
    args = parser.parse_args(argv)

    logger = configurar_logger(LOG_PATH)
    logger.info(
        "Prueba por CLI con %d observaciones aleatorias (seed=%s)",
        args.n,
        args.seed,
    )
    resultado = predecir(muestrear_test(args.n, args.seed), logger=logger)
    print(resultado.to_string(index=False))
    print(f"\nLog: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
