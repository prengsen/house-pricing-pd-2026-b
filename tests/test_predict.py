"""Pruebas del script de inferencia con observaciones aleatorias de test.

Ejecutar desde la raíz del proyecto::

    pytest
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_pricing.modeling import predict

SEMILLA = 42
N_MUESTRA = 25


@pytest.fixture(scope="module")
def modelo():
    """Pipeline del modelo ganador (se carga una sola vez)."""
    return predict.cargar_modelo()


@pytest.fixture
def muestra() -> pd.DataFrame:
    """Observaciones aleatorias crudas de test.csv."""
    return predict.muestrear_test(N_MUESTRA, seed=SEMILLA)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    """Archivo de log temporal para no mezclar pruebas con producción."""
    return tmp_path / "predict_test.log"


@pytest.fixture
def logger(log_path: Path) -> logging.Logger:
    """Logger de inferencia apuntando al archivo temporal."""
    return predict.configurar_logger(log_path)


class ModeloExtremo:
    """Envuelve el modelo real pero devuelve predicciones incoherentes."""

    def __init__(self, modelo) -> None:
        self._modelo = modelo

    def __getattr__(self, nombre: str):
        return getattr(self._modelo, nombre)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        valores = np.full(len(X), 200_000.0)
        valores[0] = 1e9
        valores[1] = np.nan
        return valores


def test_prediccion_muestra_aleatoria(modelo, muestra, logger) -> None:
    resultado = predict.predecir(muestra, modelo=modelo, logger=logger)

    assert len(resultado) == len(muestra)
    assert list(resultado.columns) == ["Id", "SalePrice", "es_coherente"]
    assert resultado["Id"].tolist() == muestra["Id"].tolist()
    assert resultado["es_coherente"].all()
    assert resultado["SalePrice"].between(
        predict.PRECIO_MINIMO, predict.PRECIO_MAXIMO
    ).all()


def test_lista_de_dicts_equivale_a_dataframe(modelo, muestra, logger) -> None:
    desde_df = predict.predecir(muestra, modelo=modelo, logger=logger)
    desde_dicts = predict.predecir(
        muestra.to_dict("records"), modelo=modelo, logger=logger
    )

    pd.testing.assert_frame_equal(desde_df, desde_dicts)


def test_entrada_solo_con_features_requeridas(
    modelo, muestra, logger
) -> None:
    features = predict.cargar_features()
    reducida = muestra[["Id"] + features]

    resultado = predict.predecir(reducida, modelo=modelo, logger=logger)
    completo = predict.predecir(muestra, modelo=modelo, logger=logger)

    pd.testing.assert_frame_equal(resultado, completo)


def test_entrada_vacia_lanza_error(modelo, logger) -> None:
    with pytest.raises(predict.ErrorValidacionEntrada):
        predict.predecir([], modelo=modelo, logger=logger)


def test_columna_requerida_faltante_lanza_error(
    modelo, muestra, logger
) -> None:
    with pytest.raises(predict.ErrorValidacionEntrada, match="GrLivArea"):
        predict.predecir(
            muestra.drop(columns="GrLivArea"), modelo=modelo, logger=logger
        )


def test_valor_no_numerico_lanza_error(modelo, muestra, logger) -> None:
    muestra = muestra.astype({"GrLivArea": object})
    muestra.loc[0, "GrLivArea"] = "mil quinientos"

    with pytest.raises(predict.ErrorValidacionEntrada, match="GrLivArea"):
        predict.predecir(muestra, modelo=modelo, logger=logger)


def test_tipo_de_entrada_invalido_lanza_error(modelo, logger) -> None:
    with pytest.raises(predict.ErrorValidacionEntrada):
        predict.predecir("no es una tabla", modelo=modelo, logger=logger)


def test_predicciones_incoherentes_se_marcan(
    modelo, muestra, logger, log_path
) -> None:
    resultado = predict.predecir(
        muestra, modelo=ModeloExtremo(modelo), logger=logger
    )

    assert resultado["es_coherente"].tolist()[:2] == [False, False]
    assert resultado["SalePrice"].iloc[:2].isna().all()
    assert resultado["es_coherente"].iloc[2:].all()
    assert "incoherentes: 2" in log_path.read_text(encoding="utf-8")


def test_log_registra_conteo_de_filas(
    modelo, muestra, logger, log_path
) -> None:
    predict.predecir(muestra, modelo=modelo, logger=logger)
    contenido = log_path.read_text(encoding="utf-8")

    assert f"filas recibidas: {N_MUESTRA}" in contenido
    assert f"coherentes: {N_MUESTRA}" in contenido


def test_advertencias_de_calidad_en_log(
    modelo, muestra, logger, log_path
) -> None:
    muestra = muestra.copy()
    muestra.loc[0, "LotFrontage"] = np.nan
    muestra.loc[1, "Neighborhood"] = "BarrioNuevo"
    muestra["ColumnaExtra"] = 1

    resultado = predict.predecir(muestra, modelo=modelo, logger=logger)
    contenido = log_path.read_text(encoding="utf-8")

    assert len(resultado) == len(muestra)
    assert "Columnas desconocidas ignoradas" in contenido
    assert "Valores faltantes que serán imputados" in contenido
    assert "BarrioNuevo" in contenido


def test_cli_con_muestra_aleatoria(monkeypatch, capsys, log_path) -> None:
    monkeypatch.setattr(predict, "LOG_PATH", log_path)

    assert predict.main(["--n", "5", "--seed", str(SEMILLA)]) == 0
    assert "SalePrice" in capsys.readouterr().out
    assert "Prueba por CLI" in log_path.read_text(encoding="utf-8")
