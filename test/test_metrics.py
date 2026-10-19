import numpy as np
import pytest

from src.eval.metrics import evaluar, recall_a_fpr, umbral_optimo_f1

SEMILLAS = range(20)
LIMITES_FPR = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def datos_azar(semilla, n=200, prevalencia=0.10):
    """Etiquetas y puntuaciones al azar, siempre las mismas para una semilla dada."""
    rng = np.random.default_rng(semilla)
    y = (rng.random(n) < prevalencia).astype(int)
    p = rng.random(n)
    return y, p


def f1_a_mano(y, p, corte):
    """F1 marcando como ilícitas las transacciones con puntuación >= corte."""
    marcadas = p >= corte
    tp = np.sum(marcadas & (y == 1))
    fp = np.sum(marcadas & (y == 0))
    fn = np.sum(~marcadas & (y == 1))
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_umbral_optimo_es_el_mejor_corte(semilla):
    y, p = datos_azar(semilla)

    umbral = umbral_optimo_f1(y, p)
    f1_elegido = evaluar(y, p, umbral)["f1"]

    f1_mejor = max(f1_a_mano(y, p, corte) for corte in np.unique(p))

    assert f1_elegido == pytest.approx(f1_mejor)


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_matriz_y_metricas_son_coherentes(semilla):
    y, p = datos_azar(semilla)
    n_ilicitas = int(y.sum())
    n_licitas = len(y) - n_ilicitas

    res = evaluar(y, p, 0.5)
    tn, fp, fn, tp = res["tn"], res["fp"], res["fn"], res["tp"]

    assert tp + fn == n_ilicitas
    assert tn + fp == n_licitas
    assert res["precision"] == pytest.approx(tp / (tp + fp))
    assert res["recall"] == pytest.approx(tp / (tp + fn))

    prec, rec = res["precision"], res["recall"]
    assert res["f1"] == pytest.approx(2 * prec * rec / (prec + rec))

    for clave in ["f1", "precision", "recall", "auc_pr", "recall_fpr1"]:
        assert 0 <= res[clave] <= 1


@pytest.mark.parametrize("semilla", SEMILLAS)
def test_recall_a_fpr_nunca_baja(semilla):
    y, p = datos_azar(semilla)

    recalls = [recall_a_fpr(y, p, fpr_max=limite) for limite in LIMITES_FPR]

    for antes, despues in zip(recalls, recalls[1:]):
        assert despues >= antes
    assert recalls[-1] == pytest.approx(1.0)