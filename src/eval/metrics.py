import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)


def umbral_optimo_f1(y_val, p_val):
    """Umbral que maximiza el F1 en VALIDACIÓN. Después se aplica tal cual en test."""
    precision, recall, umbrales = precision_recall_curve(y_val, p_val)
    # precision y recall tienen un valor más que umbrales: el último se descarta
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(umbrales[np.argmax(f1)])


def recall_a_fpr(y, p, fpr_max=0.01):
    """Qué fracción del blanqueo se detecta marcando como mucho el 1 % de las lícitas."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= fpr_max].max())


def evaluar(y, p, umbral):
    """Todas las métricas de un conjunto, con un umbral ya elegido."""
    y, p = np.asarray(y), np.asarray(p)          # por si llegan como listas
    pred = (p >= umbral).astype(int)

    # Filas = realidad, columnas = predicción  →  [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        'f1':          float(f1_score(y, pred, zero_division=0)),
        'precision':   float(precision_score(y, pred, zero_division=0)),
        'recall':      float(recall_score(y, pred, zero_division=0)),
        'auc_pr':      float(average_precision_score(y, p)),
        'recall_fpr1': recall_a_fpr(y, p, 0.01),
        'umbral':      float(umbral),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def formato_es(texto):
    return texto.translate(str.maketrans(',.', '.,'))


def _pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "sin datos"


def imprimir_matriz(res):
    """Muestra por pantalla la matriz de confusión de un resultado de evaluar()."""
    tn, fp, fn, tp = res['tn'], res['fp'], res['fn'], res['tp']
    lineas = [
        f"Matriz de confusión (umbral {res['umbral']:.3f})",
        "",
        f"{'':18}{'Predicho':^24}",
        f"{'':18}{'Lícita':>12}{'Blanqueo':>12}",
        f"{'Real   Lícita':18}{tn:>12,}{fp:>12,}",
        f"{'       Blanqueo':18}{fn:>12,}{tp:>12,}",
        "",
        f"Falsas alarmas:      {fp:,} de {tn + fp:,} lícitas ({_pct(fp, tn + fp)})",
        f"Blanqueo detectado:  {tp:,} de {fn + tp:,} ilícitas ({_pct(tp, fn + tp)})",
        f"Alertas correctas:   {tp:,} de {tp + fp:,} alertas ({_pct(tp, tp + fp)})",
    ]
    print(formato_es("\n".join(lineas)))
