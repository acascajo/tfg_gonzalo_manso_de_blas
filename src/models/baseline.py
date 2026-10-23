import numpy as np
import time
import xgboost as xgb



def peso_clase(y_train): 
    """
    Función para el calculo del peso de la clase positiva sobre la negativa y asi poder balancear
    """

    licitas = np.sum(y_train == 0)
    ilicitas = np.sum(y_train == 1)

    return float(licitas/ilicitas)

def col_to_index(nombres, columnas):
    """
    Función que más tarde nos permite filtrar por filas y quedarnos con columnas en concreto
    """
    return [nombres.index(c) for c in columnas]


def entrenar_xgboost(X_tr, y_tr, X_v, y_v, params, peso, semilla):
    modelo = xgb.XGBClassifier(**params, scale_pos_weight = peso, random_state = semilla)

    t0 = time.perf_counter()

    modelo.fit(X_tr, y_tr, eval_set = [(X_v, y_v)], verbose = False)

    t1 = time.perf_counter() - t0


    return modelo, t1

def puntuacion(modelo, X):
    t0 = time.perf_counter()

    p = modelo.predict_proba(X)[:, 1]

    t1 = time.perf_counter() - t0


    return p, t1


