"""Experimento "¿estructura o más datos?" (1): XGBoost con la misma información que la GNN.

La GNN, al puntuar val o test, ve en su grafo de mensajes las transacciones de esos
periodos. XGBoost (e2) solo ve perfiles calculados con train. Aquí se le dan a
XGBoost perfiles calculados con TODO lo que la GNN puede ver en cada vista:

    filas de train → perfiles con train            (igual que e2)
    filas de val   → perfiles con train + val
    filas de test  → perfiles con todo

Como las ventanas tienen distinta duración (unos 6, 8 y 10 días), los perfiles se
calculan POR DÍA: si no, en test todas las cuentas parecerían más activas solo
porque la ventana es más larga. Para XGBoost, dividir todas las filas de train por
el mismo número no cambia nada (sus árboles solo comparan órdenes).

Dos experimentos, mismo código:
    e2_dia       perfiles de train por día en todas las filas  → debe reproducir e2 (control)
    e2b_visibles perfiles de lo visible por día                → el experimento

Lectura: si e2b alcanza a la GNN, su ventaja venía de ver datos más recientes;
si la GNN sigue claramente por encima, venía de la estructura.

Uso:  python -m scripts.baseline_visibles --config configs/baseline_full.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.metrics import evaluar, imprimir_matriz, umbral_optimo_f1
from src.models.baseline import entrenar_xgboost, peso_clase, puntuacion

ROOT = Path(__file__).resolve().parents[1]
PERFIL = ["n_env", "vol_env", "n_rec", "vol_rec"]


def importes_usd(data, meta):
    """Deshace el escalado de edge_attr[:, 0]: estandarizado → log1p(usd) → usd."""
    log_usd = data.edge_attr[:, 0].double().numpy() * meta["std_e"][0] + meta["mu_e"][0]   # [E]
    return np.expm1(log_usd)                                                               # [E]


def perfiles(edge_index, usd, mascara, n_cuentas, dias=1.0):
    """Perfil de cada cuenta con las transacciones de `mascara`, por día.

    Mismo cálculo que features_nodo + escalar (sin estandarizar):
    log1p(nº envíos), log1p(importe enviado), log1p(nº cobros), log1p(importe cobrado).
    Devuelve [N, 4].
    """
    o, d, u = edge_index[0, mascara], edge_index[1, mascara], usd[mascara]
    n_env = np.bincount(o, minlength=n_cuentas)                  # [N] cuántas veces paga
    vol_env = np.bincount(o, weights=u, minlength=n_cuentas)     # [N] cuánto paga ($)
    n_rec = np.bincount(d, minlength=n_cuentas)                  # [N] cuántas veces cobra
    vol_rec = np.bincount(d, weights=u, minlength=n_cuentas)     # [N] cuánto cobra ($)
    crudo = np.column_stack([n_env, vol_env, n_rec, vol_rec]) / dias
    return np.log1p(crudo)                                       # [N, 4]


def duracion_vistas(cfg_grafo, meta):
    """Días que abarca cada vista (train; train+val; todo), a partir del CSV."""
    csv = Path(cfg_grafo["csv"])
    csv = csv if csv.is_absolute() else ROOT / csv
    ts = pd.to_datetime(pd.read_csv(csv, usecols=["Timestamp"])["Timestamp"], format="%Y/%m/%d %H:%M")
    dia0 = ts.min().normalize()
    inicio = meta.get("ventana_inicio") or 1
    fin = meta.get("ventana_fin") or meta["truncar_dia"]
    t_ini = dia0 + pd.Timedelta(days=inicio - 1)
    t_fin = dia0 + pd.Timedelta(days=fin)
    un_dia = pd.Timedelta(days=1)
    return {
        "train": (meta["corte_val"] - t_ini) / un_dia,
        "val": (meta["corte_test"] - t_ini) / un_dia,
        "test": (t_fin - t_ini) / un_dia,
    }


def tabla(data, P_por_fila):
    """[E, 9]: importe | perfil del origen | perfil del destino (como cargar_features)."""
    ei = data.edge_index.numpy()
    filas = np.arange(ei.shape[1])
    importe = data.edge_attr[:, :1].numpy()                       # [E, 1]
    P_o = P_por_fila(filas, ei[0])                                 # [E, 4]
    P_d = P_por_fila(filas, ei[1])                                 # [E, 4]
    return np.hstack([importe, P_o, P_d]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="el YAML del baseline (grafo, semillas, xgb)")
    ap.add_argument("--salida", default="results/baseline/full_visibles")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    salida = ROOT / args.salida
    salida.mkdir(parents=True, exist_ok=True)

    obj = torch.load(ROOT / cfg["grafo"], weights_only=False)
    data, meta = obj["data"], obj["meta"]
    ei = data.edge_index.numpy()                                   # [2, E]
    split = data.edge_split.numpy()                                # [E] 0/1/2
    y = data.edge_label.numpy()                                    # [E]
    N = data.x.size(0)
    usd = importes_usd(data, meta)

    # --- Comprobación: con train y sin dividir por días, tiene que salir data.x ---
    P_train_1 = perfiles(ei, usd, split == 0, N, dias=1.0)
    x_rehecho = (P_train_1 - meta["mu_x"]) / meta["std_x"]
    dif = np.abs(x_rehecho - data.x.numpy()).max()
    assert dif < 1e-3, f"los perfiles rehechos no coinciden con data.x (diferencia máx. {dif})"
    print(f"OK  perfiles de train rehechos = data.x (diferencia máx. {dif:.1e})")

    # --- Perfiles por día de cada vista ---
    dias = duracion_vistas(obj["config"], meta)
    print("Días por vista: " + ", ".join(f"{k} {v:.2f}" for k, v in dias.items()))
    P = {
        0: perfiles(ei, usd, split == 0, N, dias["train"]),       # lo que se ve al puntuar train
        1: perfiles(ei, usd, split <= 1, N, dias["val"]),         # al puntuar val
        2: perfiles(ei, usd, split <= 2, N, dias["test"]),        # al puntuar test
    }

    def solo_train(filas, cuentas):
        return P[0][cuentas]

    def lo_visible(filas, cuentas):
        out = P[0][cuentas].copy()
        for s in (1, 2):
            m = split[filas] == s
            out[m] = P[s][cuentas[m]]
        return out

    experimentos = {"e2_dia": solo_train, "e2b_visibles": lo_visible}
    nombres = ["log_importe_usd"] + [f"origen_{p}" for p in PERFIL] + [f"destino_{p}" for p in PERFIL]
    tr, va, te = split == 0, split == 1, split == 2
    peso = peso_clase(y[tr])
    print(f"train {int(tr.sum())} | val {int(va.sum())} | test {int(te.sum())} | peso {peso:.1f}\n")

    for nombre, P_por_fila in experimentos.items():
        X = tabla(data, P_por_fila)
        for semilla in cfg["semillas"]:
            modelo, t_ent = entrenar_xgboost(X[tr], y[tr], X[va], y[va], cfg["xgb"], peso, semilla)
            p_va, _ = puntuacion(modelo, X[va])
            p_te, t_pun = puntuacion(modelo, X[te])
            umbral = umbral_optimo_f1(y[va], p_va)
            res = evaluar(y[te], p_te, umbral)
            res.update({
                "experimento": nombre,
                "semilla": semilla,
                "columnas": nombres,
                "peso": peso,
                "best_iteration": int(modelo.best_iteration),
                "t_entrenar": t_ent,
                "t_puntuar_test": t_pun,
                "dias_vistas": dias,
                "params": cfg["xgb"],
                "grafo": cfg["grafo"],
            })
            base = f"{nombre}_semilla{semilla}"
            with open(salida / f"{base}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            np.savez(salida / f"{base}.npz", p_val=p_va, p_test=p_te)

            print(f"{nombre:13s} semilla {semilla} | F1 {res['f1']:.3f} | AUC-PR {res['auc_pr']:.4f} | "
                  f"recall@1% {res['recall_fpr1']:.3f} | árboles {modelo.best_iteration} | {t_ent:.1f} s")
            imprimir_matriz(res)


if __name__ == "__main__":
    main()