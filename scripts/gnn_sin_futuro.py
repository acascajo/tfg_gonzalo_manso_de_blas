"""Experimento "¿estructura o más datos?" (2): ¿cuánto depende la GNN de ver el futuro?

Con el protocolo de las tres vistas, al puntuar una transacción de test la GNN ve
todas las transacciones del periodo de test, también las POSTERIORES a ella.
Aquí se vuelven a puntuar val y test con los modelos ya entrenados, pero en
bloques de H horas: las transacciones de cada bloque se puntúan con un grafo que
solo contiene lo ocurrido hasta el final de ese bloque.

    H = 0   sin corte: el protocolo de siempre (sirve de control: debe dar lo mismo)
    H = 24  revisión diaria: al final de cada día, con todo lo ocurrido hasta entonces
    H = 1   casi en tiempo real: como mucho, una hora de "futuro"

No se reentrena nada: se usan los pesos de results/gnn/full/gnn_s*.pt.

Lectura: si la AUC-PR apenas baja con H = 1, la GNN no necesita saber lo que pasa
después. Si baja mucho, parte de su ventaja venía de ver el futuro (o de que se
entrenó viendo el futuro: con este experimento no se pueden separar esas dos cosas).

Uso:  python -m scripts.gnn_sin_futuro --config configs/gnn_full.yaml --horas 0 24 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.amlworld import cargar, con_inversas, truncar, ventana_desarrollo
from src.eval.metrics import evaluar, imprimir_matriz, umbral_optimo_f1
from src.models.gnn import GINEAristas, cargador, puntuar

ROOT = Path(__file__).resolve().parents[1]


def minutos_por_arista(obj):
    """Minuto (desde el inicio del primer día) de cada transacción, en el orden del grafo.

    Repite los mismos pasos que build_graph (cargar → truncar → ventana), así que la
    fila i del DataFrame es la arista i. Se comprueba con las etiquetas.
    """
    cfg, data = obj["config"], obj["data"]
    csv = Path(cfg["csv"])
    csv = csv if csv.is_absolute() else ROOT / csv
    df = cargar(csv)
    df = truncar(df, dia=cfg["truncar_dia"])
    ventana = cfg.get("ventana") or {}
    df = ventana_desarrollo(df, ventana.get("inicio"), ventana.get("fin")).reset_index(drop=True)

    assert len(df) == data.edge_index.size(1), "el CSV procesado no tiene tantas filas como aristas"
    assert (df["Is Laundering"].to_numpy() == data.edge_label.numpy()).all(), \
        "las etiquetas del CSV no coinciden con las del grafo: el orden no cuadra"

    dia0 = df["Timestamp"].min().normalize()
    return ((df["Timestamp"] - dia0).dt.total_seconds() // 60).to_numpy().astype(np.int64)   # [E]


def vista_hasta(data, mascara_grafo, mascara_semillas):
    """Como una entrada de vistas(), pero con el grafo que se quiera (siempre con inversas)."""
    grafo = data.edge_subgraph(torch.from_numpy(mascara_grafo))
    del grafo.edge_label, grafo.edge_split
    grafo = con_inversas(grafo).contiguous()
    m = torch.from_numpy(mascara_semillas)
    return {
        "data": grafo,
        "edge_label_index": data.edge_index[:, m].contiguous(),
        "edge_label": data.edge_label[m],
        "edge_label_id": data.edge_id[m],
    }


def puntuar_en_bloques(modelo, data, minutos, split, trozo, horas, cfg, device, semilla):
    """Probabilidad de cada semilla de `trozo` (1 = val, 2 = test), en el orden del grafo.

    horas = 0: un solo bloque con todo lo visible (protocolo de siempre).
    horas > 0: bloques de `horas`; cada bloque ve solo hasta su final.
    """
    c = cfg["cargador"]
    semillas = split == trozo                                  # [E]
    p = np.full(split.shape[0], np.nan)                        # [E] solo se rellenan las semillas
    visibles = split <= trozo                                  # lo que ve esta vista

    if horas == 0:
        bloques = [(semillas, visibles)]
    else:
        ancho = horas * 60
        bloque = minutos // ancho                              # [E] número de bloque de cada arista
        bloques = []
        for b in np.unique(bloque[semillas]):
            fin = (b + 1) * ancho
            bloques.append((semillas & (bloque == b), visibles & (minutos < fin)))

    for m_sem, m_grafo in bloques:
        vista = vista_hasta(data, m_grafo, m_sem)
        loader = cargador(vista, c["num_vecinos"], c["lote_eval"], shuffle=False)
        torch.manual_seed(semilla)
        p_bloque, _ = puntuar(modelo, loader, vista, data, True, device)
        p[np.flatnonzero(m_sem)] = p_bloque

    assert not np.isnan(p[semillas]).any(), "hay semillas sin puntuar"
    return p[semillas], len(bloques)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="el YAML con el que se entrenó la GNN")
    ap.add_argument("--horas", type=int, nargs="+", default=[0, 24, 1])
    ap.add_argument("--salida", default="results/gnn/full_sin_futuro")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["inversas"], "este script asume el modelo entrenado con inversas"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    salida = ROOT / args.salida
    salida.mkdir(parents=True, exist_ok=True)

    obj = torch.load(ROOT / cfg["grafo"], weights_only=False)
    data = obj["data"]
    minutos = minutos_por_arista(obj)
    split = data.edge_split.numpy()
    y = data.edge_label.numpy()
    print(f"OK  tiempos alineados con el grafo ({len(minutos)} aristas) | dispositivo {device}\n")

    for semilla in cfg["semillas"]:
        modelo = GINEAristas(data.x.size(1), data.edge_attr.size(1) + 1, **cfg["modelo"]).to(device)
        pesos = ROOT / cfg["salida"] / f"gnn_s{semilla}.pt"
        modelo.load_state_dict(torch.load(pesos, map_location=device))

        for horas in args.horas:
            t0 = time.perf_counter()
            p_va, n_va = puntuar_en_bloques(modelo, data, minutos, split, 1, horas, cfg, device, semilla)
            p_te, n_te = puntuar_en_bloques(modelo, data, minutos, split, 2, horas, cfg, device, semilla)
            umbral = umbral_optimo_f1(y[split == 1], p_va)
            res = evaluar(y[split == 2], p_te, umbral)
            nombre = "gnn_sin_corte" if horas == 0 else f"gnn_bloques_{horas}h"
            res.update({
                "experimento": nombre,
                "semilla": semilla,
                "horas": horas,
                "bloques_val": n_va,
                "bloques_test": n_te,
                "pesos": str(pesos.relative_to(ROOT)),
                "t_total": time.perf_counter() - t0,
            })
            with open(salida / f"{nombre}_s{semilla}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

            print(f"{nombre:16s} semilla {semilla} | AUC-PR {res['auc_pr']:.4f} | F1 {res['f1']:.3f} | "
                  f"recall@1% {res['recall_fpr1']:.3f} | bloques {n_va}+{n_te} | {res['t_total']:.0f} s")
        print()


if __name__ == "__main__":
    main()