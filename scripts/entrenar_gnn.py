"""Entrena la GNN con varias semillas y guarda los resultados como el baseline.

Uso:
    python -m scripts.entrenar_gnn --config configs/gnn_dev.yaml

Por cada semilla, en la carpeta `salida` del YAML:
    gnn_s{semilla}.json  nota en test, umbral, historial de épocas, tiempos y configuración
    gnn_s{semilla}.npz   p_val, p_test, ids_val, ids_test (mismo orden que los .npz del baseline)
    gnn_s{semilla}.pt    pesos de la mejor época (según la AUC-PR de validación)

Protocolo (igual que XGBoost): se entrena en train; validación decide la mejor
época y el umbral; test solo se mira al final, una vez.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from src.data.amlworld import vistas
from src.eval.metrics import evaluar, imprimir_matriz, umbral_optimo_f1
from src.models.gnn import GINEAristas, cargador, entrenar_epoca, puntuar

ROOT = Path(__file__).resolve().parents[1]


def fijar_semilla(semilla):
    """Misma semilla en Python, NumPy y PyTorch (CPU y GPU)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def entrenar_una_semilla(cfg, grafo_path, data, vs, semilla, device, salida):
    fijar_semilla(semilla)
    inv, c, m, t = cfg["inversas"], cfg["cargador"], cfg["modelo"], cfg["entrenamiento"]
    tr, va, te = vs["train"], vs["val"], vs["test"]

    loader_tr = cargador(tr, c["num_vecinos"], c["lote_train"], shuffle=True)
    loader_va = cargador(va, c["num_vecinos"], c["lote_eval"], shuffle=False)
    loader_te = cargador(te, c["num_vecinos"], c["lote_eval"], shuffle=False)

    n_x = data.x.size(1)                        # 4 (perfil de la cuenta)
    n_e = tr["data"].edge_attr.size(1)          # 2 con inversas (importe, sentido)
    modelo = GINEAristas(n_x, n_e, **m).to(device)
    n_param = sum(p.numel() for p in modelo.parameters())
    perdida = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(t["pos_weight"], device=device))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=t["lr"])

    y_va = va["edge_label"].numpy()             # [S_val]
    y_te = te["edge_label"].numpy()             # [S_test]
    base = f"gnn_s{semilla}"
    ruta_pesos = salida / f"{base}.pt"

    # --- Épocas: entrenar, mirar validación, quedarse con la mejor ---------------
    historial, mejor, mejor_epoca, sin_mejora = [], -1.0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_inicio = time.perf_counter()

    for epoca in range(1, t["epocas"] + 1):
        loss, t_tr = entrenar_epoca(modelo, loader_tr, tr, data, inv, optimizador, perdida, device)
        p_va, t_va = puntuar(modelo, loader_va, va, data, inv, device)
        auc_pr = float(average_precision_score(y_va, p_va))
        historial.append({"epoca": epoca, "perdida": loss, "auc_pr_val": auc_pr,
                          "t_train": t_tr, "t_val": t_va})

        marca = ""
        if auc_pr > mejor:
            mejor, mejor_epoca, sin_mejora = auc_pr, epoca, 0
            torch.save(modelo.state_dict(), ruta_pesos)       # guarda la mejor hasta ahora
            marca = "  <- mejor"
        else:
            sin_mejora += 1
        print(f"  época {epoca:2d} | pérdida {loss:.4f} | AUC-PR val {auc_pr:.4f} "
              f"(×{auc_pr / y_va.mean():.1f}) | {t_tr:.0f} s + {t_va:.0f} s{marca}", flush=True)

        if sin_mejora >= t["paciencia"]:
            print(f"  parada temprana: {t['paciencia']} épocas seguidas sin mejorar")
            break
    t_total = time.perf_counter() - t_inicio

    # --- Nota final: mejor época, umbral en validación, nota en test -------------
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    torch.manual_seed(semilla)                  # mismo muestreo al puntuar
    p_va, _ = puntuar(modelo, loader_va, va, data, inv, device)
    p_te, t_te = puntuar(modelo, loader_te, te, data, inv, device)

    umbral = umbral_optimo_f1(y_va, p_va)
    res = evaluar(y_te, p_te, umbral)
    vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    res.update({
        "experimento": "gnn",
        "semilla": semilla,
        "mejor_epoca": mejor_epoca,
        "auc_pr_val_mejor": mejor,
        "historial": historial,
        "t_entrenar_total": t_total,
        "t_puntuar_test": t_te,
        "vram_max_mb": vram,
        "n_parametros": n_param,
        "dispositivo": str(device),
        "grafo": str(grafo_path),
        "config": cfg,
    })

    with open(salida / f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez(salida / f"{base}.npz", p_val=p_va, p_test=p_te,
             ids_val=va["edge_label_id"].numpy(), ids_test=te["edge_label_id"].numpy())

    print(f"gnn semilla {semilla} | F1 {res['f1']:.3f} | AUC-PR {res['auc_pr']:.4f} | "
          f"recall@1% {res['recall_fpr1']:.3f} | mejor época {mejor_epoca} | {t_total / 60:.1f} min")
    imprimir_matriz(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grafo_path = ROOT / cfg["grafo"]
    salida = ROOT / cfg["salida"]
    salida.mkdir(parents=True, exist_ok=True)

    obj = torch.load(grafo_path, weights_only=False)
    data = obj["data"]
    vs = vistas(data, inversas=cfg["inversas"])

    for nombre in ("train", "val", "test"):
        y = vs[nombre]["edge_label"]
        print(f"{nombre}: {y.numel()} semillas, {int(y.sum())} ilícitas")
    print(f"Dispositivo: {device}\n")

    for semilla in cfg["semillas"]:
        print(f"--- semilla {semilla} ---")
        entrenar_una_semilla(cfg, grafo_path, data, vs, semilla, device, salida)
        print()


if __name__ == "__main__":
    main()