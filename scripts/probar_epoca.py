"""Prueba rápida de la GNN en el grafo de desarrollo: ¿entrena sin errores?

No es un experimento. Responde a: ¿baja la pérdida?, ¿cuánto tarda una época?,
¿cuánta memoria de GPU usa? Para entrenar de verdad: scripts/entrenar_gnn.py.

Uso:  python -m scripts.probar_epoca [--epocas 3]
"""
import argparse
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score

from src.data.amlworld import vistas
from src.models.gnn import GINEAristas, cargador, entrenar_epoca, puntuar

ROOT = Path(__file__).resolve().parents[1]
GRAFO = ROOT / "data" / "processed" / "amlworld_dev.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    obj = torch.load(GRAFO, weights_only=False)
    data = obj["data"]
    vs = vistas(data, inversas=True)
    tr, va = vs["train"], vs["val"]

    loader_tr = cargador(tr, [100, 100], 2048, shuffle=True)
    loader_va = cargador(va, [100, 100], 4096, shuffle=False)
    print(f"Lotes por época: train {len(loader_tr)} | val {len(loader_va)}")

    n_x = data.x.size(1)                        # 4: perfil de la cuenta
    n_e = tr["data"].edge_attr.size(1)          # 2: importe + sentido
    modelo = GINEAristas(n_x, n_e, oculto=64, capas=2).to(device)
    n_param = sum(p.numel() for p in modelo.parameters())
    print(f"Modelo: n_x={n_x}, n_e={n_e}, {n_param} parámetros")

    perdida = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.0, device=device))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.006)

    y_va = va["edge_label"].numpy()             # [S_val] etiquetas de validación
    print(f"Val: {len(y_va)} semillas, {int(y_va.sum())} ilícitas | AUC-PR del azar {y_va.mean():.4f}\n")

    for epoca in range(1, args.epocas + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss, t_tr = entrenar_epoca(modelo, loader_tr, tr, data, True, optimizador, perdida, device)
        p_va, t_va = puntuar(modelo, loader_va, va, data, True, device)     # [S_val] probabilidades
        auc_pr = average_precision_score(y_va, p_va)
        vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        print(f"Época {epoca} | pérdida {loss:.4f} | AUC-PR val {auc_pr:.4f} (×{auc_pr / y_va.mean():.1f}) | "
              f"train {t_tr:.1f} s | val {t_va:.1f} s | VRAM máx. {vram:.0f} MB")


if __name__ == "__main__":
    main()