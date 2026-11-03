"""Tabla final de resultados: media ± desviación de cada experimento, leída de los JSON.

La desviación es la muestral (n - 1), la misma para XGBoost y para la GNN.

Uso:
    python -m scripts.tabla_resultados
    python -m scripts.tabla_resultados results/baseline/full results/gnn/full

Además de imprimirla, la guarda en results/tabla_resultados.md para la memoria.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARPETAS = ["results/baseline/full", "results/baseline/full_controles", "results/gnn/full"]
METRICAS = [("auc_pr", "AUC-PR", 4), ("f1", "F1", 3), ("recall_fpr1", "Recall al 1 %", 3)]
PREV_TEST = 1143 / 1015418          # blanqueo en test del grafo completo (para el "× azar")


def es(x, dec):
    """Número con coma decimal, como en la memoria."""
    return f"{x:.{dec}f}".replace(".", ",")


def cargar(carpetas):
    """{(carpeta, experimento): [json de cada semilla]}"""
    grupos = defaultdict(list)
    for carpeta in carpetas:
        for ruta in sorted((ROOT / carpeta).glob("*.json")):
            with open(ruta, encoding="utf-8") as f:
                res = json.load(f)
            grupos[(carpeta, res["experimento"])].append(res)
    return grupos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpetas", nargs="*", default=CARPETAS)
    args = ap.parse_args()

    grupos = cargar(args.carpetas)
    if not grupos:
        raise SystemExit("No hay JSON en esas carpetas")

    cab = ["Experimento", "Carpeta", "Semillas"] + [n for _, n, _ in METRICAS] + ["× azar"]
    filas = []
    for (carpeta, exp), lista in grupos.items():
        fila = [exp, carpeta.replace("results/", ""), str(len(lista))]
        for clave, _, dec in METRICAS:
            v = np.array([r[clave] for r in lista], dtype=float)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            fila.append(f"{es(v.mean(), dec)} ± {es(sd, dec)}")
        auc = np.mean([r["auc_pr"] for r in lista])
        fila.append(f"×{auc / PREV_TEST:.0f}")
        filas.append(fila)

    md = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    md += ["| " + " | ".join(f) + " |" for f in filas]
    texto = "\n".join(md)
    print(texto)

    salida = ROOT / "results" / "tabla_resultados.md"
    salida.write_text(texto + "\n\nDesviación muestral (n − 1). Test del grafo completo.\n", encoding="utf-8")
    print(f"\nGuardada en {salida.relative_to(ROOT)}")


if __name__ == "__main__":
    main()