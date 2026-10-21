"""
scripts/probar_grafo.py

Construye el grafo con src/data/amlworld.py y comprueba que cumple lo esperado.
No guarda nada: es solo una prueba.

Uso, desde la raíz del repositorio:

    python -m scripts.probar_grafo                      # ventana de desarrollo (días 5-7)
    python -m scripts.probar_grafo --inicio 0 --fin 0   # dataset completo (días 1-10)
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from src.data.amlworld import build_graph, vistas

ROOT = Path(__file__).resolve().parents[1]


def ok(msg):
    print(f"  OK  {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=ROOT / 'data' / 'raw' / 'HI-Small_Trans.csv')
    p.add_argument('--inicio', type=int, default=5, help='0 = sin ventana')
    p.add_argument('--fin', type=int, default=7, help='0 = sin ventana')
    args = p.parse_args()

    inicio = args.inicio or None
    fin = args.fin or None

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # ── Construcción ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    data, meta = build_graph(args.csv, truncar_dia=10,
                             ventana_inicio=inicio, ventana_fin=fin)
    t_build = time.perf_counter() - t0

    print()
    print(data)
    print(f"Tiempo de construcción: {t_build:.1f} s")
    print("Features de arista:", meta['features_arista'])
    print("Cortes:", meta['corte_val'], "|", meta['corte_test'])
    print()

    E = data.edge_index.size(1)
    s = data.edge_split

    # ── 1. Features de arista ─────────────────────────────────────────────────
    print("Comprobaciones:")
    assert data.edge_attr.size(1) == len(meta['features_arista']) == 1, \
        f"se esperaba 1 feature de arista, hay {data.edge_attr.size(1)}"
    ok("edge_attr tiene solo el importe")

    # ── 2. Atributos de arista e identificador ────────────────────────────────
    for clave in ('edge_attr', 'edge_label', 'edge_split', 'edge_id'):
        assert data.is_edge_attr(clave), f"'{clave}' no es atributo de arista"
    assert torch.equal(data.edge_id, torch.arange(E)), "edge_id no es 0..E-1"
    ok("edge_label, edge_split y edge_id son atributos de arista; edge_id = 0..E-1")

    # ── 3. Particiones ────────────────────────────────────────────────────────
    n = torch.bincount(s, minlength=3)
    assert (n > 0).all() and n.sum() == E
    assert meta['corte_val'] < meta['corte_test']
    # El split es cronológico: las particiones aparecen en orden 0, 1, 2.
    assert (s[1:] >= s[:-1]).all(), "edge_split no es monótono en el tiempo"
    ok(f"particiones cronológicas: {n.tolist()}")

    # ── 4. Escalado con estadísticos de entrenamiento ────────────────────────
    ea = data.edge_attr[s == 0].double()
    assert ea.mean(0).abs().max() < 1e-3 and (ea.std(0, unbiased=False) - 1).abs().max() < 1e-3, \
        "edge_attr de train no está estandarizado"
    nodos_train = torch.unique(data.edge_index[:, s == 0])
    xt = data.x[nodos_train].double()
    assert xt.mean(0).abs().max() < 1e-3 and (xt.std(0, unbiased=False) - 1).abs().max() < 1e-3, \
        "x de los nodos de train no está estandarizado"
    ok("edge_attr y x estandarizados con media 0 y desviación 1 en train")

    # ── 5. Vistas ─────────────────────────────────────────────────────────────
    v = vistas(data)
    permitido = {'train': 0, 'val': 1, 'test': 2}   # la vista ve split <= este valor
    for nombre, vista in v.items():
        g = vista['data']
        lim = permitido[nombre]

        # El grafo de la vista no lleva etiquetas ni partición.
        assert 'edge_label' not in g and 'edge_split' not in g, \
            f"{nombre}: la vista contiene etiquetas"

        # Conserva todos los nodos y solo las aristas permitidas.
        assert g.num_nodes == data.num_nodes
        assert g.edge_index.size(1) == int((s <= lim).sum())
        assert (s[g.edge_id] <= lim).all(), f"{nombre}: la vista ve aristas futuras"

        # Las semillas son exactamente las de su partición y están alineadas.
        ids = vista['edge_label_id']
        assert (s[ids] == lim).all()
        assert ids.numel() == int((s == lim).sum())
        assert torch.equal(vista['edge_label'], data.edge_label[ids])
        assert torch.equal(vista['edge_label_index'], data.edge_index[:, ids])

        # Toda semilla está dentro del grafo de la vista.
        assert torch.isin(ids, g.edge_id).all()
    ok("vistas sin etiquetas, sin aristas futuras y con semillas alineadas")

    # ── Resumen de prevalencias ───────────────────────────────────────────────
    print()
    for nombre, lim in permitido.items():
        y = v[nombre]['edge_label']
        print(f"{nombre:5s} {y.numel():9d} transacciones  {int(y.sum()):5d} ilícitas  "
              f"({100 * y.float().mean():.4f} %)")
    print("\nTodo correcto.")


if __name__ == '__main__':
    main()