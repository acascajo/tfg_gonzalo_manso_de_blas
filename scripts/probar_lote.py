import time
from pathlib import Path

import torch
from torch_geometric.loader import LinkNeighborLoader

from src.data.amlworld import vistas

ROOT = Path(__file__).resolve().parents[1]
GRAFO = ROOT / "data" / "processed" / "amlworld_dev.pt"


def main():
    obj = torch.load(GRAFO, weights_only=False)
    data = obj["data"]
    v = vistas(data, inversas=True)["train"]

    # Comprobaciones de las inversas
    g = v["data"]
    E = g.edge_index.size(1) // 2
    assert E == int((data.edge_split == 0).sum()), "no hay el doble de aristas"
    assert torch.equal(g.edge_index[:, E:], g.edge_index[:, :E].flip(0)), "las inversas no unen las mismas cuentas al revés"
    assert torch.equal(g.edge_id[E:], g.edge_id[:E]), "las dos copias no tienen el mismo edge_id"
    assert torch.equal(g.edge_attr[E:, :-1], g.edge_attr[:E, :-1]), "las features de las copias no coinciden"
    assert (g.edge_attr[:E, -1] == 0).all() and (g.edge_attr[E:, -1] == 1).all(), "la columna de sentido está mal"
    print(f"OK  inversas: {E} originales + {E} al revés")

    loader = LinkNeighborLoader(
        v["data"],                            # el grafo de la vista, sin etiquetas
        num_neighbors=[100, 100],             # 100 vecinos en cada uno de los 2 saltos
        edge_label_index=v["edge_label_index"],
        edge_label=v["edge_label"],
        batch_size=2048,
        shuffle=True,
    )

    t0 = time.perf_counter()
    batch = next(iter(loader))
    print(f"Primer lote en {time.perf_counter() - t0:.2f} s\n")
    print(batch, "\n")

    # De la semilla del lote a su transacción en el grafo completo
    ids = v["edge_label_id"][batch.input_id]
    print(f"Semillas en el lote: {ids.numel()} | primeras: {ids[:5].tolist()}")

    # Comprobación 1: las dos cuentas de cada semilla son las de su transacción
    cuentas_lote = batch.n_id[batch.edge_label_index]
    cuentas_grafo = data.edge_index[:, ids]
    assert torch.equal(cuentas_lote, cuentas_grafo), "las cuentas de las semillas no coinciden"

    # Comprobación 2: las etiquetas son las de su transacción
    assert torch.equal(batch.edge_label.long(), data.edge_label[ids].long()), "etiquetas desalineadas"
    print("OK  cuentas y etiquetas de las semillas coinciden con el grafo")

    dentro = torch.isin(ids, batch.edge_id).float().mean()
    print(f"Semillas que aparecen entre las aristas del subgrafo: {100 * dentro:.1f} %")
    print(f"Ilícitas en el lote: {int(batch.edge_label.sum())}")


if __name__ == "__main__":
    main()