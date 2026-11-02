"""GNN para clasificar transacciones (aristas) con GINEConv.

Idea: cada cuenta resume en una ficha de `oculto` números lo que pasa a su
alrededor (2 capas = hasta 2 saltos), y la cabeza decide sobre cada transacción
semilla mirando a la vez [ficha del origen, ficha del destino, ficha de la
transacción].

Basado en la clase GINe de Multi-GNN (IBM, models.py), con estos cambios:
una sola salida + BCEWithLogitsLoss, solo se puntúan las semillas, y sus
features salen del grafo completo por edge_id.

Notación de formas en los comentarios:
    N = cuentas del lote · M = transacciones del subgrafo del lote
    S = semillas del lote · H = oculto (64) · n_x = 4 · n_e = 2 (importe, sentido)
"""
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GINEConv


class GINEAristas(nn.Module):
    def __init__(self, n_x, n_e, oculto=64, capas=2, dropout=0.01, final_dropout=0.1):
        super().__init__()
        self.dropout = dropout

        # Paso 0: pasar perfiles y transacciones a fichas del mismo tamaño
        self.emb_x = nn.Linear(n_x, oculto)          # [N, 4] -> [N, H]
        self.emb_e = nn.Linear(n_e, oculto)          # [M, 2] -> [M, H]

        # Una capa de "escuchar a los vecinos" por salto
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(capas):
            mlp = nn.Sequential(nn.Linear(oculto, oculto), nn.ReLU(), nn.Linear(oculto, oculto))
            self.convs.append(GINEConv(mlp, edge_dim=oculto))   # suma mensajes (vecino + transacción) y pasa por mlp
            self.norms.append(nn.BatchNorm1d(oculto))           # mantiene los números en una escala estable

        # Cabeza: [ficha origen | ficha destino | ficha transacción] -> 1 número
        self.cabeza = nn.Sequential(
            nn.Linear(3 * oculto, oculto), nn.ReLU(), nn.Dropout(final_dropout),     # [S, 3H] -> [S, H]
            nn.Linear(oculto, oculto // 2), nn.ReLU(), nn.Dropout(final_dropout),    # [S, H]  -> [S, H/2]
            nn.Linear(oculto // 2, 1),                                               # [S, H/2] -> [S, 1]
        )

    def forward(self, x, edge_index, edge_attr, semillas, attr_semillas):
        # x [N, n_x] · edge_index [2, M] · edge_attr [M, n_e]
        # semillas [2, S] (números de cuenta dentro del lote) · attr_semillas [S, n_e]
        h = self.emb_x(x)                            # [N, H]  ficha inicial de cada cuenta
        e = self.emb_e(edge_attr)                    # [M, H]  ficha de cada transacción

        for conv, norm in zip(self.convs, self.norms):
            nuevo = F.relu(norm(conv(h, edge_index, e)))         # [N, H]  lo que oye de sus vecinos
            h = (h + nuevo) / 2                                  # [N, H]  mitad lo que sabía, mitad lo nuevo
            h = F.dropout(h, p=self.dropout, training=self.training)

        z = torch.cat([h[semillas[0]],               # [S, H]  ficha del origen de cada semilla
                       h[semillas[1]],               # [S, H]  ficha del destino
                       self.emb_e(attr_semillas)],   # [S, H]  ficha de la propia transacción
                      dim=1)                         # [S, 3H]
        return self.cabeza(z).squeeze(-1)            # [S]     un logit por semilla


def cargador(vista, num_vecinos, batch_size, shuffle):
    """LinkNeighborLoader de una vista: lotes de semillas con su trocito de grafo."""
    return LinkNeighborLoader(
        vista["data"],                               # grafo de mensajes (sin etiquetas)
        num_neighbors=num_vecinos,                   # p. ej. [100, 100]: vecinos por salto
        edge_label_index=vista["edge_label_index"],  # [2, S_total] semillas
        edge_label=vista["edge_label"],              # [S_total]    sus etiquetas
        batch_size=batch_size,
        shuffle=shuffle,
    )


def attr_semillas(data, ids, inversas):
    """Features de las semillas, sacadas del grafo completo por su id.

    Así se puntúan todas aunque el muestreo no coja su transacción. Con inversas,
    se añade sentido = 0, porque la semilla es siempre la transacción original.
    """
    a = data.edge_attr[ids]                                              # [S, n_e - 1] importe
    if inversas:
        a = torch.cat([a, torch.zeros(a.size(0), 1, dtype=a.dtype)], dim=1)   # [S, n_e]  importe, 0
    return a


def logits_lote(modelo, batch, vista, data, inversas, device):
    """Pasa un lote por el modelo. Devuelve logits [S], etiquetas [S] e ids [S]."""
    ids = vista["edge_label_id"][batch.input_id]     # [S]  id de cada semilla en el grafo completo
    a = attr_semillas(data, ids, inversas).to(device)
    batch = batch.to(device)
    logits = modelo(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, a)
    return logits, batch.edge_label.float(), ids


def entrenar_epoca(modelo, loader, vista, data, inversas, optimizador, perdida, device):
    """Una pasada por todos los lotes de train. Devuelve (pérdida media, segundos)."""
    modelo.train()                                   # activa dropout y BatchNorm en modo entrenamiento
    total, n = 0.0, 0
    t0 = time.perf_counter()
    for batch in loader:
        optimizador.zero_grad()                      # borra las correcciones del lote anterior
        logits, y, _ = logits_lote(modelo, batch, vista, data, inversas, device)
        loss = perdida(logits, y)                    # cuánto se equivoca en este lote
        loss.backward()                              # hacia dónde mover cada parámetro
        optimizador.step()                           # los mueve un poco (lr)
        total += loss.item() * y.numel()
        n += y.numel()
    return total / n, time.perf_counter() - t0


@torch.no_grad()                                     # solo puntuar: no guarda nada para aprender
def puntuar(modelo, loader, vista, data, inversas, device):
    """Probabilidad de blanqueo de cada semilla, en el orden de vista['edge_label_id'].

    Devuelve (p [S_total] en numpy, segundos).
    """
    modelo.eval()                                    # apaga dropout; BatchNorm usa lo aprendido
    n = vista["edge_label_id"].numel()
    p = torch.full((n,), float("nan"))               # [S_total]  se rellena lote a lote
    t0 = time.perf_counter()
    for batch in loader:
        logits, _, _ = logits_lote(modelo, batch, vista, data, inversas, device)
        p[batch.input_id] = torch.sigmoid(logits).cpu()   # cada probabilidad en la posición de su semilla
    assert not torch.isnan(p).any(), "hay semillas sin puntuar"
    return p.numpy(), time.perf_counter() - t0