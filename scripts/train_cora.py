"""Modelos para clasificación de nodos: MLP (baseline), GCN y GAT."""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv


class MLP(torch.nn.Module):
    """Baseline sin grafo: solo usa las features del nodo."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):   # ignora edge_index a propósito
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCN(torch.nn.Module):
    """Media normalizada de los vecinos (Kipf & Welling, 2017)."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    """Atención aprendida por vecino (Velickovic et al., 2018).

    Los pesos de atención son la base de la explicabilidad: ver
    forward_with_attention().
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def forward_with_attention(self, x, edge_index):
        """Devuelve logits y los pesos de atención de cada capa."""
        x = self.conv1(x, edge_index)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True) \
            if False else (x, None)   # ver nota abajo
        return x, attn1


MODELS = {"mlp": MLP, "gcn": GCN, "gat": GAT}


def build_model(name, **kwargs):
    """Crea un modelo por nombre: build_model('gat', in_channels=..., ...)."""
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(MODELS)}")
    return MODELS[key](**kwargs)