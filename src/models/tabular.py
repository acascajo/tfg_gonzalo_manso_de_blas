import torch
from pathlib import Path


PERFIL = ["n_env", "vol_env", "n_rec", "vol_rec"]

def cargar_features(data, nombre_arista):

    origen = data.edge_index[0]
    destino = data.edge_index[1]

    perfil_origen = data.x[origen]
    perfil_destino = data.x[destino]

    X = torch.cat([data.edge_attr, perfil_origen, perfil_destino], dim=1)

    nombres = nombre_arista + ["origen_" + p for p in PERFIL] + ["destino_" + p for p in PERFIL]

    return X.numpy(), data.edge_label.numpy(), data.edge_split.numpy(), nombres

