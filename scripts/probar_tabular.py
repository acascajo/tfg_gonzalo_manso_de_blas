import numpy as np
import torch
from pathlib import Path


from src.models.tabular import cargar_features


ROOT = Path(__file__).resolve().parents[1]
GRAFO = ROOT / "data" / "processed" / "amlworld_dev.pt"



def main():

    grafo  = torch.load(GRAFO, weights_only=False)

    data = grafo["data"]
    meta = grafo["meta"]

    X, y, split, nombres = cargar_features(data, meta["features_arista"])

    
    E = data.edge_index.size(1)
    n_e = data.edge_attr.size(1)     

    origen = data.edge_index[0].numpy()
    destino = data.edge_index[1].numpy()

    x = data.x.numpy()
    edge_attr = data.edge_attr.numpy()


    assert X.shape == (E, n_e + 8), f"forma inesperada: {X.shape}"
    assert y.shape == (E,) and split.shape == (E,), "y o split no tienen una posición por transacción"
    assert len(nombres) == X.shape[1], f"hay {len(nombres)} nombres para {X.shape[1]} columnas"
    assert nombres[:n_e] == meta["features_arista"], f"los nombres de arista no coinciden: {nombres[:n_e]}"

    assert np.array_equal(y, data.edge_label.numpy()), "y no coincide con edge_label"
    assert np.array_equal(split, data.edge_split.numpy()), "split no coincide con edge_split"


    assert not np.isnan(X).any(), "hay NaN en la tabla"


    rng = np.random.default_rng(0)
    indices = [int(i) for i in rng.choice(E, size=5, replace=False)]
    indices.append(int(np.flatnonzero(y == 1)[0]))      # la primera ilícita
    for i in indices:
        o, d = origen[i], destino[i]
        assert np.array_equal(X[i, :n_e], edge_attr[i]), f"transacción {i}: importe mal"
        assert np.array_equal(X[i, n_e:n_e + 4], x[o]), f"transacción {i}: perfil de origen mal"
        assert np.array_equal(X[i, n_e + 4:], x[d]), f"transacción {i}: perfil de destino mal"

    
    i = int(np.flatnonzero(origen == destino)[0])
    assert np.array_equal(X[i, n_e:n_e + 4], X[i, n_e + 4:]), f"bucle {i}: los dos perfiles no coinciden"



    print()
    for nombre, k in (("train", 0), ("val", 1), ("test", 2)):
        m = split == k
        print(f"{nombre:5s} {int(m.sum()):9d} transacciones  {int(y[m].sum()):5d} ilícitas")
    print("\nTodo correcto.")





if __name__ == '__main__':
    main()

