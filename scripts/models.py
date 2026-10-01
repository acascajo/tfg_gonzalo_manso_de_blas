"""Compara MLP, GCN y GAT sobre Cora."""

import sys
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_cora import build_model
from src.utils.seeds import get_device, set_seed


def load_data(device):
    dataset = Planetoid(root="data/raw/Planetoid", name="Cora",
                        transform=NormalizeFeatures())
    return dataset, dataset[0].to(device)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    return int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())


def run(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    """Entrena y devuelve el test de la mejor época de validación."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_test = evaluate(model, data, data.test_mask)

        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val {val_acc:.4f}")

    return best_val, best_test


def main():
    set_seed(42)
    device = get_device()
    dataset, data = load_data(device)

    print(f"\nCora: {data.num_nodes} nodos, {data.num_edges} aristas, "
          f"{dataset.num_features} features, {dataset.num_classes} clases\n")

    configs = {
        "mlp": {"hidden_channels": 16},
        "gcn": {"hidden_channels": 16},
        "gat": {"hidden_channels": 8, "heads": 8},
    }

    results = {}
    for name, params in configs.items():
        print(f"--- {name.upper()} ---")
        model = build_model(
            name,
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            **params,
        ).to(device)

        lr = 0.005 if name == "gat" else 0.01
        best_val, best_test = run(model, data, lr=lr)
        results[name] = best_test
        print(f"  mejor val {best_val:.4f} -> test {best_test:.4f}\n")

    print("=== Resumen ===")
    for name, acc in results.items():
        print(f"  {name.upper():5s}: {acc:.4f}")


if __name__ == "__main__":
    main()