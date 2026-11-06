"""Prototipo del ataque: ¿se puede bajar la puntuación de una transacción de blanqueo
añadiendo transacciones desde la cuenta del blanqueador?

No es el ataque definitivo (ese elegirá en cada paso la mejor acción según su coste).
Aquí, para unas pocas transacciones de blanqueo que la GNN detecta, se prueban tres
tipos de acción por separado y se mira cómo cambia la puntuación:

    conocida    la cuenta de origen paga a una cuenta a la que ya pagaba
    mula        paga a una cuenta mula nueva (sin historial), creada por él
    ajena       paga a una cuenta cualquiera del banco, elegida al azar
    desde_mula  una mula nueva suya le paga a él (para parecer un negocio que cobra)

Supuestos (modelo de amenaza, 4 nov): caja blanca, solo se AÑADEN transacciones desde
la cuenta de origen, el perfil x no se recalcula, y el subgrafo de la transacción se
muestrea una vez y se deja fijo (si no, la puntuación cambiaría por azar).
Limitación del prototipo: si el destino es una cuenta que ya existe, se añade con su
perfil pero sin sus propios vecinos.

Uso:  python -m scripts.probar_ataque --config configs/gnn_full.yaml --semilla 0 --casos 5 --pasos 5
Guarda una traza por caso en results/ataque/prototipo/ (la leerá el visualizador).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import LinkNeighborLoader

from src.data.amlworld import vistas
from src.eval.metrics import umbral_optimo_f1
from src.models.gnn import GINEAristas, attr_semillas

ROOT = Path(__file__).resolve().parents[1]
ACCIONES = ("conocida", "mula", "ajena", "desde_mula")


# ── Subgrafo fijo de una transacción y su puntuación ─────────────────────────

def subgrafo_fijo(vista, pos, num_vecinos, semilla):
    """El trocito de grafo que ve la GNN para la semilla `pos` de la vista, muestreado una vez."""
    torch.manual_seed(semilla)
    loader = LinkNeighborLoader(
        vista["data"], num_neighbors=num_vecinos,
        edge_label_index=vista["edge_label_index"][:, pos:pos + 1],
        edge_label=vista["edge_label"][pos:pos + 1],
        batch_size=1, shuffle=False,
    )
    return next(iter(loader))


@torch.no_grad()
def puntuar(modelo, sub, a_semilla, device):
    """Probabilidad de blanqueo de la semilla del subgrafo."""
    s = sub.clone().to(device) 
    logit = modelo(s.x, s.edge_index, s.edge_attr, s.edge_label_index, a_semilla.to(device))
    return torch.sigmoid(logit).item()


# ── Acciones del atacante ────────────────────────────────────────────────────

def escalar_importe(usd, meta):
    """Dólares → la misma escala que edge_attr[:, 0] (log1p y estandarización de train)."""
    return float((np.log1p(usd) - meta["mu_e"][0]) / meta["std_e"][0])


def cuenta_local(sub, n_id, x):
    """Posición de la cuenta n_id en el subgrafo; si no está, se añade con el perfil x.

    Devuelve (subgrafo, posición). Las mulas nuevas llevan n_id negativo.
    """
    pos = (sub.n_id == n_id).nonzero()
    if n_id >= 0 and len(pos) > 0:
        return sub, int(pos[0])
    sub = sub.clone()
    sub.x = torch.cat([sub.x, x.view(1, -1)])                                 # [N+1, 4]
    sub.n_id = torch.cat([sub.n_id, torch.tensor([n_id])])
    return sub, sub.x.size(0) - 1


def anadir_pago(sub, de, a, importe_esc):
    """Copia del subgrafo con un pago de → a (posiciones locales) y su copia al revés."""
    sub = sub.clone()
    nuevas = torch.tensor([[de, a],                                            # pago (sentido 0)
                           [a, de]])                                           # copia al revés (sentido 1)
    attr = torch.tensor([[importe_esc, 0.0], [importe_esc, 1.0]], dtype=sub.edge_attr.dtype)
    sub.edge_index = torch.cat([sub.edge_index, nuevas], dim=1)                # [2, M+2]
    sub.edge_attr = torch.cat([sub.edge_attr, attr])                           # [M+2, 2]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--semilla", type=int, default=0, help="qué modelo entrenado usar")
    ap.add_argument("--casos", type=int, default=5)
    ap.add_argument("--pasos", type=int, default=5, help="pagos añadidos por tipo de acción")
    ap.add_argument("--salida", default="results/ataque/prototipo")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    salida = ROOT / args.salida
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    obj = torch.load(ROOT / cfg["grafo"], weights_only=False)
    data, meta = obj["data"], obj["meta"]
    vs = vistas(data, inversas=True)
    te, va = vs["test"], vs["val"]

    modelo = GINEAristas(data.x.size(1), data.edge_attr.size(1) + 1, **cfg["modelo"]).to(device)
    modelo.load_state_dict(torch.load(ROOT / cfg["salida"] / f"gnn_s{args.semilla}.pt", map_location=device))
    modelo.eval()

    # Umbral de validación y transacciones de blanqueo que el modelo detecta en test
    npz = np.load(ROOT / cfg["salida"] / f"gnn_s{args.semilla}.npz")
    umbral = umbral_optimo_f1(va["edge_label"].numpy(), npz["p_val"])
    y_te = te["edge_label"].numpy()
    detectadas = np.flatnonzero((y_te == 1) & (npz["p_test"] >= umbral))
    print(f"Umbral {umbral:.3f} | blanqueo en test {int(y_te.sum())} | detectado {len(detectadas)}")
    casos = rng.choice(detectadas, size=min(args.casos, len(detectadas)), replace=False)

    ei = data.edge_index
    x_sin_historial = torch.from_numpy(((0.0 - meta["mu_x"]) / meta["std_x"]).astype(np.float32))

    for pos in casos:
        id_tx = int(te["edge_label_id"][pos])
        origen, destino = (int(v) for v in ei[:, id_tx])
        a = attr_semillas(data, torch.tensor([id_tx]), True)                   # [1, 2]
        usd = float(np.expm1(data.edge_attr[id_tx, 0].item() * meta["std_e"][0] + meta["mu_e"][0]))

        sub0 = subgrafo_fijo(te, int(pos), cfg["cargador"]["num_vecinos"], args.semilla)
        origen_local = int(sub0.edge_label_index[0, 0])
        p0 = puntuar(modelo, sub0, a, device)
        print(f"\nTransacción {id_tx}: {origen} → {destino}, {usd:,.0f} $ | "
              f"puntuación {p0:.3f} (en la evaluación: {npz['p_test'][pos]:.3f}) | "
              f"subgrafo {sub0.x.size(0)} cuentas, {sub0.edge_index.size(1)} aristas")

        # Candidatos de cada tipo de acción
        visibles = data.edge_split <= 2
        conocidas = torch.unique(ei[1, (ei[0] == origen) & visibles]).numpy()
        conocidas = conocidas[conocidas != destino]
        traza = {"id_tx": id_tx, "origen": origen, "destino": destino, "importe_usd": usd,
                 "umbral": umbral, "puntuacion_inicial": p0, "acciones": {}}

        for accion in ACCIONES:
            sub, pasos = sub0, []
            for k in range(args.pasos):
                importe = usd * 0.1                                             # cada pago: 10 % de lo que esconde
                esc = escalar_importe(importe, meta)
                if accion == "conocida":
                    if len(conocidas) == 0:
                        break
                    d = int(rng.choice(conocidas)); x_d = data.x[d]
                elif accion in ("mula", "desde_mula"):
                    d = -(k + 1); x_d = x_sin_historial                         # cuenta nueva: id negativo
                else:
                    d = int(rng.integers(data.x.size(0))); x_d = data.x[d]
                sub, d_local = cuenta_local(sub, d, x_d)
                if accion == "desde_mula":
                    sub = anadir_pago(sub, d_local, origen_local, esc)          # la mula le paga a él
                else:
                    sub = anadir_pago(sub, origen_local, d_local, esc)          # él paga
                p = puntuar(modelo, sub, a, device)
                pasos.append({"destino": d, "importe_usd": importe, "puntuacion": p})
            traza["acciones"][accion] = pasos
            camino = " → ".join(f"{s['puntuacion']:.3f}" for s in pasos) or "(sin cuentas conocidas)"
            bajo = next((i + 1 for i, s in enumerate(pasos) if s["puntuacion"] < umbral), None)
            print(f"  {accion:8s}: {p0:.3f} → {camino}" + (f"   ← bajo el umbral en el pago {bajo}" if bajo else ""))

        with open(salida / f"tx{id_tx}_s{args.semilla}.json", "w", encoding="utf-8") as f:
            json.dump(traza, f, indent=2)


if __name__ == "__main__":
    main()