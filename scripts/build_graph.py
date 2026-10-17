import argparse
import time
from pathlib import Path

import torch
import yaml

from src.data.amlworld import build_graph


config = None
p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
args = p.parse_args()

with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)


CSV = config["csv"]
TRUNCAR_DIA = config["truncar_dia"]
VENTANA_INICIO = config["ventana"]["inicio"]
VENTANA_FIN = config["ventana"]["fin"]
PROPS = config["split"]
FEATURES_ARISTA = config["features_arista"]

ROOT = Path(__file__).resolve().parents[1]

def main():

    
    t0 = time.perf_counter()

    data, meta = build_graph(CSV, truncar_dia=TRUNCAR_DIA, 
                             ventana_inicio=VENTANA_INICIO, ventana_fin=VENTANA_FIN,
                             props=PROPS, features=FEATURES_ARISTA)

    print(f"Tiempo de construcción: {time.perf_counter() - t0:.1f} s")

    salida = ROOT / config['salida']
    salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'data': data, 'meta': meta, 'config': config}, salida)
    print(data)
    print(f"Guardado en {salida}")


if __name__ == '__main__':
    main()





