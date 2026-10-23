import argparse, json, yaml, torch
import numpy as np
from pathlib import Path

from src.eval.metrics import umbral_optimo_f1, evaluar, imprimir_matriz
from src.models.tabular import cargar_features
from src.models.baseline import peso_clase, col_to_index, entrenar_xgboost, puntuacion


ROOT = Path(__file__).resolve().parents[1]

def main():
    config = None
    p = argparse.ArgumentParser()
    p.add_argument('--config', type = Path, required = True)
    args = p.parse_args()

    with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)

    grafo = torch.load(ROOT / config["grafo"], weights_only = False)

    X, y, split, nombres = cargar_features(grafo["data"], grafo["meta"]["features_arista"])

    train = split == 0
    validation = split == 1
    test = split == 2

    peso = peso_clase(y[train])

    salida = ROOT / config["salida"]

    print(f"train {int(train.sum())} | val {int(validation.sum())} | test {int(test.sum())} | peso {peso:.1f}")

    for exp, columns in config["experimentos"].items():
        index = col_to_index(nombres, columns)
        Xe = X[:, index]

        for seed in config["semillas"]:
            modelo, entrenar = entrenar_xgboost(Xe[train], y[train], Xe[validation], y[validation],
                                            config["xgb"], peso, seed)
            p_val, _ = puntuacion(modelo, Xe[validation])
            p_test, t_predecir = puntuacion(modelo, Xe[test])

            umbral = umbral_optimo_f1(y[validation], p_val)

            res = evaluar(y[test], p_test, umbral)

            res.update ({
                "experimento":   exp,
                "semilla":       seed,
                "columnas":      columns,
                "peso":          peso,
                "best_iteration": int(modelo.best_iteration),
                "t_entrenar_s":  entrenar,
                "t_predecir_s":  t_predecir,
                "importancia":   {c: float(v) for c, v in zip(columns, modelo.feature_importances_)},
                "curva_val":     [float(v) for v in modelo.evals_result()["validation_0"]["aucpr"]],
                "params":        modelo.get_params(),
                "grafo":         config["grafo"],
            })

            base = f"{exp}_semilla{seed}"
            with open(salida / f"{base}.json", "w") as f:
                json.dump(res, f, indent=2)
            np.savez(salida / f"{base}.npz", p_val=p_val, p_test=p_test)

            print(f"{exp:22s} semilla {seed} | "
                  f"F1 {res['f1']:.3f} | AUC-PR {res['auc_pr']:.4f} | "
                  f"recall@1% {res['recall_fpr1']:.3f} | "
                  f"árboles {modelo.best_iteration} | {entrenar:.1f} s")
            imprimir_matriz(res)
            print()
             

if __name__ == '__main__':
    main()






    
