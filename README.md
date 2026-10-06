# TFG — Detección de blanqueo de capitales con GNN y robustez adversarial

Trabajo de Fin de Grado del Grado en Ingeniería Informática de la Universidad Carlos III de Madrid, curso 2026/27.

Autor: Gonzalo Manso de Blas Tutor: Alberto Cascajo García


## De qué va

La idea es entrenar un detector de blanqueo de capitales sobre un grafo de transacciones bancarias, donde los nodos son cuentas y las aristas son transacciones, y después medir cuánto aguanta ese detector cuando alguien intenta esquivarlo a propósito.

La parte adversarial es lo que diferencia el trabajo de un clasificador normal. En lugar de limitar el ataque a que sea "imperceptible", que es como se suele plantear en visión por computador y aquí no tiene mucho sentido, el atacante tiene un presupuesto económico: abrir cuentas mula cuesta, mover dinero de camuflaje cuesta, y crear aristas cuesta. Con eso se puede dibujar una curva de cuánto se degrada la detección según lo que el atacante esté dispuesto a gastar, que es una pregunta que sí se parece a la realidad.

Lo último es proponer defensas y ver si la curva mejora.

## Dataset

AMLworld HI-Small, de IBM Research y ETH Zurich. Se descarga de Kaggle:

https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

No está en el repositorio: el CSV pesa cerca de medio giga, por encima del límite de GitHub, y además tiene licencia CDLA-Sharing-1.0. Hay que bajarlo y dejarlo en data/raw/. En data/raw/DATASET.md están la versión exacta que he usado y el hash del fichero, porque el dataset se ha actualizado varias veces desde que se publicó el paper y las cifras de la Tabla 4 del artículo ya no coinciden con lo que te descargas hoy.

Son 5.078.345 transacciones con 5.177 marcadas como blanqueo, una de cada 981. Ese desbalanceo condiciona todo: la exactitud no sirve como métrica porque prediciendo siempre "lícito" se acierta el 99,9 %.


### Estructura

configs/      parámetros de los experimentos
data/raw/     el CSV de Kaggle (no versionado)
data/processed/  el grafo ya construido en formato PyG (no versionado)
notebooks/    análisis exploratorio
src/data/     construcción del grafo
src/models/   arquitecturas
src/eval/     métricas
src/utils/    semillas y utilidades
scripts/      entrenamiento y experimentos
results/      figuras y tablas
tests/


### Montarlo

python -m venv .venv
.venv\Scripts\activate         
pip install -r requirements.txt

#Un aviso sobre PyTorch: está fijado a la versión 2.6.0 con CUDA 12.4 a propósito. Con la 2.14 me daba OSError: [WinError 1114] al cargar c10.dll en Windows y no hubo manera de arreglarlo de otra forma. Si instalas torchvision por tu cuenta se te va a volver a subir la versión de torch, así que ojo.#

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

Las pruebas las he hecho con una RTX 4050 de 6 GB. El grafo completo entra en memoria sin problema, lo que hay que vigilar es el tamaño de los lotes del muestreo de vecindario.


### Nota sobre reproducibilidad

Las semillas se fijan en #src/utils/seeds.py#. Aun así, el muestreo de vecindario de PyG introduce variabilidad entre ejecuciones, así que los resultados finales van con cinco semillas y se reporta la desviación. Durante la fase exploratoria uso una sola para no perder tiempo.