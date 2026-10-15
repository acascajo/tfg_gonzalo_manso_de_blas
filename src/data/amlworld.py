"""
src/data/amlworld.py

Construcción del grafo de transacciones a partir del CSV de AMLworld HI-Small.

Nodo = cuenta bancaria. Arista = transacción. Tarea = clasificación de aristas.

El flujo de construcción es:
    cargar CSV → truncar → ventana → split → agregaciones → escalado → Data
    (el orden de los dos últimos pasos es importante para evitar fuga de información)

    

"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


FEATURES_DISPONIBLES = ('importe', 'hora', 'formato')

FORMATOS = ('ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire')


# ── Carga y preparación ──────────────────────────────────────────────────────

def cargar(ruta_csv: Path) -> pd.DataFrame:
    """Lee el CSV, convierte Timestamp a datetime, crea Cambio_Div y ordena
    cronológicamente con kind='stable'.
    """
    df = pd.read_csv(ruta_csv)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    df['Cambio_Div'] = df['Payment Currency'] != df['Receiving Currency']
    df = df.sort_values('Timestamp', kind='stable').reset_index(drop=True)

    assert df['Timestamp'].is_monotonic_increasing, "el DataFrame no ha quedado ordenado"
    return df


def tabla_tipos_cambio(df: pd.DataFrame) -> dict[str, float]:
    """Devuelve el precio en dólares de cada divisa. """
    cols = ['Amount Received', 'Amount Paid']
    sumas = (df[df['Cambio_Div']]
             .groupby(['Payment Currency', 'Receiving Currency'])[cols]
             .sum())
    tipos = sumas['Amount Received'] / sumas['Amount Paid']

    a_usd = {origen: r for (origen, destino), r in tipos.items()
             if destino == 'US Dollar'}
    a_usd['US Dollar'] = 1.0

    faltan = set(df['Payment Currency'].unique()) - set(a_usd)
    assert not faltan, f"divisas sin tipo directo a dólar: {faltan}"
    assert all(np.isfinite(r) and r > 0 for r in a_usd.values()), "tipos no válidos"
    return a_usd


def normalizar_importes(df: pd.DataFrame, a_usd: dict) -> pd.DataFrame:
    """Añade la columna amount_usd. Comprueba que no queden nulos."""

    df = df.copy()

    df['amount_usd'] = df['Amount Paid'] * df['Payment Currency'].map(a_usd)

    assert df['amount_usd'].isna().sum() == 0, "hay importes sin convertir"

    assert (df['amount_usd'] > 0).all(), "hay importes no positivos"
    
    return df

def truncar(df: pd.DataFrame, dia: int = 10) -> pd.DataFrame:
    """Descarta las transacciones posteriores al día indicado. En este caso día 10"""
    corte = df['Timestamp'].min().normalize() + pd.Timedelta(days=dia)
    n_antes = len(df)
    df = df[df['Timestamp'] < corte].reset_index(drop=True)
    print(f"Truncado: {n_antes} → {len(df)} transacciones "
        f"({df['Is Laundering'].sum()} ilícitas)")

    assert len(df) > 0, "el truncado ha eliminado todas las transacciones"

    return df


def ventana_desarrollo(df: pd.DataFrame,
                       inicio: int | None = None,
                       fin: int | None = None) -> pd.DataFrame:
    """Recorta a los días [inicio, fin], ambos incluidos (el primer día es el 1).

    Sin inicio ni fin devuelve el DataFrame entero.
    """
    if inicio is None and fin is None:
        return df
    if inicio is None or fin is None:
        raise ValueError("hay que indicar inicio y fin de la ventana, o ninguno de los dos")
    assert 1 <= inicio <= fin, f"ventana mal definida: inicio={inicio}, fin={fin}"

    t0 = df['Timestamp'].min().normalize()
    dia = (df['Timestamp'].dt.normalize() - t0).dt.days + 1

    n_antes = len(df)
    df = df[(dia >= inicio) & (dia <= fin)].reset_index(drop=True)
    print(f"Ventana días {inicio}-{fin}: {n_antes} → {len(df)} transacciones "
          f"({df['Is Laundering'].sum()} ilícitas)")

    assert len(df) > 0, "la ventana no contiene ninguna transacción"
    return df


# ── Partición ────────────────────────────────────────────────────────────────

def corte_limpio(ts: pd.Series, pos):
    """Avanza pos hasta el primer índice con marca temporal distinta, de modo que el corte
    no parta un grupo de transacciones simultáneas entre dos particiones."""
    t = ts.iloc[pos]
    while pos < len(ts) and ts.iloc[pos] == t:
        pos += 1
    return pos

def split_temporal(df: pd.DataFrame,
                   props: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> np.ndarray:
    """Devuelve un array de 0/1/2 por transacción.

    El corte se desplaza al primer cambio de marca temporal, de modo que un grupo
    de transacciones simultáneas no quede repartido entre dos particiones.
    """

    assert abs(sum(props) - 1.0) < 1e-9, "las proporciones deben sumar 1"

    n = len(df)
    c1 = corte_limpio(df['Timestamp'], int(n * props[0]))
    c2 = corte_limpio(df['Timestamp'], int(n * (props[0] + props[1])))

    assert 0 < c1 < c2 < n, "los cortes no dejan las tres particiones con datos"

    split = np.zeros(n, dtype=int)
    split[c1:c2] = 1
    split[c2:] = 2

    return split


# ── Estructura del grafo ─────────────────────────────────────────────────────

def mapear_cuentas(df: pd.DataFrame) -> dict[str, int]:
    """Asigna un índice entero consecutivo a cada cuenta única.

    Se construye sobre el mismo DataFrame al que después se aplica: un mapeo
    construido sobre otro conjunto produce nulos.
    """

    unicas = pd.unique(pd.concat([df['Account'], df['Account.1']], ignore_index=True))
    mapeo = {cuenta: i for i, cuenta in enumerate(unicas)}

    assert df['Account'].isin(mapeo).all(),   "hay emisores fuera del mapeo"
    assert df['Account.1'].isin(mapeo).all(), "hay receptores fuera del mapeo"
    return mapeo

def construir_edge_index(df: pd.DataFrame, mapeo: dict) -> np.ndarray:
    """Matriz (2, n_aristas) en formato COO, de tipo entero."""

    origen  = df['Account'].map(mapeo)
    destino = df['Account.1'].map(mapeo)

    # ANTES del cast: un NaN se convierte en un entero negativo enorme sin lanzar
    # excepción, y supera el resto de comprobaciones.
    assert origen.notna().all() and destino.notna().all(), \
        "hay cuentas que no están en el mapeo"

    edge_index = np.vstack([origen.to_numpy(), destino.to_numpy()]).astype(np.int64)

    assert edge_index.shape == (2, len(df)), "edge_index con forma incorrecta"
    assert edge_index.min() >= 0, "hay índices de nodo negativos"
    return edge_index


# ── Características ──────────────────────────────────────────────────────────

def features_arista(df: pd.DataFrame,
                    features: tuple[str, ...] = ('importe',)) -> tuple[np.ndarray, list[str]]:
    """Características por transacción. Sin la posición temporal absoluta."""
    desconocidas = set(features) - set(FEATURES_DISPONIBLES)
    assert not desconocidas, f"features desconocidas: {desconocidas}"
    assert len(features) > 0, "hay que pedir al menos una feature"
 
    columnas, nombres = [], []
 
    if 'importe' in features:
        columnas.append(np.log1p(df['amount_usd'].to_numpy())[:, None])
        nombres.append('log_importe_usd')
 
    if 'hora' in features:
        # Codificación cíclica: 23 h y 0 h quedan juntas, no en extremos opuestos.
        ang = 2 * np.pi * df['Timestamp'].dt.hour.to_numpy() / 24
        columnas.append(np.column_stack([np.sin(ang), np.cos(ang)]))
        nombres += ['hora_sin', 'hora_cos']
 
    if 'formato' in features:
        fmt = df['Payment Format']
        raros = set(fmt.unique()) - set(FORMATOS)
        assert not raros, f"formatos de pago no previstos: {raros}"
        columnas.append(np.column_stack([(fmt == f).to_numpy() for f in FORMATOS]))
        nombres += [f'formato_{f}' for f in FORMATOS]
 
    edge_attr = np.hstack(columnas).astype(np.float32)
 
    assert edge_attr.shape == (len(df), len(nombres)), "edge_attr con forma incorrecta"
    assert np.isfinite(edge_attr).all(), "hay valores no finitos en edge_attr"
    return edge_attr, nombres



def features_nodo(df: pd.DataFrame, split: np.ndarray, mapeo: dict) -> np.ndarray:
    """Características por cuenta, agregadas SOLO sobre split == 0.

    Calcularlas sobre el conjunto completo introduciría información de validación
    y prueba en el perfil de cada cuenta.
    """
    df_train = df[split == 0]

    salidas  = df_train.groupby('Account')['amount_usd'].agg(['count', 'sum'])
    entradas = df_train.groupby('Account.1')['amount_usd'].agg(['count', 'sum'])

    x = np.zeros((len(mapeo), 4), dtype=np.float32)

    pos_out = salidas.index.map(mapeo)
    x[pos_out, 0] = salidas['count'].to_numpy()
    x[pos_out, 1] = np.log1p(salidas['sum'].to_numpy())

    pos_in = entradas.index.map(mapeo)
    x[pos_in, 2] = entradas['count'].to_numpy()
    x[pos_in, 3] = np.log1p(entradas['sum'].to_numpy())

    # Cada transacción de entrenamiento tiene un emisor y un receptor: los conteos
    # tienen que sumar exactamente el número de transacciones. Va antes del log1p
    # de escalar(), que destruye el invariante.
    assert x[:, 0].sum() == len(df_train), "los conteos de salida no cuadran"
    assert x[:, 2].sum() == len(df_train), "los conteos de entrada no cuadran"
    return x




def escalar(x: np.ndarray, edge_attr: np.ndarray,
            edge_index: np.ndarray, split: np.ndarray):
    """Aplica log1p a los conteos y estandariza, con estadísticos de
    entrenamiento únicamente.

    Devuelve (x, edge_attr, mu_x, std_x, mu_e, std_e). Los estadísticos hacen
    falta después para escalar las aristas que añada el atacante.
    """

    x = x.astype(np.float64)
    edge_attr = edge_attr.astype(np.float64)

    # 1. log1p a los conteos, que los sitúa en el mismo orden que los volúmenes
    x[:, 0] = np.log1p(x[:, 0])
    x[:, 2] = np.log1p(x[:, 2])

    # 2. Estandarización con estadísticos de ENTRENAMIENTO únicamente, por el mismo motivo por
    #    el que las agregaciones de x se calculan solo sobre split == 0.
    mask_train = (split == 0)

    mu_e  = edge_attr[mask_train].mean(axis=0)
    std_e = edge_attr[mask_train].std(axis=0)
    std_e[std_e == 0] = 1.0
    edge_attr = (edge_attr - mu_e) / std_e

    nodos_train = np.unique(edge_index[:, mask_train])
    mu_x  = x[nodos_train].mean(axis=0)
    std_x = x[nodos_train].std(axis=0)
    std_x[std_x == 0] = 1.0
    x = (x - mu_x) / std_x

    return x, edge_attr, mu_x, std_x, mu_e, std_e


# ── Orquestación ─────────────────────────────────────────────────────────────

def build_graph(ruta_csv: Path,
                truncar_dia: int = 10,
                ventana_inicio: int | None = None,
                ventana_fin: int | None = None,
                props: tuple = (0.6, 0.2, 0.2),
                features: tuple[str, ...] = ('importe',)) -> tuple[Data, dict]:
    """Construye el objeto Data completo y sus metadatos.

    Orden obligatorio: truncar → ventana → split → agregaciones → escalado.
    Invertir cualquiera de esos dos últimos pasos introduce fuga de información.

    Devuelve (data, meta), con meta conteniendo mapeo, a_usd, mu/std y los
    timestamps de corte.
    """
    # --- preparación ---
    df = cargar(ruta_csv)
    df = truncar(df, dia=truncar_dia)
    df = ventana_desarrollo(df, ventana_inicio, ventana_fin)
    df = df.reset_index(drop=True)

    a_usd = tabla_tipos_cambio(df)
    df = normalizar_importes(df, a_usd)
    mapeo = mapear_cuentas(df)

    # --- partición temporal: todo lo que venga después la respeta ---
    split = split_temporal(df, props)

    # --- tensores ---
    edge_index           = construir_edge_index(df, mapeo)
    edge_attr, nombres_e = features_arista(df, features)
    x                    = features_nodo(df, split, mapeo)
    y                    = df['Is Laundering'].to_numpy()
 
    x, edge_attr, mu_x, std_x, mu_e, std_e = escalar(x, edge_attr, edge_index, split)

    # --- objeto PyG ---
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    # Nombres explícitos de atributo de arista: PyG infiere el tipo en parte por el
    # nombre, y 'y' es convencionalmente la etiqueta de NODO. Con etiquetas por arista,
    # los muestreadores de subgrafos las recortarían mal y en silencio.
    data.edge_label = torch.tensor(y, dtype=torch.long)
    data.edge_split = torch.tensor(split, dtype=torch.long)
    # Identificador de cada transacción: entre dos cuentas puede haber muchas,
    # y es la única forma de saber cuál concreta hay que clasificar.
    data.edge_id = torch.arange(data.edge_index.size(1), dtype=torch.long)

    # --- invariantes ---
    assert data.edge_index.max() < data.x.size(0), "índice de nodo fuera de rango"
    assert data.edge_index.size(1) == data.edge_label.size(0) == data.edge_attr.size(0), \
        "número de aristas inconsistente entre index, label y attr"
    assert not torch.isnan(data.x).any(), "hay NaN en las features de nodo"
    assert not torch.isnan(data.edge_attr).any(), "hay NaN en las features de arista"

    for nombre, v in (('train', 0), ('val', 1), ('test', 2)):
        m = (split == v)
        assert m.sum() > 0, f"la partición {nombre} está vacía"
        assert y[m].sum() > 0, f"la partición {nombre} no tiene ninguna ilícita"

    meta = {
        'mapeo': mapeo,
        'a_usd': a_usd,
        'features_arista': nombres_e,
        'mu_x': mu_x, 'std_x': std_x,
        'mu_e': mu_e, 'std_e': std_e,
        'corte_val':  df['Timestamp'][split == 1].min(),
        'corte_test': df['Timestamp'][split == 2].min(),
        'truncar_dia': truncar_dia,
        'ventana_inicio': ventana_inicio,
        'ventana_fin': ventana_fin,
        'props': props,
    }
    return data, meta

def vistas(data: Data) -> dict:
    """Las tres vistas del protocolo de los tres grafos.

    Cada entrada devuelve las aristas sobre las que se propaga el mensaje y las
    aristas que hay que predecir:

        train → (edge_split == 0, edge_split == 0)
        val   → (edge_split <= 1, edge_split == 1)
        test  → (todas,           edge_split == 2)

    Encaja directamente con LinkNeighborLoader, que acepta por separado el grafo
    del que muestrea y las aristas semilla.
    """
    s = data.edge_split
    ve      = {'train': s == 0, 'val': s <= 1, 'test': s <= 2}
    predice = {'train': s == 0, 'val': s == 1, 'test': s == 2}

    out = {}
    for nombre in ('train', 'val', 'test'):
        m_ve, m_pred = ve[nombre], predice[nombre]

        # Toda arista a predecir tiene que ser visible: si no, el loader la
        # muestrearía de un grafo donde no existe.
        assert not (m_pred & ~m_ve).any(), f"{nombre}: hay semillas no visibles"

        grafo = data.edge_subgraph(m_ve)
        del grafo.edge_label, grafo.edge_split

        out[nombre] = {
            'data': grafo,
            'edge_label_index': data.edge_index[:, m_pred],
            'edge_label': data.edge_label[m_pred],
            'edge_label_id': data.edge_id[m_pred],
        }

    # Las tres particiones son disjuntas y cubren el grafo entero.
    total = sum(v['edge_label'].numel() for v in out.values())
    assert total == data.edge_label.numel(), "las semillas no cubren todas las aristas"

    return out