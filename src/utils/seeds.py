import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Fija las semillas de random, numpy y torch.

    Garantiza que los experimentos sean reproducibles entre ejecuciones.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Devuelve la GPU si está disponible, si no la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
