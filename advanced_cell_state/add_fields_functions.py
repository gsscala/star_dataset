import numpy as np

def derivative(library: list, pool: int) -> list:
    n = len(library)
    derivatives = []
    
    for x in range(n):
        x1 = max(x - pool, 0)
        x2 = min(x + pool, n - 1)
        variation = (library[x2] - library[x1]) / (x2 - x1)
        derivatives.append(variation)
        
    return derivatives

def relaxed_sign(num: float, eps: float) -> int:
    if num > eps:
        return 1
    if (num < -eps):
        return -1
    return 0

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj