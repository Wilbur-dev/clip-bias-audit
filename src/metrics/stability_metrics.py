import numpy as np


def thresholded_sign(x: float, eps: float = 1e-4) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def prompt_directional_consistency(gaps, eps: float = 1e-4) -> float:
    """
    C = | mean(sign_eps(delta_p)) |
    """
    signs = np.array([thresholded_sign(x, eps=eps) for x in gaps])
    return float(abs(signs.mean()))


def prompt_sensitivity(gaps) -> float:
    """
    V = std(delta across prompts)
    """
    gaps = np.asarray(gaps, dtype = float)

    if len(gaps) <= 1:
        return 0.0

    return float(gaps.std(ddof = 1))