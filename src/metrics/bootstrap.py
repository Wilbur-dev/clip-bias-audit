import numpy as np


def bootstrap_mean_ci(values, n_boot = 200, ci = 95, seed = 42):
    rng = np.random.default_rng(seed)

    values = np.asarray(values)
    n = len(values)

    stats = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(sample.mean())

    stats = np.array(stats)

    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)

    return float(lower), float(upper)