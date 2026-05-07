import numpy as np
from scipy.stats import wasserstein_distance


def mean_gap(group_a_scores, group_b_scores):
    group_a_scores = np.asarray(group_a_scores)
    group_b_scores = np.asarray(group_b_scores)

    return group_a_scores.mean() - group_b_scores.mean()


def cohens_d(group_a_scores, group_b_scores, eps=1e-12):
    group_a_scores = np.asarray(group_a_scores)
    group_b_scores = np.asarray(group_b_scores)

    n_a = len(group_a_scores)
    n_b = len(group_b_scores)

    mean_a = group_a_scores.mean()
    mean_b = group_b_scores.mean()

    std_a = group_a_scores.std(ddof = 1)
    std_b = group_b_scores.std(ddof = 1)

    pooled_std = np.sqrt(
        ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2)
        / (n_a + n_b - 2)
    )

    return (mean_a - mean_b) / (pooled_std + eps)


def wasserstein_1d(group_a_scores, group_b_scores):
    group_a_scores = np.asarray(group_a_scores)
    group_b_scores = np.asarray(group_b_scores)
    return float(wasserstein_distance(group_a_scores, group_b_scores))