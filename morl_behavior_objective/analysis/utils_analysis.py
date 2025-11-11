"""Utility functions for analyzing MORL behavior embeddings and computing metrics."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Compute Lipschitz constants
def compute_lipschitz_constants(
    pareto_frontier: np.ndarray, embeddings: np.ndarray, *, normalize=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise Lipschitz constants between consecutive points in the Pareto frontier and embedding space.

    Args:
        pareto_frontier: (n_points, d_obj) array of objective values
        embeddings: (n_points, d_emb) array of embeddings
        normalize: whether to normalize the Pareto front to [-1, 1]

    Returns:
        lipschitz_constants: (n_points-1,) array of Lipschitz constants
        d_objs: distances in objective space
        d_embs: distances in embedding space
        sorted_pf: Pareto front ordered by proximity
        sorted_emb: embeddings ordered by proximity
    """
    # Lexicographic sort: np.lexsort expects keys in reverse order
    sort_idx = np.lexsort(pareto_frontier.T)
    sorted_pf = pareto_frontier[sort_idx]
    sorted_emb = embeddings[sort_idx]

    if normalize:
        # Normalize sorted pf to range [-1,1]
        scaler_pf = MinMaxScaler(feature_range=(-1, 1))
        sorted_pf = scaler_pf.fit_transform(sorted_pf)

    lipschitz_constants = []
    d_objs = []
    d_embs = []
    for i in range(len(sorted_pf) - 1):
        print(f"Computing Lipschitz constant between policy {i} and {i + 1}")
        d_obj = np.linalg.norm(sorted_pf[i + 1] - sorted_pf[i])
        print(d_obj)
        d_emb = np.linalg.norm(sorted_emb[i + 1] - sorted_emb[i])
        print(d_emb)
        d_objs.append(d_obj)
        d_embs.append(d_emb)
        lipschitz_constants.append(d_emb / d_obj)
        print(f"Lipshitz: {d_emb / d_obj}")

        print(
            f"Lipschitz constant for policy {i} ({sorted_pf[i]}, {sorted_emb[i]}) to {i + 1} ({sorted_pf[i + 1]}, {sorted_emb[i + 1]}): {lipschitz_constants[-1]}, d_obj: {d_obj}, d_emb: {d_emb}"
        )

    return np.array(lipschitz_constants), np.array(d_objs), np.array(d_embs), sorted_pf, sorted_emb


def compute_lipschitz_constants_3d(
    pareto_frontier: np.ndarray, embeddings: np.ndarray, *, normalize=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise Lipschitz constants between consecutive points in the Pareto frontier and embedding space for 3D (or higher) fronts.

    Args:
        pareto_frontier: (n_points, d_obj) array of objective values
        embeddings: (n_points, d_emb) array of embeddings
        normalize: whether to normalize the Pareto front to [-1, 1]

    Returns:
        lipschitz_constants: (n_points-1,) array
        d_objs: distances in objective space
        d_embs: distances in embedding space
        sorted_pf: Pareto front ordered by proximity
        sorted_emb: embeddings ordered by proximity
    """
    # Step 1: sequence points by nearest neighbor
    pf_points = pareto_frontier.copy()
    emb_points = embeddings.copy()
    n_points = len(pf_points)

    seq_idx = [0]  # start with first point
    remaining_idx = list(range(1, n_points))

    while remaining_idx:
        last = pf_points[seq_idx[-1]]
        remaining = pf_points[remaining_idx]
        distances = np.linalg.norm(remaining - last, axis=1)
        nearest_idx = remaining_idx[np.argmin(distances)]
        seq_idx.append(nearest_idx)
        remaining_idx.remove(nearest_idx)

    sorted_pf = pf_points[seq_idx]
    sorted_emb = emb_points[seq_idx]

    if normalize:
        scaler_pf = MinMaxScaler(feature_range=(-1, 1))
        sorted_pf = scaler_pf.fit_transform(sorted_pf)

    # Step 2: compute Lipschitz constants
    lipschitz_constants = []
    d_objs = []
    d_embs = []

    for i in range(n_points - 1):
        d_obj = np.linalg.norm(sorted_pf[i + 1] - sorted_pf[i])
        d_emb = np.linalg.norm(sorted_emb[i + 1] - sorted_emb[i])
        d_objs.append(d_obj)
        d_embs.append(d_emb)
        lipschitz_constants.append(d_emb / d_obj)

    return np.array(lipschitz_constants), np.array(d_objs), np.array(d_embs), sorted_pf, sorted_emb
