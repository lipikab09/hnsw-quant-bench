import numpy as np


def generate_clustered_embeddings(
    n_vectors: int = 50_000,
    dim: int = 256,
    n_clusters: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic, anisotropic L2-normalized semantic embeddings
    modeled as a Gaussian Mixture on the unit hypersphere (mimicking
    real transformer dense embeddings rather than uniform white noise).
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    cluster_ids = rng.integers(0, n_clusters, size=n_vectors)
    spread = rng.uniform(0.15, 0.35, size=(n_clusters, 1)).astype(np.float32)

    noise = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    vectors = centroids[cluster_ids] + noise * spread[cluster_ids]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(vectors, dtype=np.float32), cluster_ids


def generate_queries(
    base_vectors: np.ndarray,
    n_queries: int = 200,
    noise_scale: float = 0.18,
    seed: int = 109,
) -> np.ndarray:
    """Generates realistic out-of-sample query vectors near data manifolds."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base_vectors), size=n_queries, replace=False)
    queries = base_vectors[idx] + rng.standard_normal((n_queries, base_vectors.shape[1])).astype(np.float32) * noise_scale
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(queries, dtype=np.float32)


def compute_exact_ground_truth(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Flat L2 / Cosine ground truth search (100% Recall baseline).
    Uses chunked BLAS matrix multiplication to keep peak memory bounded.
    """
    n_queries = len(queries)
    gt_indices = np.empty((n_queries, k), dtype=np.int32)
    gt_dists = np.empty((n_queries, k), dtype=np.float32)

    base_sq = np.sum(base_vectors ** 2, axis=1, keepdims=True).T  # (1, N)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = queries[start:end]
        q_sq = np.sum(q_batch ** 2, axis=1, keepdims=True)
        # Squared L2 distance: ||q||^2 + ||x||^2 - 2 q.x
        dists = np.maximum(q_sq + base_sq - 2.0 * (q_batch @ base_vectors.T), 0.0)
        part_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        part_dists = np.take_along_axis(dists, part_idx, axis=1)
        order = np.argsort(part_dists, axis=1)
        gt_indices[start:end] = np.take_along_axis(part_idx, order, axis=1)
        gt_dists[start:end] = np.take_along_axis(part_dists, order, axis=1)

    return gt_indices, gt_dists
