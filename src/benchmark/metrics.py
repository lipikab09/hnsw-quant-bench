import numpy as np


def compute_recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Computes mean Recall@K across all evaluation queries."""
    n_queries = len(ground_truth)
    recalls = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k].tolist())
        ret_set = set(retrieved[i, :k].tolist())
        recalls[i] = len(gt_set & ret_set) / float(k)
    return float(np.mean(recalls))


def extract_pareto_frontier(configs: list[dict]) -> list[dict]:
    """
    Identifies Pareto-optimal index configurations maximizing Recall@10
    while minimizing p95 Latency (ms) and Index Memory (MB).
    """
    pareto = []
    for i, a in enumerate(configs):
        dominated = False
        for j, b in enumerate(configs):
            if i == j:
                continue
            better_or_equal = (
                b["recall_at_10"] >= a["recall_at_10"]
                and b["p95_latency_ms"] <= a["p95_latency_ms"]
                and b["total_index_mb"] <= a["total_index_mb"]
            )
            strictly_better = (
                b["recall_at_10"] > a["recall_at_10"]
                or b["p95_latency_ms"] < a["p95_latency_ms"]
                or b["total_index_mb"] < a["total_index_mb"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        a["is_pareto_optimal"] = not dominated
        if not dominated:
            pareto.append(a)
    return sorted(pareto, key=lambda x: (-x["recall_at_10"], x["p95_latency_ms"]))
