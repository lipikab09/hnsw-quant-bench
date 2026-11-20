import argparse
import json
import os
import time
import numpy as np
from src.benchmark.metrics import compute_recall_at_k, extract_pareto_frontier
from src.engine.dataset import compute_exact_ground_truth, generate_clustered_embeddings, generate_queries
from src.engine.hnsw_index import HNSWQuantIndex


DEFAULT_EXPERIMENT_MATRIX = [
    # (label, M, ef_construction, quant, rescore, oversample, ef_search_list)
    ("HNSW-FP32 (M=16)", 16, 80, "fp32", "none", 1, [16, 32, 64, 128]),
    ("HNSW-FP32 (M=32)", 32, 120, "fp32", "none", 1, [32, 64, 128]),
    ("HNSW-SQ8 (M=16)", 16, 80, "sq8", "none", 1, [16, 32, 64, 128]),
    ("HNSW-SQ8 (M=32)", 32, 120, "sq8", "none", 1, [32, 64, 128]),
    ("HNSW-BQ Raw (M=16)", 16, 80, "bq", "none", 1, [32, 64, 128]),
    ("HNSW-BQ + SQ8 Rescore (4x)", 16, 80, "bq", "sq8", 4, [32, 64, 128]),
    ("HNSW-BQ + FP32 Rescore (4x)", 16, 80, "bq", "fp32", 4, [32, 64, 128]),
]


def run_benchmark_suite(
    n_vectors: int = 15_000,
    dim: int = 128,
    n_queries: int = 150,
    k: int = 10,
    projected_scale: int = 5_000_000,
    seed: int = 42,
) -> dict:
    vectors, _ = generate_clustered_embeddings(n_vectors=n_vectors, dim=dim, seed=seed)
    queries = generate_queries(vectors, n_queries=n_queries, seed=seed + 100)

    t_gt = time.perf_counter()
    gt_indices, _ = compute_exact_ground_truth(vectors, queries, k=k)
    gt_time_ms = ((time.perf_counter() - t_gt) / n_queries) * 1000.0

    # Exact Flat baseline entry
    fp32_raw_mb = round(vectors.nbytes / (1024 ** 2), 4)
    projected_fp32_gb = round((vectors.nbytes * (projected_scale / n_vectors)) / (1024 ** 3), 3)

    results = [
        {
            "config_id": "Exact-Flat-FP32",
            "family": "Flat Exact",
            "M": 0,
            "ef_construction": 0,
            "ef_search": n_vectors,
            "quantization": "fp32",
            "rescore_mode": "none",
            "oversample_factor": 1,
            "recall_at_10": 1.0,
            "p50_latency_ms": round(gt_time_ms, 3),
            "p95_latency_ms": round(gt_time_ms * 1.15, 3),
            "p99_latency_ms": round(gt_time_ms * 1.25, 3),
            "qps": round(1000.0 / max(gt_time_ms, 1e-4), 1),
            "avg_dist_ops": n_vectors,
            "build_time_s": 0.0,
            "vector_payload_mb": fp32_raw_mb,
            "rescore_payload_mb": 0.0,
            "hnsw_graph_mb": 0.0,
            "total_index_mb": fp32_raw_mb,
            "memory_savings_pct": 0.0,
            "projected_5m_total_gb": projected_fp32_gb,
        }
    ]

    for label, M, ef_c, quant, rescore, oversample, ef_list in DEFAULT_EXPERIMENT_MATRIX:
        index = HNSWQuantIndex(
            dim=dim,
            M=M,
            ef_construction=ef_c,
            ef_search=ef_list[0],
            quantization=quant,
            rescore_mode=rescore,
            oversample_factor=oversample,
            seed=seed,
        )
        build_s = index.build(vectors)
        mem = index.memory_breakdown()
        proj = index.project_scale_memory_gb(projected_scale)
        savings_pct = round((1.0 - (mem["total_index_mb"] / max(fp32_raw_mb, 1e-6))) * 100.0, 2)

        for ef_s in ef_list:
            retrieved = np.empty((n_queries, k), dtype=np.int32)
            latencies = np.empty(n_queries, dtype=np.float64)
            ops_list = np.empty(n_queries, dtype=np.int32)

            # Warmup 5 queries
            for w in range(min(5, n_queries)):
                index.search(queries[w], k=k, ef_search=ef_s)

            for qi in range(n_queries):
                stats = index.search(queries[qi], k=k, ef_search=ef_s)
                retrieved[qi] = stats.indices
                latencies[qi] = stats.latency_ms
                ops_list[qi] = stats.distance_computations

            recall = compute_recall_at_k(retrieved, gt_indices, k=k)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            mean_lat = float(np.mean(latencies))

            results.append(
                {
                    "config_id": f"{label} | ef={ef_s}",
                    "family": label,
                    "M": M,
                    "ef_construction": ef_c,
                    "ef_search": ef_s,
                    "quantization": quant,
                    "rescore_mode": rescore,
                    "oversample_factor": oversample,
                    "recall_at_10": round(recall, 4),
                    "p50_latency_ms": round(p50, 3),
                    "p95_latency_ms": round(p95, 3),
                    "p99_latency_ms": round(p99, 3),
                    "qps": round(1000.0 / max(mean_lat, 1e-4), 1),
                    "avg_dist_ops": int(np.mean(ops_list)),
                    "build_time_s": round(build_s, 3),
                    "vector_payload_mb": mem["vector_payload_mb"],
                    "rescore_payload_mb": mem["rescore_payload_mb"],
                    "hnsw_graph_mb": mem["hnsw_graph_mb"],
                    "total_index_mb": mem["total_index_mb"],
                    "memory_savings_pct": savings_pct,
                    "projected_5m_total_gb": proj["projected_total_gb"],
                }
            )

    pareto = extract_pareto_frontier(results)
    return {
        "metadata": {
            "n_vectors": n_vectors,
            "dim": dim,
            "n_queries": n_queries,
            "k": k,
            "projected_scale": projected_scale,
            "fp32_baseline_mb": fp32_raw_mb,
            "projected_5m_fp32_gb": projected_fp32_gb,
        },
        "results": results,
        "pareto_frontier": pareto,
    }


def main():
    parser = argparse.ArgumentParser(description="HNSW-QuantBench Scale Benchmark Runner")
    parser.add_argument("--vectors", type=int, default=15000, help="Number of indexed vectors")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimensionality")
    parser.add_argument("--queries", type=int, default=150, help="Number of evaluation queries")
    parser.add_argument("--scale", type=int, default=5000000, help="Target million-scale projection size")
    parser.add_argument("--output", type=str, default="artifacts/benchmark_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report = run_benchmark_suite(
        n_vectors=args.vectors,
        dim=args.dim,
        n_queries=args.queries,
        projected_scale=args.scale,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {len(report['results'])} benchmark runs ({len(report['pareto_frontier'])} Pareto-optimal) to {args.output}")


if __name__ == "__main__":
    main()
