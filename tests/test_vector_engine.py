import numpy as np
from fastapi.testclient import TestClient
from src.api.server import app
from src.benchmark.metrics import compute_recall_at_k, extract_pareto_frontier
from src.engine.dataset import compute_exact_ground_truth, generate_clustered_embeddings, generate_queries
from src.engine.hnsw_index import HNSWQuantIndex
from src.engine.quantizers import BinaryQuantizerBQ, ScalarQuantizerSQ8


def test_sq8_quantization_reconstruction_and_compression():
    vecs, _ = generate_clustered_embeddings(n_vectors=500, dim=64, seed=7)
    sq8 = ScalarQuantizerSQ8().fit(vecs)
    codes = sq8.encode(vecs)
    assert codes.dtype == np.uint8
    assert sq8.memory_bytes(codes) < vecs.nbytes * 0.30
    reconstructed = sq8.decode(codes)
    mse = float(np.mean((vecs - reconstructed) ** 2))
    assert mse < 1e-4


def test_bq_bitpacking_and_hamming_ordering():
    vecs, _ = generate_clustered_embeddings(n_vectors=400, dim=128, seed=11)
    bq = BinaryQuantizerBQ().fit(vecs)
    codes = bq.encode(vecs)
    assert codes.shape == (400, 16)
    dists = bq.batch_distances(vecs[0], codes)
    assert int(np.argmin(dists)) == 0
    assert dists[0] < float(np.median(dists))
    assert np.all(dists >= 0.0)


def test_hnsw_high_recall_and_two_stage_rescoring():
    vecs, _ = generate_clustered_embeddings(n_vectors=1200, dim=256, n_clusters=16, seed=21)
    queries = generate_queries(vecs, n_queries=30, seed=99)
    gt_idx, _ = compute_exact_ground_truth(vecs, queries, k=10)

    idx_fp32 = HNSWQuantIndex(dim=256, M=16, ef_construction=80, ef_search=64, quantization="fp32")
    idx_fp32.build(vecs)

    idx_bq_raw = HNSWQuantIndex(dim=256, M=16, ef_construction=80, ef_search=128, quantization="bq", rescore_mode="none")
    idx_bq_raw.build(vecs)

    idx_hybrid = HNSWQuantIndex(
        dim=256, M=16, ef_construction=80, ef_search=128, quantization="bq", rescore_mode="sq8", oversample_factor=8
    )
    idx_hybrid.build(vecs)

    ret_fp32 = np.array([idx_fp32.search(q, k=10).indices for q in queries])
    ret_raw = np.array([idx_bq_raw.search(q, k=10).indices for q in queries])
    ret_hyb = np.array([idx_hybrid.search(q, k=10).indices for q in queries])

    rec_fp32 = compute_recall_at_k(ret_fp32, gt_idx, k=10)
    rec_raw = compute_recall_at_k(ret_raw, gt_idx, k=10)
    rec_hyb = compute_recall_at_k(ret_hyb, gt_idx, k=10)

    assert rec_fp32 >= 0.92
    assert rec_hyb > rec_raw
    assert rec_hyb >= 0.85
    assert idx_hybrid.memory_breakdown()["total_index_mb"] < idx_fp32.memory_breakdown()["total_index_mb"]


def test_pareto_frontier_extraction():
    sample = [
        {"config_id": "A", "recall_at_10": 0.98, "p95_latency_ms": 2.0, "total_index_mb": 50.0},
        {"config_id": "B_dominated", "recall_at_10": 0.90, "p95_latency_ms": 3.0, "total_index_mb": 60.0},
        {"config_id": "C_fast_small", "recall_at_10": 0.93, "p95_latency_ms": 0.8, "total_index_mb": 12.0},
    ]
    pareto = extract_pareto_frontier(sample)
    ids = {x["config_id"] for x in pareto}
    assert ids == {"A", "C_fast_small"}


def test_fastapi_endpoints():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    bench = client.post("/benchmark/run", json={"n_vectors": 600, "dim": 64, "n_queries": 15, "projected_scale": 5000000})
    assert bench.status_code == 200
    data = bench.json()
    assert len(data["results"]) > 5
    rec = client.post("/recommend", json={"min_recall": 0.80, "max_p95_ms": 50.0, "max_5m_memory_gb": 10.0})
    assert rec.status_code == 200
