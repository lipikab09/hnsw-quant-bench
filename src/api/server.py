from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from src.benchmark.runner import run_benchmark_suite
from src.engine.dataset import generate_clustered_embeddings
from src.engine.hnsw_index import HNSWQuantIndex

app = FastAPI(
    title="HNSW-QuantBench Vector Search & Telemetry API",
    version="1.0.0",
    description="Production multi-quantization HNSW retrieval engine with SLO-aware Pareto routing.",
)


class BenchmarkRequest(BaseModel):
    n_vectors: int = Field(default=5000, ge=500, le=200000)
    dim: int = Field(default=128, ge=32, le=1536)
    n_queries: int = Field(default=50, ge=10, le=1000)
    projected_scale: int = Field(default=5_000_000, ge=100_000)


class SLORequest(BaseModel):
    min_recall: float = Field(default=0.95, ge=0.5, le=1.0)
    max_p95_ms: float = Field(default=5.0, gt=0.0)
    max_5m_memory_gb: float = Field(default=4.0, gt=0.1)


_STATE: dict = {"last_report": None, "active_index": None, "vectors": None}


@app.get("/health")
def health():
    return {"status": "healthy", "engine": "HNSW-QuantBench", "indexed": _STATE["active_index"] is not None}


@app.post("/benchmark/run")
def trigger_benchmark(req: BenchmarkRequest):
    report = run_benchmark_suite(
        n_vectors=req.n_vectors,
        dim=req.dim,
        n_queries=req.n_queries,
        projected_scale=req.projected_scale,
    )
    _STATE["last_report"] = report
    return report


@app.post("/recommend")
def recommend_configuration(slo: SLORequest):
    if _STATE["last_report"] is None:
        _STATE["last_report"] = run_benchmark_suite(n_vectors=4000, dim=128, n_queries=40)

    candidates = [
        r
        for r in _STATE["last_report"]["results"]
        if r["recall_at_10"] >= slo.min_recall
        and r["p95_latency_ms"] <= slo.max_p95_ms
        and r["projected_5m_total_gb"] <= slo.max_5m_memory_gb
    ]
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No configuration satisfies all SLO constraints simultaneously. Relax recall or memory budget.",
        )
    best = sorted(candidates, key=lambda x: (-x["recall_at_10"], x["projected_5m_total_gb"], x["p95_latency_ms"]))[0]
    return {"recommended_config": best, "matching_count": len(candidates)}


@app.post("/index/build-demo")
def build_demo_index(n_vectors: int = 3000, dim: int = 128, quantization: str = "bq", rescore_mode: str = "sq8"):
    vecs, _ = generate_clustered_embeddings(n_vectors=n_vectors, dim=dim)
    idx = HNSWQuantIndex(dim=dim, M=16, ef_construction=80, ef_search=64, quantization=quantization, rescore_mode=rescore_mode)
    build_s = idx.build(vecs)
    _STATE["active_index"] = idx
    _STATE["vectors"] = vecs
    return {"build_time_s": round(build_s, 3), "memory": idx.memory_breakdown(), "scale_5m": idx.project_scale_memory_gb(5_000_000)}
