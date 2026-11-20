# ⚡ HNSW-QuantBench: Million-Scale Vector Search & Adaptive Quantization Engine

A production-engineered **Vector Search Benchmarking & Two-Stage Retrieval Observatory** designed to measure and optimize the systems trade-offs across **FP32**, **8-bit Scalar Quantization (SQ8)**, **1-bit Orthogonal Binary Quantization (BQ)**, and **Oversampled Two-Stage Rescoring** at million-vector scale.

> **Resume Highlight:** *Benchmarked vector search across millions of embeddings, tuning HNSW parameters (`M`, `efConstruction`, `efSearch`) and scalar/binary quantization with two-stage rescoring to balance Recall@10, p95 tail latency, and memory usage.*

---

## 🚨 Why Systems Break Between 10K and 5M Embeddings

A retrieval setup that performs effortlessly at **10,000 vectors** hits severe hardware bottlenecks at **5,000,000+ vectors** (768-D):

| Scale Regime | Raw FP32 Payload | HNSW Graph Overhead (`M=16`) | Hardware Bottleneck | Engineering Solution |
| :--- | :---: | :---: | :--- | :--- |
| **10,000 Vectors** | **30.7 MB** | ~1.3 MB | Fits inside modern CPU L3 cache; brute-force or unquantized HNSW completes in `<1 ms`. | Standard FP32 HNSW (`M=16`, `efSearch=64`). |
| **5,000,000 Vectors** | **15.36 GB** | ~0.64 GB | Exceeds L3 cache & standard container RAM; random graph hops trigger DRAM stalls & TLB misses, spiking **p95 tail latency**. | **SQ8 (4x compression)** or **BQ + 4x SQ8 Rescoring (32x primary payload compression, ~78% total RAM saved)**. |

### Core Quantization & Graph Techniques Implemented
1. **HNSW Multi-Layer Navigable Small-World Index (`src/engine/hnsw_index.py`)**:
   - Configurable graph connectivity (`M`, `M0 = 2M`), build beam width (`efConstruction`), and query beam width (`efSearch`).
   - Exact byte-level accounting separating **Primary Vector Codes**, **Stage-2 Rescore Payloads**, and **HNSW Adjacency Links**.
   - High-precision neighbor pruning during graph construction to preserve navigable small-world topology even when routing over 1-bit codes.
2. **8-Bit Scalar Quantization (`SQ8`) with Asymmetric Distance Computation (`ADC`)**:
   - Percentile-clipped (`99.9%`) per-dimension affine mapping into `uint8` (`4.0x` memory reduction).
   - Keeps the query vector in `FP32` during distance computation (`ADC`) to preserve `>98%` Recall@10.
3. **1-Bit Binary Quantization (`BQ`) with Orthogonal QR Rotation**:
   - Applies a deterministic orthogonal projection ($R^T R = I$) prior to sign-bit packing (`32x` vector compression) to balance variance across dimensions and eliminate correlation collapse on clustered embeddings.
4. **Two-Stage Coarse-to-Fine Oversampled Reranking**:
   - **Stage 1**: Traverses the HNSW graph using compact 1-bit codes with widened beam search (`efSearch`).
   - **Stage 2**: Over-fetches `k * oversample_factor` candidates and rescores them with `SQ8` ADC, recovering high Recall@10 while cutting total index RAM by **~78%–90%**.
5. **Multi-Objective Pareto Frontier & SLO Solver (`src/benchmark/metrics.py`)**:
   - Extracts non-dominated configurations maximizing **Recall@10** while minimizing **p95 Latency (ms)** and **Index RAM (GB)**, recommending optimal configs for user-defined Service Level Objectives (SLOs).

---

## 📁 Repository Structure

```text
hnsw-quant-bench/
├── artifacts/
│   └── benchmark_results.json      # Pre-computed grid benchmark telemetry
├── dashboard/
│   └── app.py                      # Interactive Streamlit Visual Observatory
├── src/
│   ├── api/
│   │   └── server.py               # FastAPI inference, benchmark & SLO recommendation API
│   ├── benchmark/
│   │   ├── metrics.py              # Recall@K & multi-objective Pareto frontier extraction
│   │   └── runner.py               # CLI automated grid-search benchmark harness
│   └── engine/
│       ├── dataset.py              # Anisotropic spherical embedding generator & exact BLAS ground truth
│       ├── hnsw_index.py           # Multi-layer HNSW graph & two-stage oversampled reranker
│       └── quantizers.py           # FP32, SQ8 (ADC), and Orthogonal BQ quantizers
├── tests/
│   └── test_vector_engine.py       # Pytest verification suite (quantization, recall, Pareto, FastAPI)
├── Dockerfile                      # Production container image
├── requirements.txt                # Python dependencies
└── README.md