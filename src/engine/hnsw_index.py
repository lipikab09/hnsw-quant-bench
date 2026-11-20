import heapq
import math
import time
from dataclasses import dataclass
import numpy as np
from src.engine.quantizers import BaseQuantizer, BinaryQuantizerBQ, FP32Quantizer, ScalarQuantizerSQ8


@dataclass
class SearchStats:
    indices: np.ndarray
    distances: np.ndarray
    latency_ms: float
    distance_computations: int


class HNSWQuantIndex:
    """
    Hierarchical Navigable Small World (HNSW) graph index supporting pluggable
    vector quantization (FP32, SQ8, BQ) and Two-Stage Oversampled Rescoring.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        quantization: str = "fp32",
        rescore_mode: str = "none",
        oversample_factor: int = 4,
        seed: int = 42,
    ):
        self.dim = dim
        self.M = M
        self.M0 = M * 2
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        self.quantization = quantization.lower()
        self.rescore_mode = rescore_mode.lower()
        self.oversample_factor = max(1, oversample_factor)
        self.rng = np.random.default_rng(seed)
        self.level_mult = 1.0 / math.log(max(M, 2))

        if self.quantization == "fp32":
            self.quantizer: BaseQuantizer = FP32Quantizer()
        elif self.quantization == "sq8":
            self.quantizer = ScalarQuantizerSQ8()
        elif self.quantization == "bq":
            self.quantizer = BinaryQuantizerBQ()
        else:
            raise ValueError(f"Unsupported quantization: {quantization}")

        self.rescore_quantizer: BaseQuantizer | None = None
        if self.rescore_mode == "sq8":
            self.rescore_quantizer = ScalarQuantizerSQ8()
        elif self.rescore_mode == "fp32":
            self.rescore_quantizer = FP32Quantizer()

        self.codes: np.ndarray | None = None
        self.rescore_codes: np.ndarray | None = None
        self.graphs: list[list[np.ndarray]] = []
        self.levels: np.ndarray | None = None
        self.enter_point: int = -1
        self.max_level: int = -1
        self.build_time_s: float = 0.0

    def _sample_level(self) -> int:
        r = max(float(self.rng.random()), 1e-12)
        return int(-math.log(r) * self.level_mult)

    def build(self, vectors: np.ndarray) -> float:
        t0 = time.perf_counter()
        n = len(vectors)
        self.quantizer.fit(vectors)
        self.codes = self.quantizer.encode(vectors)

        if self.rescore_quantizer is not None:
            self.rescore_quantizer.fit(vectors)
            self.rescore_codes = self.rescore_quantizer.encode(vectors)

        build_q = self.rescore_quantizer if self.rescore_quantizer is not None else self.quantizer
        build_c = self.rescore_codes if self.rescore_codes is not None else self.codes

        self.levels = np.array([self._sample_level() for _ in range(n)], dtype=np.int32)
        self.max_level = int(self.levels.max()) if n > 0 else -1
        self.graphs = [[np.empty(0, dtype=np.int32) for _ in range(n)] for _ in range(self.max_level + 1)]

        if n == 0:
            return 0.0

        self.enter_point = 0
        curr_max_level = int(self.levels[0])

        for i in range(1, n):
            q_vec = vectors[i]
            lvl = int(self.levels[i])
            ep = self.enter_point

            for lc in range(curr_max_level, lvl, -1):
                ep, _ = self._search_layer_greedy(q_vec, ep, lc, quantizer=build_q, codes=build_c)

            for lc in range(min(lvl, curr_max_level), -1, -1):
                candidates = self._search_layer_beam(
                    q_vec, [ep], self.ef_construction, lc, quantizer=build_q, codes=build_c
                )[0]
                max_links = self.M0 if lc == 0 else self.M
                neighbors = self._select_neighbors_heuristic(candidates, max_links)
                self.graphs[lc][i] = neighbors

                for nb in neighbors:
                    nb_int = int(nb)
                    existing = self.graphs[lc][nb_int]
                    if len(existing) == 0:
                        self.graphs[lc][nb_int] = np.array([i], dtype=np.int32)
                    else:
                        combined = np.append(existing, np.int32(i))
                        if len(combined) > max_links:
                            nb_dists = build_q.batch_distances(vectors[nb_int], build_c[combined])
                            keep = np.argsort(nb_dists)[:max_links]
                            combined = combined[keep]
                        self.graphs[lc][nb_int] = combined

                if candidates:
                    ep = candidates[0][1]

            if lvl > curr_max_level:
                curr_max_level = lvl
                self.enter_point = i

        self.max_level = curr_max_level
        self.build_time_s = time.perf_counter() - t0
        return self.build_time_s

    def _search_layer_greedy(
        self,
        query: np.ndarray,
        ep: int,
        layer: int,
        quantizer: BaseQuantizer | None = None,
        codes: np.ndarray | None = None,
    ) -> tuple[int, int]:
        q_impl = quantizer or self.quantizer
        c_impl = codes if codes is not None else self.codes
        curr = ep
        curr_dist = q_impl.single_distance(query, c_impl[curr])
        dist_ops = 1
        while True:
            changed = False
            neighbors = self.graphs[layer][curr]
            if len(neighbors) > 0:
                dists = q_impl.batch_distances(query, c_impl[neighbors])
                dist_ops += len(neighbors)
                min_idx = int(np.argmin(dists))
                if float(dists[min_idx]) < curr_dist:
                    curr_dist = float(dists[min_idx])
                    curr = int(neighbors[min_idx])
                    changed = True
            if not changed:
                break
        return curr, dist_ops

    def _search_layer_beam(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
        quantizer: BaseQuantizer | None = None,
        codes: np.ndarray | None = None,
    ) -> tuple[list[tuple[float, int]], int]:
        q_impl = quantizer or self.quantizer
        c_impl = codes if codes is not None else self.codes
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        top_results: list[tuple[float, int]] = []
        dist_ops = 0

        ep_arr = np.array(entry_points, dtype=np.int32)
        ep_dists = q_impl.batch_distances(query, c_impl[ep_arr])
        dist_ops += len(ep_arr)

        for ep, d in zip(entry_points, ep_dists):
            d_f = float(d)
            heapq.heappush(candidates, (d_f, ep))
            heapq.heappush(top_results, (-d_f, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_dist = -top_results[0][0]
            if c_dist > worst_dist and len(top_results) >= ef:
                break

            neighbors = self.graphs[layer][c_id]
            unvisited = [int(nb) for nb in neighbors if int(nb) not in visited]
            if not unvisited:
                continue

            visited.update(unvisited)
            u_arr = np.array(unvisited, dtype=np.int32)
            u_dists = q_impl.batch_distances(query, c_impl[u_arr])
            dist_ops += len(u_arr)

            for nb_id, d in zip(unvisited, u_dists):
                d_f = float(d)
                worst_dist = -top_results[0][0]
                if d_f < worst_dist or len(top_results) < ef:
                    heapq.heappush(candidates, (d_f, nb_id))
                    heapq.heappush(top_results, (-d_f, nb_id))
                    if len(top_results) > ef:
                        heapq.heappop(top_results)

        ordered = sorted([(-neg_d, idx) for neg_d, idx in top_results], key=lambda x: x[0])
        return ordered, dist_ops

    @staticmethod
    def _select_neighbors_heuristic(
        candidates: list[tuple[float, int]], max_links: int
    ) -> np.ndarray:
        selected = [idx for _, idx in candidates[:max_links]]
        return np.array(selected, dtype=np.int32)

    def search(self, query: np.ndarray, k: int = 10, ef_search: int | None = None) -> SearchStats:
        t0 = time.perf_counter()
        ef = max(ef_search or self.ef_search, k)
        if self.rescore_quantizer is not None:
            ef = max(ef, k * self.oversample_factor)

        ep = self.enter_point
        total_ops = 0
        for lc in range(self.max_level, 0, -1):
            ep, ops = self._search_layer_greedy(query, ep, lc)
            total_ops += ops

        candidates, ops = self._search_layer_beam(query, [ep], ef, 0)
        total_ops += ops

        if self.rescore_quantizer is not None and self.rescore_codes is not None:
            cand_ids = np.array([idx for _, idx in candidates], dtype=np.int32)
            rescored_dists = self.rescore_quantizer.batch_distances(query, self.rescore_codes[cand_ids])
            total_ops += len(cand_ids)
            order = np.argsort(rescored_dists)[:k]
            final_ids = cand_ids[order]
            final_dists = rescored_dists[order]
        else:
            top_k = candidates[:k]
            final_ids = np.array([idx for _, idx in top_k], dtype=np.int32)
            final_dists = np.array([d for d, _ in top_k], dtype=np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SearchStats(
            indices=final_ids,
            distances=final_dists,
            latency_ms=latency_ms,
            distance_computations=total_ops,
        )

    def memory_breakdown(self) -> dict[str, float]:
        vec_bytes = self.quantizer.memory_bytes(self.codes) if self.codes is not None else 0
        rescore_bytes = (
            self.rescore_quantizer.memory_bytes(self.rescore_codes)
            if (self.rescore_quantizer is not None and self.rescore_codes is not None)
            else 0
        )
        graph_bytes = 0
        for layer in self.graphs:
            for adj in layer:
                graph_bytes += int(adj.nbytes)

        total_bytes = vec_bytes + rescore_bytes + graph_bytes
        return {
            "vector_payload_mb": round(vec_bytes / (1024 ** 2), 4),
            "rescore_payload_mb": round(rescore_bytes / (1024 ** 2), 4),
            "hnsw_graph_mb": round(graph_bytes / (1024 ** 2), 4),
            "total_index_mb": round(total_bytes / (1024 ** 2), 4),
        }

    def project_scale_memory_gb(self, target_vectors: int = 5_000_000) -> dict[str, float]:
        n_curr = max(len(self.codes) if self.codes is not None else 1, 1)
        mem = self.memory_breakdown()
        scale_factor = target_vectors / n_curr
        vec_gb = (mem["vector_payload_mb"] * scale_factor) / 1024.0
        rescore_gb = (mem["rescore_payload_mb"] * scale_factor) / 1024.0
        graph_gb = (mem["hnsw_graph_mb"] * scale_factor) / 1024.0
        return {
            "target_vectors": target_vectors,
            "projected_vector_gb": round(vec_gb, 3),
            "projected_rescore_gb": round(rescore_gb, 3),
            "projected_graph_gb": round(graph_gb, 3),
            "projected_total_gb": round(vec_gb + rescore_gb + graph_gb, 3),
        }
