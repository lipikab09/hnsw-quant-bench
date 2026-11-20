from abc import ABC, abstractmethod
import numpy as np


class BaseQuantizer(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, vectors: np.ndarray) -> "BaseQuantizer":
        ...

    @abstractmethod
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def batch_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def single_distance(self, query: np.ndarray, code: np.ndarray) -> float:
        ...

    @abstractmethod
    def memory_bytes(self, codes: np.ndarray) -> int:
        ...


class FP32Quantizer(BaseQuantizer):
    name = "FP32 (Exact)"

    def fit(self, vectors: np.ndarray) -> "FP32Quantizer":
        self.dim = vectors.shape[1]
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def batch_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        diff = codes - query
        return np.einsum("ij,ij->i", diff, diff)

    def single_distance(self, query: np.ndarray, code: np.ndarray) -> float:
        diff = code - query
        return float(np.dot(diff, diff))

    def memory_bytes(self, codes: np.ndarray) -> int:
        return int(codes.nbytes)


class ScalarQuantizerSQ8(BaseQuantizer):
    name = "SQ8 (Scalar 8-bit)"

    def __init__(self, clip_percentile: float = 99.9):
        self.clip_percentile = clip_percentile
        self.mins: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.inv_scales: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizerSQ8":
        low_q = 100.0 - self.clip_percentile
        self.mins = np.percentile(vectors, low_q, axis=0).astype(np.float32)
        maxs = np.percentile(vectors, self.clip_percentile, axis=0).astype(np.float32)
        ranges = np.maximum(maxs - self.mins, 1e-6)
        self.scales = (ranges / 255.0).astype(np.float32)
        self.inv_scales = (255.0 / ranges).astype(np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        scaled = np.round((vectors - self.mins) * self.inv_scales)
        return np.ascontiguousarray(np.clip(scaled, 0, 255), dtype=np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) * self.scales + self.mins

    def batch_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        reconstructed = codes.astype(np.float32) * self.scales + self.mins
        diff = reconstructed - query
        return np.einsum("ij,ij->i", diff, diff)

    def single_distance(self, query: np.ndarray, code: np.ndarray) -> float:
        reconstructed = code.astype(np.float32) * self.scales + self.mins
        diff = reconstructed - query
        return float(np.dot(diff, diff))

    def memory_bytes(self, codes: np.ndarray) -> int:
        meta = (self.mins.nbytes + self.scales.nbytes) if self.mins is not None else 0
        return int(codes.nbytes + meta)


class BinaryQuantizerBQ(BaseQuantizer):
    name = "BQ (Binary 1-bit)"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.dim: int = 0
        self.rotation: np.ndarray | None = None
        self.inv_sqrt_d: float = 1.0

    def fit(self, vectors: np.ndarray) -> "BinaryQuantizerBQ":
        self.dim = vectors.shape[1]
        self.inv_sqrt_d = 1.0 / np.sqrt(float(self.dim))
        rng = np.random.default_rng(self.seed)
        q_mat, _ = np.linalg.qr(rng.standard_normal((self.dim, self.dim)).astype(np.float32))
        self.rotation = np.ascontiguousarray(q_mat, dtype=np.float32)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        rotated = vectors @ self.rotation
        bits = (rotated >= 0.0).astype(np.uint8)
        return np.ascontiguousarray(np.packbits(bits, axis=-1))

    def encode_query(self, query: np.ndarray) -> np.ndarray:
        rotated = query @ self.rotation
        bits = (rotated >= 0.0).astype(np.uint8)
        return np.packbits(bits)

    def _unpack_codes(self, codes: np.ndarray) -> np.ndarray:
        bits = np.unpackbits(codes, axis=-1)[..., : self.dim].astype(np.float32)
        return (bits * 2.0 - 1.0) * self.inv_sqrt_d

    def batch_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        q_rot = query @ self.rotation
        unpacked = self._unpack_codes(codes)
        q_norm_sq = float(np.dot(q_rot, q_rot))
        return np.maximum(q_norm_sq + 1.0 - 2.0 * (unpacked @ q_rot), 0.0)

    def single_distance(self, query: np.ndarray, code: np.ndarray) -> float:
        q_rot = query @ self.rotation
        unpacked = self._unpack_codes(code)
        q_norm_sq = float(np.dot(q_rot, q_rot))
        return max(q_norm_sq + 1.0 - 2.0 * float(np.dot(unpacked, q_rot)), 0.0)

    def memory_bytes(self, codes: np.ndarray) -> int:
        meta = self.rotation.nbytes if self.rotation is not None else 0
        return int(codes.nbytes + meta)
