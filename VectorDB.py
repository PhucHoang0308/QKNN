import pandas as pd
import numpy as np
import faiss
from abc import ABC, abstractmethod
import qknn as qknn
import importlib


importlib.reload(qknn)

# --- Strategy Pattern for Distance Calculation ---
class DistanceStrategy(ABC):
    @abstractmethod
    def distance(self, vec1, vec2):
        pass

class EuclideanDistance(DistanceStrategy):
    def calculate(self, vec1, vec2):
        return np.linalg.norm(np.asarray(vec1) - np.asarray(vec2))
    
    def distance(self, q, vectors):
        return np.array([self.calculate(q, v) for v in vectors])
    
class QuantumEuclideanDistance(DistanceStrategy):
    def distance(self, q, vectors):
        vectors_df = pd.DataFrame(vectors)
        N = len(vectors_df)
        d = len(vectors_df.columns) 
        vectors_df, q, _ = qknn.load_file(vectors_df, pd.DataFrame([q]), verbose=False, store=False)
        circuit, _, index_qubits_num, _, target_norm = qknn.build_qknn_circuit(vectors_df, q, N, d)
        probabilities = circuit()
        target_norm_squared = target_norm ** 2
        distances = qknn.compute_euclidean_distances(probabilities, index_qubits_num, target_norm_squared, len(vectors))
        return np.array([distances[j] for j in range(len(vectors))])

# You can add more strategies, e.g., CosineDistance, QuantumFidelityDistance, etc.

class FlatEmbedder:
    def embedding(self, data):
        return data

class VectorDatabase(ABC):
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = []
        self.value = []

    @abstractmethod
    def insert(self, param):
        pass

    @abstractmethod
    def drop(self, param):
        pass

    @abstractmethod
    def retrieve(self, param):
        pass

class KNNVectorDatabase(VectorDatabase):
    """
    Classical K-NN (L2 distance) with Strategy Pattern.
    """
    def __init__(self, embedder = FlatEmbedder(), distance_strategy=None):
        super().__init__(embedder)
        self.distance_strategy = distance_strategy or EuclideanDistance()

    def embed_it(self, vector):
        embedded_vector = self.embedder.embedding(vector)
        return np.asarray(embedded_vector, dtype=float)
    
    def insert_n(self, vectors, values):
        self.vectors = vectors
        self.value = values

    def insert(self, vector, value):
        embedded_vector = self.embed_it(vector)
        self.vectors.append(embedded_vector)
        self.value.append(value)

    def drop(self, idx: int):
        if 0 <= idx < len(self.vectors):
            self.vectors.pop(idx)
            self.value.pop(idx)

    def retrieve(self, query, k: int = 1):
        if not self.vectors:
            return []
        q = self.embed_it(query)
        dists = self.distance_strategy.distance(q, self.vectors)
        order = np.argsort(dists)[:k]
        return [(int(i), self.value[i], float(dists[i])) for i in order]
    
class HNSWVectorDatabase(VectorDatabase):
    """
    FAISS-based HNSW (Hierarchical Navigable Small World) vector database.
    """
    def __init__(self, embedder, dim=None, M=32, efConstruction=200):
        super().__init__(embedder)
        self.dim = dim
        self.index = None
        self.id_map = {}
        self.next_id = 0
        self.M = M
        self.efConstruction = efConstruction

    def _ensure_index(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.efConstruction

    def embed_it(self, vector):
        embedded_vector = self.embedder.embedding(vector)
        embedded_vector = np.asarray(embedded_vector, dtype=np.float32).reshape(1, -1)
        if self.index is None:
            self.dim = embedded_vector.shape[1]
            self._ensure_index(self.dim)
        return embedded_vector
    
    def insert_n(self, vectors, values):
        if len(vectors) != len(values):
            raise ValueError("Vectors and values must have the same length.")
        for vector, value in zip(vectors, values):
            self.insert(vector, value)

    def insert(self, vector, value):
        vec = self.embed_it(vector)
        self.index.add(vec)
        self.id_map[self.next_id] = value
        self.next_id += 1

    def drop(self, idx: int):
        raise NotImplementedError("FAISS HNSW index does not support delete by default.")

    def retrieve(self, query, k: int = 1):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.embed_it(query)
        distances, indices = self.index.search(q, k)
        results = []
        for i, d in zip(indices[0], distances[0]):
            if i in self.id_map:
                results.append((int(i), self.id_map[i], float(d)))
        return results

class LSHVectorDatabase(VectorDatabase):
    """
    FAISS-based LSH (Locality Sensitive Hashing) vector database.
    """
    def __init__(self, embedder, dim=None, n_bits=128):
        super().__init__(embedder)
        self.dim = dim
        self.index = None
        self.id_map = {}
        self.next_id = 0
        self.n_bits = n_bits

    def _ensure_index(self, dim):
        self.index = faiss.IndexLSH(dim, self.n_bits)

    def embed_it(self, vector):
        embedded_vector = self.embedder.embedding(vector)
        embedded_vector = np.asarray(embedded_vector, dtype=np.float32).reshape(1, -1)
        if self.index is None:
            self.dim = embedded_vector.shape[1]
            self._ensure_index(self.dim)
        return embedded_vector
    
    def insert_n(self, vectors, values):
        if len(vectors) != len(values):
            raise ValueError("Vectors and values must have the same length.")
        for vector, value in zip(vectors, values):
            self.insert(vector, value)

    def insert(self, vector, value):
        vec = self.embed_it(vector)
        self.index.add(vec)
        self.id_map[self.next_id] = value
        self.next_id += 1

    def drop(self, idx: int):
        raise NotImplementedError("FAISS LSH index does not support delete by default.")

    def retrieve(self, query, k: int = 1):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.embed_it(query)
        distances, indices = self.index.search(q, k)
        results = []
        for i, d in zip(indices[0], distances[0]):
            if i in self.id_map:
                results.append((int(i), self.id_map[i], float(d)))
        return results
