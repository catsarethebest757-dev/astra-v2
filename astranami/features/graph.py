"""Collaborative graph from community recommendation votes ("fans of A recommend B")."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..util import unit


def edge_list(records: list[dict], ids: dict[str, int]) -> dict[tuple[int, int], int]:
    """Undirected edges keyed (low, high) with the larger vote count of either direction."""
    edges: dict[tuple[int, int], int] = {}
    for record in records:
        source = ids[record["id"]]
        for target_id, votes in record["recs"].items():
            target = ids.get(target_id)
            if target is None or target == source:
                continue
            key = (min(source, target), max(source, target))
            edges[key] = max(edges.get(key, 0), int(votes))
    return edges


class Graph:
    def __init__(self, size: int, edges: dict[tuple[int, int], int], dimensions: int = 64):
        self.size = size
        if edges:
            pairs = np.array(list(edges.keys()), dtype=np.int64)
            weights = np.log1p(np.array(list(edges.values()), dtype=np.float32))
            rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
            columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
            data = np.concatenate([weights, weights])
        else:
            rows = columns = np.zeros(0, dtype=np.int64)
            data = np.zeros(0, dtype=np.float32)
        self.adjacency = sparse.csr_matrix((data, (rows, columns)), shape=(size, size), dtype=np.float32)
        self.votes = sparse.csr_matrix(
            (np.concatenate([np.array(list(edges.values()), dtype=np.float32)] * 2) if edges else data,
             (rows, columns)), shape=(size, size))
        degree = np.asarray(self.adjacency.sum(axis=1)).ravel()
        self.degree = degree
        self.has_edges = degree > 0
        self.transition = (sparse.diags(1.0 / np.maximum(degree, 1e-12)) @ self.adjacency).tocsr()
        self.max_weight = float(data.max()) if len(data) else 1.0
        self.embedding = self._spectral(dimensions)

    def _spectral(self, dimensions: int) -> np.ndarray:
        active = np.flatnonzero(self.has_edges)
        embedding = np.zeros((self.size, dimensions), dtype=np.float32)
        if len(active) <= dimensions + 2:
            return embedding
        scale = 1.0 / np.sqrt(np.maximum(self.degree[active], 1e-12))
        sub = self.adjacency[active][:, active]
        normalized = sparse.diags(scale) @ sub @ sparse.diags(scale)
        # ARPACK's exact sparse SVD can take minutes on the full AniList graph.
        # A bounded randomized range finder is enough for this similarity feature.
        width = min(len(active), dimensions + 16)
        rng = np.random.default_rng(11)
        basis = rng.standard_normal((len(active), width)).astype(np.float32)
        for _ in range(3):
            basis, _ = np.linalg.qr(normalized @ basis, mode="reduced")
        reduced = basis.T @ normalized
        small_left, singular, _ = np.linalg.svd(reduced, full_matrices=False)
        left = basis @ small_left[:, :dimensions]
        singular = singular[:dimensions]
        embedding[active] = unit((left * np.sqrt(np.abs(singular))).astype(np.float32))
        return embedding

    def seed_signals(self, seed: int, candidates: np.ndarray) -> np.ndarray:
        """Per-candidate [direct edge weight, 2-hop random-walk probability, spectral cosine]."""
        output = np.zeros((len(candidates), 3), dtype=np.float32)
        if not self.has_edges[seed]:
            return output
        direct = self.adjacency[seed]
        one_hop = self.transition[seed]
        two_hop = (one_hop @ self.transition).toarray().ravel()
        output[:, 0] = direct[:, candidates].toarray().ravel() / self.max_weight
        output[:, 1] = np.sqrt(two_hop[candidates])
        output[:, 2] = self.embedding[candidates] @ self.embedding[seed]
        return output

    def personalized_scores(self, seed: int) -> np.ndarray:
        """One- and two-hop walk scores from a seed, with the seed itself zeroed."""
        if not self.has_edges[seed]:
            return np.zeros(self.size, dtype=np.float32)
        scores = (self.transition[seed] + self.transition[seed] @ self.transition * 0.5).toarray().ravel()
        scores[seed] = 0
        return scores.astype(np.float32)

    def votes_between(self, left: int, right: int) -> int:
        return int(self.votes[left, right]) if self.votes.nnz else 0

    def neighbors(self, seed: int, limit: int = 60) -> np.ndarray:
        if not self.has_edges[seed]:
            return np.zeros(0, dtype=np.int64)
        scores = (self.transition[seed] + self.transition[seed] @ self.transition * 0.5).toarray().ravel()
        scores[seed] = 0
        top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        return top[scores[top] > 0]
