"""Unified cross-domain tag space: IDF-weighted tag vectors plus soft matching between related tags.

Tag-to-tag similarity blends (a) embedding similarity of tag names, which aligns vocabularies across
providers ("sci fi" ~ "science fiction", "detective" ~ "crime"), and (b) PPMI co-occurrence, which links
tags that describe the same kind of work ("mecha" ~ "military").
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from ..util import unit


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    lengths = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return (sparse.diags(1.0 / np.maximum(lengths, 1e-12)) @ matrix).tocsr().astype(np.float32)


class TagSpace:
    def __init__(self, records: list[dict], name_vectors_fn=None, neighbors: int = 8, min_similarity: float = 0.55):
        frequency = Counter()
        for record in records:
            frequency.update(set(record["tags"]) | set(record["genres"]))
        self.names = sorted(name for name, count in frequency.items() if count >= 2)
        self.index = {name: position for position, name in enumerate(self.names)}
        rows, columns, values = [], [], []
        for row, record in enumerate(records):
            weights = {genre: 0.7 for genre in record["genres"]}
            weights.update(record["tags"])
            for name, weight in weights.items():
                column = self.index.get(name)
                if column is not None and weight > 0:
                    rows.append(row)
                    columns.append(column)
                    values.append(weight)
        shape = (len(records), len(self.names))
        raw = sparse.csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)), shape=shape)
        df = np.asarray((raw > 0).sum(axis=0)).ravel()
        self.idf = (np.log((len(records) + 1) / (df + 1)) + 1).astype(np.float32)
        self.matrix = _row_normalize(raw @ sparse.diags(self.idf))
        self.similarity = self._tag_similarity(raw, name_vectors_fn, neighbors, min_similarity)
        self.soft = _row_normalize(self.matrix @ self.similarity)

    def _tag_similarity(self, raw, name_vectors_fn, neighbors, min_similarity) -> sparse.csr_matrix:
        count = len(self.names)
        binary = (raw > 0).astype(np.float32)
        cooccurrence = (binary.T @ binary).toarray() if count <= 12000 else None
        blended = np.zeros((count, count), dtype=np.float32)
        weight = 0.0
        if cooccurrence is not None and count > 2:
            total = cooccurrence.sum()
            marginal = cooccurrence.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log(cooccurrence * total / (marginal @ marginal.T))
            ppmi = np.nan_to_num(np.maximum(pmi, 0), nan=0.0, posinf=0.0)
            rank = min(64, count - 2)
            left, singular, _ = svds(sparse.csr_matrix(ppmi), k=rank, random_state=7)
            vectors = unit(left * np.sqrt(singular))
            blended += vectors @ vectors.T
            weight += 1.0
        if name_vectors_fn is not None:
            vectors = unit(np.asarray(name_vectors_fn([f"Tag: {name}" for name in self.names]), dtype=np.float32))
            blended += 1.5 * (vectors @ vectors.T)
            weight += 1.5
        if weight == 0:
            return sparse.identity(count, dtype=np.float32, format="csr")
        blended /= weight
        np.fill_diagonal(blended, 0)
        rows, columns, values = list(range(count)), list(range(count)), [1.0] * count
        for row in range(count):
            top = np.argpartition(-blended[row], min(neighbors, count - 1))[:neighbors]
            for column in top:
                if blended[row, column] >= min_similarity:
                    rows.append(row)
                    columns.append(int(column))
                    values.append(float(blended[row, column]) ** 2)
        return sparse.csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)), shape=(count, count))

    def related(self, name: str, limit: int = 8) -> list[tuple[str, float]]:
        position = self.index.get(name)
        if position is None:
            return []
        row = self.similarity[position]
        order = np.argsort(-row.data)
        return [(self.names[row.indices[i]], round(float(row.data[i]), 3)) for i in order if row.indices[i] != position][:limit]
