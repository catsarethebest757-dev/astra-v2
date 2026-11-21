"""Item-side index: every per-title representation the ranker and engine need, persisted to disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ASPECTS
from .features.graph import Graph, edge_list
from .features.structure import Structure
from .features.tags import TagSpace
from .features.text import BM25, aspect_texts
from .store import load_array, load_json, load_sparse, save_array, save_json, save_sparse
from .util import log


class Index:
    """Holds records plus aspect embeddings, lexical/tag/creator matrices, and the recommendation graph."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.ids = {record["id"]: position for position, record in enumerate(records)}
        self.size = len(records)
        self.kind = np.array([record["kind"] for record in records])
        self.adult = np.array([record["adult"] for record in records])
        self.year = np.array([record["year"] for record in records], dtype=np.int32)
        self.rating = np.array([record["rating"] for record in records], dtype=np.float32)
        self.pop = np.array([record.get("pop_pct", 0.5) for record in records], dtype=np.float32)
        self.quality = np.array([record.get("quality", 0.0) for record in records], dtype=np.float32)
        self.has_synopsis = np.array([len(record["synopsis"]) >= 40 for record in records])
        franchises = {name: position for position, name in
                      enumerate(sorted({record.get("franchise", record["id"]) for record in records}))}
        self.franchise = np.array([franchises[record.get("franchise", record["id"])] for record in records])
        self.entry = np.array([record.get("entry", True) for record in records])
        self.aspects: dict[str, np.ndarray] = {}
        self._dense32: dict[str, np.ndarray] = {}
        self.adapter: np.ndarray | None = None
        self.bm25: BM25 | None = None
        self.tags: TagSpace | None = None
        self.structure: Structure | None = None
        self.graph: Graph | None = None

    # ----- construction -----
    def embed(self, encoder) -> None:
        texts = [aspect_texts(record) for record in self.records]
        for aspect in ASPECTS:
            log(f"Aspect '{aspect}':")
            self.aspects[aspect] = encoder.encode_documents([text[aspect] for text in texts])

    def build_sparse(self, encoder=None) -> None:
        log("Building BM25, tag space, and creator matrices...")
        self.bm25 = BM25(self.records)
        name_fn = (lambda names: encoder.encode_documents(names)) if encoder is not None else None
        self.tags = TagSpace(self.records, name_vectors_fn=name_fn)
        self.structure = Structure(self.records)

    def all_edges(self) -> dict[tuple[int, int], int]:
        return edge_list(self.records, self.ids)

    def set_graph(self, edges: dict[tuple[int, int], int]) -> None:
        self.graph = Graph(self.size, edges)

    def dense32(self, aspect: str) -> np.ndarray:
        """Float32 copy for fast CPU matrix-vector products (float16 matmul is slow on CPU)."""
        cached = self._dense32.get(aspect)
        if cached is None:
            cached = self._dense32[aspect] = np.ascontiguousarray(self.aspects[aspect], dtype=np.float32)
        return cached

    def adapter_input(self, rows) -> np.ndarray:
        return np.concatenate([self.aspects[aspect][rows] for aspect in ASPECTS], axis=1).astype(np.float32)

    # ----- persistence -----
    def save(self, directory: Path) -> None:
        for aspect, vectors in self.aspects.items():
            save_array(directory, f"aspect-{aspect}", vectors)
        if self.adapter is not None:
            save_array(directory, "adapter", self.adapter.astype(np.float16))
        save_sparse(directory, "bm25", self.bm25.matrix)
        save_array(directory, "bm25-idf", self.bm25.idf)
        save_array(directory, "bm25-self", self.bm25.self_scores)
        save_json(directory, "bm25-vocab", sorted(self.bm25.vocabulary, key=self.bm25.vocabulary.get))
        save_sparse(directory, "tags", self.tags.matrix)
        save_sparse(directory, "tags-soft", self.tags.soft)
        save_sparse(directory, "tags-sim", self.tags.similarity)
        save_array(directory, "tags-idf", self.tags.idf)
        save_json(directory, "tags-names", self.tags.names)
        save_sparse(directory, "creators", self.structure.creators)
        save_sparse(directory, "studios", self.structure.studios)

    @classmethod
    def load(cls, records: list[dict], directory: Path) -> "Index":
        index = cls(records)
        for aspect in ASPECTS:
            index.aspects[aspect] = load_array(directory, f"aspect-{aspect}")
        if (directory / "adapter.npy").exists():
            index.adapter = load_array(directory, "adapter").astype(np.float32)
        bm25 = BM25.__new__(BM25)
        bm25.matrix = load_sparse(directory, "bm25")
        bm25.idf = load_array(directory, "bm25-idf")
        bm25.self_scores = load_array(directory, "bm25-self")
        bm25.vocabulary = {token: position for position, token in enumerate(load_json(directory, "bm25-vocab"))}
        index.bm25 = bm25
        tags = TagSpace.__new__(TagSpace)
        tags.matrix = load_sparse(directory, "tags")
        tags.soft = load_sparse(directory, "tags-soft")
        tags.similarity = load_sparse(directory, "tags-sim")
        tags.idf = load_array(directory, "tags-idf")
        tags.names = load_json(directory, "tags-names")
        tags.index = {name: position for position, name in enumerate(tags.names)}
        index.tags = tags
        index.structure = Structure(records)
        index.structure.creators = load_sparse(directory, "creators")
        index.structure.studios = load_sparse(directory, "studios")
        index.set_graph(index.all_edges())
        return index


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest scores (descending), optionally restricted to mask."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


def adapter_scores(index: Index, seed: int) -> np.ndarray:
    return index.adapter @ index.adapter[seed]


def dense_scores(index: Index, aspect: str, vector: np.ndarray) -> np.ndarray:
    return index.dense32(aspect) @ vector.astype(np.float32)
