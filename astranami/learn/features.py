"""Seed-to-candidate pair features and candidate generation, shared by training, evaluation, and serving."""
from __future__ import annotations

import numpy as np

from ..config import ASPECTS
from ..index import Index, adapter_scores, dense_scores, top_k

FEATURES = [
    "cos_plot", "cos_themes", "cos_tone", "cos_whole", "cos_adapter", "bm25", "tag_cos", "soft_tag_cos",
    "creator_sim", "studio_sim", "year_gap", "form_match", "length_ratio", "source_match",
    "graph_direct", "graph_2hop", "graph_spectral", "has_graph",
    "cand_pop", "cand_quality", "seed_pop", "cand_has_synopsis",
]
GRAPH_FEATURES = [FEATURES.index(name) for name in ("graph_direct", "graph_2hop", "graph_spectral", "has_graph")]
ASPECT_FEATURES = {aspect: FEATURES.index(f"cos_{aspect}") for aspect in ASPECTS}


def pair_features(index: Index, seed: int, candidates: np.ndarray, use_graph: bool = True) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    output = np.zeros((len(candidates), len(FEATURES)), dtype=np.float32)
    if not len(candidates):
        return output
    for aspect, column in ASPECT_FEATURES.items():
        matrix = index.dense32(aspect)
        output[:, column] = matrix[candidates] @ matrix[seed]
    if index.adapter is not None:
        output[:, 4] = index.adapter[candidates] @ index.adapter[seed]
    output[:, 5] = np.minimum(index.bm25.item_scores(seed, candidates), 2.0)
    output[:, 6] = (index.tags.matrix[candidates] @ index.tags.matrix[seed].T).toarray().ravel()
    output[:, 7] = (index.tags.soft[candidates] @ index.tags.soft[seed].T).toarray().ravel()
    output[:, 8:14] = index.structure.seed_signals(seed, candidates)
    if use_graph and index.graph is not None and index.graph.has_edges[seed]:
        output[:, 14:17] = index.graph.seed_signals(seed, candidates)
        output[:, 17] = index.graph.has_edges[candidates]
    output[:, 18] = index.pop[candidates]
    output[:, 19] = index.quality[candidates] / 10.0
    output[:, 20] = index.pop[seed]
    output[:, 21] = index.has_synopsis[candidates]
    return output


DEFAULT_POOL = {"adapter": 300, "whole": 120, "themes": 80, "graph": 60, "bm25": 60}


def candidate_channels(index: Index, seed: int, eligible: np.ndarray, sizes: dict | None = None,
                       use_graph: bool = True) -> dict[str, np.ndarray]:
    """The strongest neighbors of a seed under each retrieval view, keyed by view name."""
    sizes = sizes or DEFAULT_POOL
    mask = eligible.copy()
    mask[seed] = False
    channels = {}
    if index.adapter is not None:
        channels["adapter"] = top_k(adapter_scores(index, seed), sizes["adapter"], mask)
    for aspect in ("whole", "themes"):
        channels[aspect] = top_k(dense_scores(index, aspect, index.dense32(aspect)[seed]), sizes[aspect], mask)
    if use_graph and index.graph is not None:
        neighbors = index.graph.neighbors(seed, sizes["graph"] * 2)
        channels["graph"] = neighbors[mask[neighbors]][:sizes["graph"]]
    channels["bm25"] = top_k(index.bm25.item_scores(seed), sizes["bm25"], mask)
    return channels


def candidate_pool(index: Index, seed: int, eligible: np.ndarray, sizes: dict | None = None,
                   use_graph: bool = True) -> np.ndarray:
    """Union of the strongest neighbors under several views; the ranker then orders this pool."""
    pools = list(candidate_channels(index, seed, eligible, sizes, use_graph).values())
    return np.unique(np.concatenate(pools)) if pools else np.zeros(0, dtype=np.int64)
