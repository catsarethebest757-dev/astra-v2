"""Leakage-free splits of the recommendation graph for training and evaluation.

- test seeds: all their edges are hidden from every trained component (cold start); half are revealed
  to the graph for the warm-start evaluation.
- ranker seeds: their edges are hidden from the adapter, so the ranker learns from honest adapter scores;
  half of each ranker seed's edges stay in the ranker's graph so it also learns how to use direct evidence.
- adapter edges: everything else.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..index import Index

Edges = dict[tuple[int, int], int]


@dataclass
class Split:
    test_seeds: np.ndarray
    ranker_seeds: np.ndarray
    adapter_edges: Edges
    ranker_graph: Edges
    cold_graph: Edges
    warm_graph: Edges
    warm_visible: dict[int, set[int]] = field(default_factory=dict)


def adjacency(edges: Edges) -> dict[int, dict[int, int]]:
    output: dict[int, dict[int, int]] = defaultdict(dict)
    for (left, right), votes in edges.items():
        output[left][right] = votes
        output[right][left] = votes
    return output


def relevant(index: Index, seed: int, neighbors: dict[int, int], exclude: set[int] | None = None) -> dict[int, float]:
    """Graded targets: log2(1 + votes), excluding the seed's own franchise, adult titles, and known items."""
    exclude = exclude or set()
    return {target: float(np.log2(1 + votes)) for target, votes in neighbors.items()
            if index.franchise[target] != index.franchise[seed] and not index.adult[target] and target not in exclude}


def make_split(index: Index, edges: Edges, test_fraction: float, ranker_fraction: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    graph = adjacency(edges)
    pool = np.array(sorted(node for node, neighbors in graph.items()
                           if len(neighbors) >= 3 and not index.adult[node]), dtype=np.int64)
    pool = rng.permutation(pool)
    test_count = int(round(len(pool) * test_fraction))
    ranker_count = int(round(len(pool) * ranker_fraction))
    if len(pool) > 4:
        test_count = min(max(test_count, 1), len(pool) // 5)
    ranker_count = min(ranker_count, max(0, len(pool) - test_count - 1))
    test = np.sort(pool[:test_count])
    ranker = np.sort(pool[test_count:test_count + ranker_count])
    test_set, ranker_set = set(test.tolist()), set(ranker.tolist())
    held = test_set | ranker_set
    adapter_edges = {key: votes for key, votes in edges.items() if key[0] not in held and key[1] not in held}
    cold_graph = {key: votes for key, votes in edges.items() if key[0] not in test_set and key[1] not in test_set}
    warm_graph = dict(cold_graph)
    warm_visible: dict[int, set[int]] = {}
    for node in test:
        neighbors = sorted(graph[node])
        rng.shuffle(neighbors)
        visible = set(neighbors[:len(neighbors) // 2])
        warm_visible[int(node)] = visible
        for other in visible:
            key = (min(node, other), max(node, other))
            warm_graph[key] = edges[key]
    ranker_graph = dict(cold_graph)
    for node in ranker:
        neighbors = sorted(graph[node])
        rng.shuffle(neighbors)
        for other in neighbors[len(neighbors) // 2:]:
            ranker_graph.pop((min(node, other), max(node, other)), None)
    return Split(test, ranker, adapter_edges, ranker_graph, cold_graph, warm_graph, warm_visible)
