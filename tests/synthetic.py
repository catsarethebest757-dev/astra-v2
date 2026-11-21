"""Seeded synthetic catalog with planted taste clusters, for offline ranking and latency checks."""
from __future__ import annotations

import numpy as np

from astranami.catalog import blank, cluster_franchises, derive_priors
from astranami.config import ASPECTS
from astranami.index import Index

NOISE = {"plot": 0.9, "themes": 0.7, "tone": 1.1, "whole": 0.8}


def synthetic_index(clusters: int = 24, per_cluster: int = 25, width: int = 48, seed: int = 5) -> Index:
    """Titles in the same cluster share tags, wording, embedding centroids, and most fan edges."""
    rng = np.random.default_rng(seed)
    vocabulary = [f"word{number}" for number in range(clusters * 6)]
    tag_names = [f"tag{number}" for number in range(clusters * 2)]
    records = []
    for position in range(clusters * per_cluster):
        cluster = position // per_cluster
        words = list(rng.choice(vocabulary[cluster * 6:cluster * 6 + 6], 8)) + list(rng.choice(vocabulary, 6))
        tags = {tag_names[cluster * 2]: 0.9, tag_names[cluster * 2 + 1]: 0.6,
                tag_names[int(rng.integers(len(tag_names)))]: 0.4}
        kind = ("anime", "anime", "tv", "movie")[int(rng.integers(4))]
        records.append(blank(
            f"{kind}:syn:{position}", f"Title {position}", kind, synopsis=" ".join(words) * 3,
            genres=[tag_names[cluster * 2]], tags=tags, year=1990 + int(rng.integers(35)),
            rating=float(np.round(5 + 4 * rng.random(), 1)), popularity=int(rng.integers(1, 5000)),
            format="MOVIE" if kind == "movie" else "TV", episodes=1 if kind == "movie" else 12, runtime=24,
            studios=[f"studio {int(rng.integers(clusters))}"], creators=[f"director:person {int(rng.integers(clusters * 3))}"],
        ))
    for position, record in enumerate(records):
        cluster = position // per_cluster
        if position % per_cluster == 1:
            record["related"] = [records[position - 1]["id"]]
        inside = rng.choice(np.arange(cluster * per_cluster, (cluster + 1) * per_cluster), 6, replace=False)
        outside = rng.integers(0, len(records), 1)
        record["recs"] = {records[int(other)]["id"]: int(rng.integers(2, 60))
                          for other in np.concatenate([inside, outside]) if int(other) != position}
    cluster_franchises(records)
    derive_priors(records)
    index = Index(records)
    for aspect in ASPECTS:
        centroids = rng.normal(size=(clusters, width)).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        vectors = np.repeat(centroids, per_cluster, axis=0)
        vectors = vectors + NOISE[aspect] / np.sqrt(width) * rng.normal(size=vectors.shape).astype(np.float32)
        index.aspects[aspect] = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)
    index.build_sparse(None)
    index.set_graph(index.all_edges())
    return index


def cluster_of(index: Index, identifier: str, per_cluster: int = 25) -> int:
    return index.ids[identifier] // per_cluster
