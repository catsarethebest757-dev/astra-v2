"""Creator and studio overlap with IDF weighting: a rare director is a stronger signal than a prolific studio."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _entity_matrix(values: list[list[str]]) -> tuple[sparse.csr_matrix, list[str]]:
    vocabulary: dict[str, int] = {}
    rows, columns = [], []
    for row, entities in enumerate(values):
        for entity in set(entities):
            if entity:
                rows.append(row)
                columns.append(vocabulary.setdefault(entity, len(vocabulary)))
    names = sorted(vocabulary, key=vocabulary.get)
    binary = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                               shape=(len(values), max(len(vocabulary), 1)))
    df = np.asarray(binary.sum(axis=0)).ravel()
    idf = np.log((len(values) + 1) / (df + 1)).astype(np.float32)
    idf[df <= 1] = 0
    weighted = binary @ sparse.diags(idf)
    lengths = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    return (sparse.diags(1.0 / np.maximum(lengths, 1e-12)) @ weighted).tocsr().astype(np.float32), names


class Structure:
    def __init__(self, records: list[dict]):
        self.creators, self.creator_names = _entity_matrix([record["creators"] for record in records])
        self.studios, self.studio_names = _entity_matrix([record["studios"] for record in records])
        self.year = np.array([record["year"] for record in records], dtype=np.float32)
        self.runtime = np.array([max(record["runtime"], 0) * max(record["episodes"], 1) if record["kind"] != "movie"
                                 else record["runtime"] for record in records], dtype=np.float32)
        self.film = np.array([record["kind"] == "movie" or record["format"] == "MOVIE" for record in records])
        sources = {name: index for index, name in enumerate(sorted({record["source"] for record in records}))}
        self.source = np.array([sources[record["source"]] if record["source"] else -1 for record in records])

    def seed_signals(self, seed: int, candidates: np.ndarray) -> np.ndarray:
        """Per-candidate [creator sim, studio sim, year gap, film/series match, total-length log ratio, source match]."""
        output = np.zeros((len(candidates), 6), dtype=np.float32)
        output[:, 0] = (self.creators[candidates] @ self.creators[seed].T).toarray().ravel()
        output[:, 1] = (self.studios[candidates] @ self.studios[seed].T).toarray().ravel()
        years = self.year[candidates]
        known = (years > 0) & (self.year[seed] > 0)
        output[:, 2] = np.where(known, np.minimum(np.abs(years - self.year[seed]) / 30.0, 1.5), 0.5)
        output[:, 3] = self.film[candidates] == self.film[seed]
        lengths = self.runtime[candidates]
        known = (lengths > 0) & (self.runtime[seed] > 0)
        output[:, 4] = np.where(known, np.abs(np.log((lengths + 1) / (self.runtime[seed] + 1))) / 4.0, 0.5)
        output[:, 5] = (self.source[candidates] == self.source[seed]) & (self.source[seed] >= 0)
        return output

    def shared(self, seed: int, candidate: int, records: list[dict]) -> tuple[list[str], list[str]]:
        creators = sorted(set(records[seed]["creators"]) & set(records[candidate]["creators"]))
        studios = sorted(set(records[seed]["studios"]) & set(records[candidate]["studios"]))
        return creators, studios
