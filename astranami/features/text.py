"""Aspect texts for the dense encoder and a BM25 lexical index."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from ..util import norm, tokens

MOOD_TAGS = {
    "psychological", "tragedy", "philosophy", "iyashikei", "slice of life", "dark", "gore", "survival",
    "coming of age", "melancholy", "nihilism", "surreal comedy", "parody", "satire", "slapstick", "cgdct",
    "wholesome", "romance", "bittersweet", "suspense", "thriller", "horror", "mystery", "noir", "dystopian",
    "feel good", "heartwarming", "violence", "war", "revenge", "anti hero", "villainess", "tearjerker",
    "episodic", "ensemble cast", "slow burn", "found family", "workplace", "school", "military", "politics",
    "time manipulation", "time loop", "isekai", "mecha", "idol", "sports", "cooking", "music", "shounen",
    "seinen", "shoujo", "josei", "kids", "family", "absurdist humor", "black comedy", "coming-of-age",
    "psychological thriller", "crime", "tense", "existentialism", "death game", "love triangle",
}
ERA = [(1980, "classic"), (2000, "retro"), (2012, "2000s-era"), (2020, "modern"), (9999, "recent")]


def _era(year: int) -> str:
    if not year:
        return ""
    return next(label for limit, label in ERA if year < limit)


def _length(record: dict) -> str:
    if record["kind"] == "movie" or record["format"] == "MOVIE":
        return "feature film"
    episodes = record["episodes"]
    if episodes and episodes <= 13:
        return "short series"
    if episodes and episodes >= 50:
        return "long-running series"
    return "series"


def top_tags(record: dict, limit: int = 18) -> list[str]:
    return [name for name, _ in sorted(record["tags"].items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def aspect_texts(record: dict) -> dict[str, str]:
    kind = {"anime": "anime", "tv": "TV show", "movie": "movie"}[record["kind"]]
    genres = ", ".join(record["genres"])
    tags = top_tags(record)
    synopsis = record["synopsis"][:1600]
    themes = ", ".join(tags) or genres
    moods = [tag for tag in record["tags"] if tag in MOOD_TAGS]
    creators = ", ".join(creator.split(":", 1)[1] for creator in record["creators"][:4])
    tone_parts = [f"A {_era(record['year'])} {_length(record)} ({kind})".replace("  ", " "),
                  f"Genres: {genres}" if genres else "", f"Mood: {', '.join(moods)}" if moods else ""]
    whole = [f"{record['title']} ({kind}, {record['year'] or 'unknown year'})",
             f"Genres: {genres}" if genres else "", f"Themes: {themes}" if themes else "",
             f"Studio: {', '.join(record['studios'][:2])}" if record["studios"] else "",
             f"Created by: {creators}" if creators else "", synopsis]
    return {
        "plot": f"Premise: {synopsis}" if synopsis else f"Premise unknown. Themes: {themes}. Genres: {genres}",
        "themes": f"Themes: {themes}. Genres: {genres}",
        "tone": ". ".join(part for part in tone_parts if part),
        "whole": ". ".join(part for part in whole if part),
    }


def document_counts(record: dict) -> Counter:
    counts = Counter(tokens(record["synopsis"]))
    for name, weight in record["tags"].items():
        boost = 1 + round(3 * weight)
        counts.update({token: boost for token in tokens(name)})
        counts["tag:" + name] += boost + 2
    for genre in record["genres"]:
        counts.update({token: 2 for token in tokens(genre)})
        counts["tag:" + genre] += 3
    return counts


class BM25:
    def __init__(self, records: list[dict], vocabulary_limit: int = 200000, k1: float = 1.2, b: float = 0.75):
        documents = [document_counts(record) for record in records]
        frequency = Counter(token for counts in documents for token in counts)
        selected = sorted((token for token, count in frequency.items() if count >= 2),
                          key=lambda token: (-frequency[token], token))[:vocabulary_limit]
        self.vocabulary = {token: index for index, token in enumerate(selected)}
        counts = self.counts(documents)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        total = len(records)
        self.idf = np.log1p((total - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        normalizer = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1.0))
        coo = counts.tocoo()
        data = coo.data * (k1 + 1) / (coo.data + normalizer[coo.row]) * self.idf[coo.col]
        self.matrix = sparse.csr_matrix((data.astype(np.float32), (coo.row, coo.col)), shape=counts.shape)
        self.self_scores = np.asarray(self.matrix.multiply(self.matrix > 0).sum(axis=1)).ravel()

    def counts(self, documents: list[Counter]) -> sparse.csr_matrix:
        values, columns, pointers = [], [], [0]
        for counter in documents:
            for token, count in counter.items():
                index = self.vocabulary.get(token)
                if index is not None:
                    columns.append(index)
                    values.append(count)
            pointers.append(len(values))
        return sparse.csr_matrix((np.asarray(values, dtype=np.float32), columns, pointers),
                                 shape=(len(documents), len(self.vocabulary)))

    def query(self, text: str) -> np.ndarray:
        counts = Counter(tokens(text))
        normalized = f" {norm(text)} "
        for token in self.vocabulary:
            if token.startswith("tag:") and f" {token[4:]} " in normalized:
                counts[token] += 3
        vector = self.counts([counts])
        vector.data[:] = 1.0
        return np.asarray((self.matrix @ vector.T).todense()).ravel()

    def item_query(self, index: int, top_terms: int = 60) -> sparse.csr_matrix:
        """Seed document as a query: its highest-weighted terms."""
        row = self.matrix[index]
        if row.nnz > top_terms:
            keep = np.argsort(-row.data)[:top_terms]
            row = sparse.csr_matrix((row.data[keep], row.indices[keep], [0, top_terms]), shape=row.shape)
        return row

    def item_scores(self, index: int, candidates: np.ndarray | None = None) -> np.ndarray:
        query = self.item_query(index)
        query.data = np.ones_like(query.data)
        matrix = self.matrix if candidates is None else self.matrix[candidates]
        scores = np.asarray((matrix @ query.T).todense()).ravel()
        return scores / max(float(self.self_scores[index]), 1e-6)
