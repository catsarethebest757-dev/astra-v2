"""Query-time ranker: set-based taste, aspect-aware dislikes, franchise collapse, and real explanations."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np

from .config import ASPECTS, SOURCES
from .features.text import MOOD_TAGS, top_tags
from .index import Index, dense_scores, top_k
from .learn.features import FEATURES, candidate_pool, pair_features
from .learn.ranker import HEURISTIC, Ranker, heuristic_scores
from .util import log, norm

WEIGHTS = ("plot", "themes", "tone", "creators")
REASON = {
    "cos_plot": "plot matches",
    "cos_themes": "themes match",
    "cos_tone": "tone matches",
    "cos_whole": "overall story matches",
    "cos_adapter": "crowd-taste match",
    "bm25": "shared wording",
    "tag_cos": "shared tags",
    "soft_tag_cos": "related tags",
    "creator_sim": "shared creators",
    "studio_sim": "shared studio",
    "graph_direct": "direct fan recommendation",
    "graph_2hop": "fans of similar titles recommend it",
    "graph_spectral": "same recommendation neighborhood",
    "cand_quality": "strong audience score",
}


def _trigrams(text: str) -> list[str]:
    padded = f"  {text} "
    return [padded[index:index + 3] for index in range(max(len(padded) - 2, 0))]


def _number(value, minimum, maximum, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return float(value)


class Engine:
    def __init__(self, index: Index, ranker: Ranker | None = None, encoder_name: str = "qwen3"):
        self.index = index
        self.ranker = ranker
        self.encoder_name = encoder_name
        self.encoder = None
        self._grams: dict[str, list[int]] = defaultdict(list)
        self._exact: dict[str, set[int]] = defaultdict(set)
        self._names: list[list[str]] = []
        for position, record in enumerate(index.records):
            names = []
            for value in (record["title"], *record["aliases"]):
                name = norm(value)
                if not name or name in names:
                    continue
                names.append(name)
                self._exact[name].add(position)
                for gram in set(_trigrams(name)):
                    self._grams[gram].append(position)
            self._names.append(names)

    def stats(self) -> dict:
        return {
            "count": self.index.size,
            "counts": dict(Counter(self.index.kind.tolist())),
            "sources": SOURCES,
            "encoder": self.encoder_name,
            "adapter": self.index.adapter is not None,
            "ranker": self.ranker is not None,
            "note": ("Crowd taste is distilled into content space, then reranked. "
                     "Scores are relative, not probabilities."),
        }

    def lookup(self, query: str, limit: int = 12, allow_adult: bool = False) -> list[dict]:
        query = norm(query)
        if not query:
            return []
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer from 1 to 50.")
        counts = Counter()
        for gram in _trigrams(query):
            counts.update(self._grams.get(gram, ()))
        pool = {index for index, _ in counts.most_common(500)} | set(self._exact.get(query, ()))
        ranked = []
        for position in pool:
            if self.index.adult[position] and not allow_adult:
                continue
            score = 0.0
            for name in self._names[position]:
                if query == name:
                    score = max(score, 1.0)
                elif name.startswith(query):
                    score = max(score, 0.9)
                elif query in name:
                    score = max(score, 0.78)
                else:
                    score = max(score, 0.7 * SequenceMatcher(None, query, name).ratio())
            if score >= 0.4 or position in self._exact.get(query, ()):
                ranked.append((score, position))
        ranked.sort(key=lambda pair: (-pair[0], self.index.records[pair[1]]["id"]))
        return [dict(self._brief(position), title_match=round(float(score), 3)) for score, position in ranked[:limit]]

    def recommend(self, request: dict) -> dict:
        parsed = self._parse(request)
        mask = self._mask(parsed)
        qvec = self._encode_query(parsed["query"]) if parsed["query"] else None
        ranked = self._rank(parsed["liked"], parsed["query"], qvec, mask, parsed)
        results = [self._card(item, parsed["query"]) for item in ranked]
        message = "Rank scores are relative, not match probabilities."
        if len(parsed["liked"]) > 24:
            message = "Used the first 24 liked titles for retrieval. " + message
        if not results:
            message = "No matches. Try broader terms or fewer filters."
        return {"results": results, "message": message}

    def rank_seed(self, seed: int, limit: int = 50, eligible: np.ndarray | None = None) -> list[int]:
        """Ordered candidate indices for evaluation. `eligible` already encodes the split's masks."""
        mask = np.ones(self.index.size, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool).copy()
        mask[seed] = False
        parsed = {
            "weights": {name: 1.0 for name in WEIGHTS}, "hidden_gems": 0.0, "diversity": 0.0, "limit": limit,
        }
        return [item["id"] for item in self._rank([seed], "", None, mask, parsed)]

    def _parse(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("Request must be an object.")
        query = request.get("query", "")
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("query must be a string of at most 1000 characters.")
        selections = {}
        for field in ("liked", "disliked", "seen", "exclude_tags"):
            values = request.get(field, [])
            if not isinstance(values, list) or len(values) > 500 or not all(isinstance(value, str) for value in values):
                raise ValueError(f"{field} must be an array of at most 500 strings.")
            selections[field] = list(dict.fromkeys(values))
        for field in ("liked", "disliked", "seen"):
            unknown = [value for value in selections[field] if value not in self.index.ids]
            if unknown:
                raise ValueError(f"Unknown {field} IDs: {', '.join(unknown[:3])}")
        liked = [self.index.ids[value] for value in selections["liked"]]
        disliked = [self.index.ids[value] for value in selections["disliked"]]
        seen = [self.index.ids[value] for value in selections["seen"]]
        if set(liked) & set(disliked):
            raise ValueError("A title cannot be both liked and disliked.")
        if not query.strip() and not liked:
            raise ValueError("Describe a mood or choose at least one liked title.")
        kind = request.get("kind", "all")
        if kind not in {"all", "anime", "movie", "tv"}:
            raise ValueError("Invalid kind filter.")
        limit = request.get("limit", 12)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer from 1 to 50.")
        diversity = _number(request.get("diversity", 0.12), 0, 0.6, "diversity")
        year_min = _number(request.get("year_min", 0), 0, 9999, "year_min")
        year_max = _number(request.get("year_max", 9999), 0, 9999, "year_max")
        if year_min > year_max:
            raise ValueError("year_min cannot exceed year_max.")
        min_rating = _number(request.get("min_rating", 0), 0, 10, "min_rating")
        hidden_gems = _number(request.get("hidden_gems", 0), 0, 1, "hidden_gems")
        for field in ("allow_adult", "cross_format"):
            if field in request and not isinstance(request[field], bool):
                raise ValueError(f"{field} must be a boolean.")
        weights = {name: 1.0 for name in WEIGHTS}
        supplied = request.get("aspect_weights", {})
        if not isinstance(supplied, dict):
            raise ValueError("aspect_weights must be an object.")
        for name in WEIGHTS:
            if name in supplied:
                weights[name] = _number(supplied[name], 0, 3, f"aspect_weights.{name}")
        return {
            "query": query.strip(), "liked": liked[:24], "disliked": disliked, "seen": seen,
            "exclude_tags": {norm(tag) for tag in selections["exclude_tags"] if norm(tag)},
            "kind": kind, "limit": limit, "diversity": diversity, "year_min": year_min, "year_max": year_max,
            "min_rating": min_rating, "hidden_gems": hidden_gems,
            "allow_adult": bool(request.get("allow_adult", False)),
            "cross_format": bool(request.get("cross_format", True)),
            "weights": weights,
        }

    def _mask(self, parsed: dict) -> np.ndarray:
        index = self.index
        mask = np.ones(index.size, dtype=bool)
        blocked = parsed["liked"] + parsed["disliked"] + parsed["seen"]
        if blocked:
            mask[np.array(blocked, dtype=np.int64)] = False
        if not parsed["allow_adult"]:
            mask &= ~index.adult
        if parsed["kind"] != "all":
            mask &= index.kind == parsed["kind"]
        elif not parsed["cross_format"] and parsed["liked"]:
            mask &= np.isin(index.kind, list(set(index.kind[parsed["liked"]])))
        mask &= (index.year >= parsed["year_min"]) & (index.year <= parsed["year_max"])
        if parsed["min_rating"] > 0:
            mask &= index.rating >= parsed["min_rating"]
        if parsed["liked"]:
            liked = np.array(parsed["liked"], dtype=np.int64)
            mask &= ~np.isin(index.franchise, list(set(index.franchise[liked].tolist())))
        if parsed["exclude_tags"]:
            banned = np.array([
                bool(parsed["exclude_tags"] & (set(record["tags"]) | set(record["genres"])))
                for record in index.records
            ])
            mask &= ~banned
        return mask

    def _encode_query(self, text: str):
        try:
            if self.encoder is None:
                from .features.embed import Encoder
                self.encoder = Encoder(self.encoder_name)
            vector = np.asarray(self.encoder.encode_query(text), dtype=np.float32).ravel()
        except Exception as error:
            log(f"Query embedding unavailable ({error}); using lexical match.")
            return None
        width = self.index.dense32("whole").shape[1]
        if vector.shape[0] != width or not np.isfinite(vector).all():
            return None
        return vector

    def _candidates(self, seeds: list[int], query: str, qvec, mask: np.ndarray) -> np.ndarray:
        pools = []
        sizes = {"adapter": 280, "whole": 140, "themes": 80, "graph": 50, "bm25": 50}
        for seed in seeds:
            pools.append(candidate_pool(self.index, seed, mask, sizes=sizes))
        if query:
            pools.append(top_k(self.index.bm25.query(query), 200, mask))
            if qvec is not None:
                for aspect, count in (("whole", 200), ("themes", 80), ("tone", 40)):
                    pools.append(top_k(dense_scores(self.index, aspect, qvec), count, mask))
        pools = [pool for pool in pools if len(pool)]
        if not pools:
            return np.zeros(0, dtype=np.int64)
        candidates = np.unique(np.concatenate(pools))
        candidates = candidates[mask[candidates]]
        if len(candidates) > 1500 and seeds:
            quick = self.index.dense32("whole")[candidates] @ self.index.dense32("whole")[seeds[0]]
            keep = np.argpartition(-quick, 1499)[:1500]
            candidates = candidates[keep]
        return candidates

    def _apply_weights(self, features: np.ndarray, weights: dict) -> np.ndarray:
        scaled = np.array(features, dtype=np.float32, copy=True)
        for aspect in ("plot", "themes", "tone"):
            scaled[:, FEATURES.index(f"cos_{aspect}")] *= weights[aspect]
        blend = (weights["plot"] + weights["themes"] + weights["tone"]) / 3
        scaled[:, FEATURES.index("cos_whole")] *= blend
        scaled[:, FEATURES.index("cos_adapter")] *= blend
        scaled[:, FEATURES.index("creator_sim")] *= weights["creators"]
        scaled[:, FEATURES.index("studio_sim")] *= weights["creators"]
        return scaled

    def _score_features(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.ranker is None:
            return heuristic_scores(features)
        scores = np.asarray(self.ranker.score(features), dtype=np.float32)
        if not np.isfinite(scores).all():
            return heuristic_scores(features)
        return scores

    def _query_features(self, query: str, qvec, candidates: np.ndarray) -> np.ndarray:
        features = np.zeros((len(candidates), len(FEATURES)), dtype=np.float32)
        if qvec is not None:
            for aspect in ASPECTS:
                features[:, FEATURES.index(f"cos_{aspect}")] = self.index.dense32(aspect)[candidates] @ qvec
            features[:, FEATURES.index("cos_adapter")] = features[:, FEATURES.index("cos_whole")]
        lexical = self.index.bm25.query(query)
        features[:, FEATURES.index("bm25")] = np.minimum(lexical[candidates], 2.0)
        folded = f" {norm(query)} "
        columns = [self.index.tags.index[name] for name in self.index.tags.names if f" {name} " in folded]
        if columns:
            weight = np.asarray(self.index.tags.matrix[candidates][:, columns].sum(axis=1)).ravel()
            features[:, FEATURES.index("tag_cos")] = np.clip(weight, 0, 1)
            features[:, FEATURES.index("soft_tag_cos")] = features[:, FEATURES.index("tag_cos")]
        features[:, FEATURES.index("cand_pop")] = self.index.pop[candidates]
        features[:, FEATURES.index("cand_quality")] = self.index.quality[candidates] / 10
        features[:, FEATURES.index("cand_has_synopsis")] = self.index.has_synopsis[candidates]
        return features

    def _aggregate(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if matrix.shape[0] == 1:
            return matrix[0], np.zeros(matrix.shape[1], dtype=np.int64)
        capped = np.where(matrix > -1e3, matrix, -1e4)
        peak = capped.max(axis=0)
        gathered = np.exp(np.clip(capped - peak, -40, 0)).sum(axis=0)
        scores = peak + np.log(np.maximum(gathered, 1e-12))
        consensus = np.zeros(matrix.shape[1], dtype=np.float32)
        for row in capped:
            valid = row > -1e3
            count = int(valid.sum())
            if count == 0:
                continue
            k = min(40, count)
            threshold = np.partition(row[valid], -k)[-k]
            consensus += (row >= threshold) & valid
        return scores + 0.18 * np.log1p(consensus), capped.argmax(axis=0)

    def _penalize(self, disliked: list[int], candidates: np.ndarray, weights: dict) -> np.ndarray:
        penalty = np.zeros(len(candidates), dtype=np.float32)
        if not disliked:
            return penalty
        for item in disliked:
            for aspect in ("plot", "themes", "tone"):
                similarity = self.index.dense32(aspect)[candidates] @ self.index.dense32(aspect)[item]
                penalty += weights[aspect] * np.maximum(similarity - 0.2, 0)
        return penalty / len(disliked)

    def _rank(self, seeds: list[int], query: str, qvec, mask: np.ndarray, parsed: dict) -> list[dict]:
        candidates = self._candidates(seeds, query, qvec, mask)
        if len(candidates) == 0:
            return []
        weights = parsed["weights"]
        best_seed = np.full(len(candidates), -1, dtype=np.int64)
        if seeds:
            matrix = np.full((len(seeds), len(candidates)), -1e4, dtype=np.float32)
            for row, seed in enumerate(seeds):
                raw = pair_features(self.index, seed, candidates)
                matrix[row] = self._score_features(self._apply_weights(raw, weights))
            scores, which = self._aggregate(matrix)
            best_seed = np.array([seeds[int(index)] for index in which], dtype=np.int64)
            if query:
                query_scores = self._score_features(self._apply_weights(self._query_features(query, qvec, candidates), weights))
                scores = scores + 0.45 * query_scores
        else:
            scores = self._score_features(self._apply_weights(self._query_features(query, qvec, candidates), weights))
        scores = scores - parsed["hidden_gems"] * self.index.pop[candidates]
        scores = scores + (1 - parsed["hidden_gems"]) * 0.12 * (self.index.quality[candidates] / 10)
        scores = scores - 0.65 * self._penalize(parsed.get("disliked", []), candidates, weights)
        valid = np.isfinite(scores)
        candidates, scores, best_seed = candidates[valid], scores[valid], best_seed[valid]
        chosen, chosen_scores, chosen_seeds = self._collapse(candidates, scores, best_seed)
        ordered = self._mmr(chosen, chosen_scores, parsed["limit"], parsed["diversity"])
        seed_of = {item: seed for item, seed in zip(chosen, chosen_seeds)}
        score_of = {item: score for item, score in zip(chosen, chosen_scores)}
        return [{"id": item, "score": float(score_of[item]), "seed": int(seed_of[item])} for item in ordered]

    def _collapse(self, candidates, scores, seeds):
        index = self.index
        best_pos = {}
        entry_pos = {}
        for position, item in enumerate(candidates):
            franchise = int(index.franchise[item])
            if index.entry[item]:
                entry_pos[franchise] = position
            current = best_pos.get(franchise)
            if current is None or scores[position] > scores[current]:
                best_pos[franchise] = position
        picked = []
        for franchise, position in best_pos.items():
            entry = entry_pos.get(franchise)
            if entry is not None and scores[entry] >= 0.9 * max(scores[position], 1e-6):
                position = entry
            picked.append(position)
        picked.sort(key=lambda position: (-scores[position], int(candidates[position])))
        return (candidates[picked], scores[picked], seeds[picked])

    def _mmr(self, ids, scores, limit: int, diversity: float) -> list[int]:
        if len(ids) == 0:
            return []
        if diversity <= 0 or len(ids) == 1:
            order = np.lexsort((ids, -scores))
            return [int(ids[index]) for index in order[:limit]]
        vectors = self.index.adapter[ids] if self.index.adapter is not None else self.index.dense32("whole")[ids]
        vectors = np.asarray(vectors, dtype=np.float32)
        selected = []
        live = scores.astype(np.float32).copy()
        redundant = np.zeros(len(ids), dtype=np.float32)
        for _ in range(min(limit, len(ids))):
            objective = (1 - diversity) * live - diversity * redundant
            pick = int(np.argmax(objective))
            if not np.isfinite(objective[pick]):
                break
            selected.append(int(ids[pick]))
            redundant = np.maximum(redundant, vectors @ vectors[pick])
            live[pick] = -np.inf
        return selected

    def _brief(self, position: int) -> dict:
        record = self.index.records[position]
        return {
            "id": record["id"], "title": record["title"], "kind": record["kind"], "year": record["year"],
            "rating": record["rating"], "synopsis": record["synopsis"], "url": record["url"],
            "image": record["image"], "format": record["format"],
            "tags": top_tags(record, 12) or list(record["genres"][:6]),
        }

    def _card(self, ranked: dict, query: str) -> dict:
        position = int(ranked["id"])
        seed = int(ranked["seed"])
        card = self._brief(position)
        card["score"] = round(float(ranked["score"]), 4)
        if seed >= 0:
            raw = pair_features(self.index, seed, np.array([position]))
        else:
            qvec = self._encode_query(query) if query else None
            raw = self._query_features(query, qvec, np.array([position]))
        weighted = self._apply_weights(raw, {name: 1.0 for name in WEIGHTS})
        if self.ranker is not None:
            attribution = self.ranker.attribute(weighted)[0]
        else:
            attribution = weighted[0] * HEURISTIC
        card["aspects"] = {
            aspect: round(max(0.0, float(raw[0, FEATURES.index(f"cos_{aspect}")])), 3) for aspect in ASPECTS
        }
        if self.index.adapter is not None and seed >= 0:
            card["aspects"]["adapter"] = round(max(0.0, float(raw[0, FEATURES.index("cos_adapter")])), 3)
        reasons, shared = self._reasons(position, seed, query, raw[0], attribution)
        card["reasons"] = reasons
        card["shared_tags"] = shared
        card["explanation"] = " · ".join(reasons)
        order = np.argsort(-attribution)
        card["signals"] = {
            FEATURES[int(column)]: round(float(attribution[int(column)]), 4)
            for column in order[:6] if attribution[int(column)] > 0
        }
        return card

    def _reasons(self, position: int, seed: int, query: str, raw: np.ndarray, attribution: np.ndarray):
        record = self.index.records[position]
        reasons = []
        shared: list[str] = []
        if seed >= 0:
            seed_record = self.index.records[seed]
            shared = [name for name, _ in sorted(
                ((name, min(record["tags"].get(name, 0), seed_record["tags"].get(name, 0)))
                 for name in set(record["tags"]) & set(seed_record["tags"])),
                key=lambda pair: (-pair[1], pair[0]))][:6]
            moods = [name for name in shared if name in MOOD_TAGS]
            if moods:
                reasons.append("same tone: " + ", ".join(moods[:4]))
            elif shared:
                reasons.append("shared tags: " + ", ".join(shared[:5]))
            creators, studios = self.index.structure.shared(seed, position, self.index.records)
            for creator in creators[:2]:
                role, _, name = creator.partition(":")
                reasons.append(f"shared {role}: {name}" if name else f"shared creator: {creator}")
            if studios:
                reasons.append("shared studio: " + ", ".join(studios[:2]))
            if self.index.graph is not None:
                votes = self.index.graph.votes_between(seed, position)
                if votes > 0:
                    reasons.append(f"{votes} fans of {seed_record['title']} recommend this")
        else:
            folded = f" {norm(query)} "
            shared = [name for name in list(record["tags"]) + list(record["genres"]) if f" {name} " in folded][:6]
            if shared:
                reasons.append("matches your words: " + ", ".join(shared[:5]))
        used = set()
        order = np.argsort(-attribution)
        for column in order:
            name = FEATURES[int(column)]
            phrase = REASON.get(name)
            if phrase is None or attribution[int(column)] <= 0 or phrase in used:
                continue
            if any(phrase.split()[0] in reason for reason in reasons):
                continue
            reasons.append(phrase)
            used.add(phrase)
            if len(reasons) >= 4:
                break
        if not reasons:
            best = max(ASPECTS, key=lambda aspect: raw[FEATURES.index(f"cos_{aspect}")])
            reasons.append(f"closest {best} match")
        return reasons[:5], shared
