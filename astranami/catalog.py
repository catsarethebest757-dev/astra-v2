"""Unified catalog: merge sources, remove cross-provider duplicates, cluster franchises, derive priors."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import BLOCKED_TAGS, KINDS
from .sources import anilist, manami, movies, tmdb, tvmaze
from .util import atomic_json, log, norm, read_json

FIELDS = {
    "id": str, "kind": str, "title": str, "aliases": list, "year": int, "end_year": int, "synopsis": str,
    "genres": list, "tags": dict, "studios": list, "creators": list, "source": str, "format": str,
    "episodes": int, "runtime": int, "country": str, "language": str, "network": str, "rating": float,
    "popularity": int, "favourites": int, "adult": bool, "url": str, "image": str, "related": list,
    "recs": dict, "external": dict,
}
DERIVED = {"franchise": str, "entry": bool, "pop_pct": float, "quality": float}
ANIME_LANGUAGES = {"japanese", "chinese", "korean", "ja", "zh", "ko"}
SEQUEL_SUFFIX = re.compile(r"(\s+(part|chapter|vol|volume|season|episode)?\s*([0-9]+|[ivx]+))+$")


def blank(identifier: str, title: str, kind: str, **fields) -> dict:
    record = {name: kind_type() for name, kind_type in FIELDS.items()}
    record.update(id=identifier, title=title, kind=kind, rating=0.0)
    record.update(fields)
    return record


def validate(records: list[dict], minimum: int = 1) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("Catalog must be a list of records.")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every catalog record must be an object.")
        for name, expected in FIELDS.items():
            value = record.get(name)
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                record[name] = value = float(value)
            if expected is int and isinstance(value, bool) or not isinstance(value, expected):
                raise ValueError(f"{record.get('id', '?')}: field {name} must be {expected.__name__}.")
        if not record["id"] or record["id"] in seen:
            raise ValueError(f"Duplicate or empty id: {record['id']!r}")
        seen.add(record["id"])
        if record["kind"] not in KINDS:
            raise ValueError(f"{record['id']}: kind must be one of {KINDS}.")
        if not record["title"].strip():
            raise ValueError(f"{record['id']}: empty title.")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                   for value in record["tags"].values()):
            raise ValueError(f"{record['id']}: tag weights must be in [0, 1].")
        if not 0 <= record["rating"] <= 10 or record["year"] < 0 or record["popularity"] < 0:
            raise ValueError(f"{record['id']}: rating must be 0-10; year and popularity nonnegative.")
        record["adult"] = record["adult"] or bool((set(record["tags"]) | set(record["genres"])) & BLOCKED_TAGS)
    if len(records) < minimum:
        raise ValueError(f"Only {len(records):,} records; at least {minimum:,} required.")
    return records


def _names(record: dict) -> set[str]:
    return {name for name in (norm(value) for value in [record["title"], *record["aliases"]]) if len(name) >= 2}


def _looks_like_anime(record: dict) -> bool:
    genres = set(record["genres"])
    language = record["language"].casefold()
    return "anime" in genres or (("animation" in genres or record["format"] == "Animation")
                                 and (language in ANIME_LANGUAGES or record["country"] in {"JP", "CN", "KR"}))


def drop_anime_duplicates(anime: list[dict], others: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Remove TV/movie entries that are the same work as an AniList entry (same name, year within one)."""
    index = defaultdict(list)
    for record in anime:
        for name in _names(record):
            index[name].append(record)
    kept, mapping = [], {}
    for record in others:
        match = None
        if _looks_like_anime(record):
            for name in _names(record):
                for candidate in index.get(name, ()):
                    if not record["year"] or not candidate["year"] or abs(candidate["year"] - record["year"]) <= 1:
                        if record["kind"] != "movie" or candidate["format"] == "MOVIE":
                            match = candidate
                            break
                if match:
                    break
        if match:
            mapping[record["id"]] = match["id"]
        else:
            kept.append(record)
    return kept, mapping


def merge_tmdb_tv(tv: list[dict], tmdb_tv: list[dict]) -> list[dict]:
    """Enrich TVmaze shows with TMDB keywords, creators, and recommendation edges (joined on IMDb id)."""
    by_imdb = {record["external"]["imdb"]: record for record in tv if record["external"].get("imdb")}
    tmdb_to_tv, unmatched = {}, []
    for record in tmdb_tv:
        target = by_imdb.get(record["external"].get("imdb"))
        if target is None:
            unmatched.append(record)
            continue
        tmdb_to_tv[record["id"]] = target["id"]
        target["tags"] = {**record["tags"], **target["tags"]}
        target["creators"] = sorted(set(target["creators"]) | set(record["creators"]))
        target["recs"] = dict(record["recs"])
        target["external"]["tmdb"] = record["external"]["tmdb"]
        if not target["image"]:
            target["image"] = record["image"]
    for record in tv:
        record["recs"] = {tmdb_to_tv.get(key, key): value for key, value in record["recs"].items()}
    for record in unmatched:
        record["recs"] = {tmdb_to_tv.get(key, key): value for key, value in record["recs"].items()}
    return tv + unmatched


def cluster_franchises(records: list[dict]) -> None:
    """Union-find over explicit relations plus a sequel-title heuristic; pick one entry point per franchise."""
    position = {record["id"]: index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for index, record in enumerate(records):
        for other in record["related"]:
            if other in position:
                join(index, position[other])
    stems = defaultdict(list)
    for index, record in enumerate(records):
        if record["kind"] != "anime":
            stem = SEQUEL_SUFFIX.sub("", norm(record["title"].split(":")[0]))
            if len(stem) >= 4:
                stems[(record["kind"], stem)].append(index)
    for (_, stem), members in stems.items():
        bases = [index for index in members if norm(records[index]["title"]) == stem]
        if len(bases) != 1 or len(members) > 12:
            continue
        for other in members:
            if other != bases[0] and abs((records[other]["year"] or 0) - (records[bases[0]]["year"] or 0)) <= 25:
                join(bases[0], other)
    groups = defaultdict(list)
    for index in range(len(records)):
        groups[root(index)].append(index)
    preference = {"TV": 0, "MOVIE": 1, "ONA": 1, "Scripted": 0, "Animation": 0, "TV_SHORT": 2, "OVA": 3}
    for members in groups.values():
        entry = min(members, key=lambda index: (
            preference.get(records[index]["format"], 2),
            records[index]["year"] or 9999,
            -records[index]["popularity"]))
        for index in members:
            records[index]["franchise"] = records[entry]["id"]
            records[index]["entry"] = index == entry


def derive_priors(records: list[dict]) -> None:
    """Per-kind popularity percentile and a Bayesian-shrunk quality score (0-10)."""
    for kind in KINDS:
        members = [record for record in records if record["kind"] == kind]
        if not members:
            continue
        popularity = np.array([record["popularity"] for record in members], dtype=np.float64)
        order = popularity.argsort(kind="stable").argsort(kind="stable")
        percentile = order / max(len(members) - 1, 1)
        rated = [record["rating"] for record in members if record["rating"] > 0]
        mean = float(np.mean(rated)) if rated else 6.5
        prior_weight = float(np.median(popularity[popularity > 0])) if (popularity > 0).any() else 1.0
        for record, pct in zip(members, percentile):
            record["pop_pct"] = round(float(pct), 4)
            votes = record["popularity"] if record["rating"] > 0 else 0
            record["quality"] = round((votes * record["rating"] + prior_weight * mean) / (votes + prior_weight), 3)


def clean_edges(records: list[dict]) -> None:
    known = {record["id"] for record in records}
    for record in records:
        record["recs"] = {key: value for key, value in record["recs"].items() if key in known and key != record["id"]}
        record["related"] = [key for key in record["related"] if key in known and key != record["id"]]


def build(directory: Path) -> list[dict]:
    anime = anilist.to_records(anilist.load_raw(directory))
    manami.merge_synonyms(anime, manami.synonyms_by_anilist(directory))
    tv = tvmaze.to_records(tvmaze.load_raw(directory))
    film = movies.load(directory)
    extra = tmdb.to_records(directory) if (directory / "raw" / "tmdb").exists() else []
    if extra:
        film_by_id = {record["id"]: record for record in film}
        for record in extra:
            if record["kind"] == "movie":
                film_by_id[record["id"]] = record
        film = list(film_by_id.values())
        tv = merge_tmdb_tv(tv, [record for record in extra if record["kind"] == "tv"])
    others, mapping = drop_anime_duplicates(anime, tv + film)
    log(f"Catalog: {len(anime):,} anime, {len(others):,} TV/movies after removing {len(mapping):,} anime duplicates.")
    records = anime + others
    for record in records:
        record["recs"] = {mapping.get(key, key): value for key, value in record["recs"].items()}
        record["rating"] = float(record["rating"])
    clean_edges(records)
    cluster_franchises(records)
    derive_priors(records)
    validate(records, minimum=1000)
    atomic_json(directory / "catalog.json", records)
    atomic_json(directory / "catalog_meta.json", {
        "counts": dict(Counter(record["kind"] for record in records)),
        "rec_edges": sum(len(record["recs"]) for record in records),
        "franchises": len({record["franchise"] for record in records}),
        "duplicates_removed": len(mapping),
    })
    return records


def load(directory: Path) -> list[dict]:
    return validate(read_json(directory / "catalog.json"))
