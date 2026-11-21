"""Historical TMDB 5000 movie CSV (used when no TMDB API key is configured)."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..config import BLOCKED_TAGS, MOVIE_CSV_URL
from ..util import clean, download, norm, number


def fetch(directory: Path) -> Path:
    path = directory / "raw" / "tmdb_5000_movies.csv"
    download(MOVIE_CSV_URL, path)
    return path


def to_records(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    required = {"id", "title", "overview", "genres", "keywords"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Movie CSV schema changed or required columns are missing.")
    records = []
    for item in reader:
        if not item["id"] or not clean(item["title"]):
            continue
        genres = sorted({norm(entry["name"]) for entry in json.loads(item["genres"] or "[]") if norm(entry["name"])})
        keywords = [norm(entry["name"]) for entry in json.loads(item["keywords"] or "[]") if norm(entry["name"])]
        companies = [norm(entry["name"]) for entry in json.loads(item.get("production_companies") or "[]")][:4]
        countries = json.loads(item.get("production_countries") or "[]")
        tags = {keyword: round(max(0.4, 0.9 - 0.02 * position), 3) for position, keyword in enumerate(keywords[:25])}
        synopsis = clean(item.get("overview"))
        if item.get("tagline"):
            synopsis = f"{synopsis} {clean(item['tagline'])}".strip()
        records.append(dict(
            id=f"movie:tmdb:{item['id']}", kind="movie", title=clean(item["title"]),
            aliases=[clean(item["original_title"])] if clean(item.get("original_title")) not in {"", clean(item["title"])} else [],
            year=int(number((item.get("release_date") or "")[:4])), end_year=0, synopsis=synopsis,
            genres=genres, tags=tags, studios=[company for company in companies if company], creators=[],
            source="", format="MOVIE", episodes=1, runtime=int(number(item.get("runtime"))),
            country=countries[0]["iso_3166_1"] if countries else "", language=item.get("original_language") or "",
            network="", rating=round(number(item.get("vote_average")), 2),
            popularity=int(number(item.get("vote_count"))), favourites=0,
            adult=bool((set(genres) | set(keywords)) & BLOCKED_TAGS),
            url=f"https://www.themoviedb.org/movie/{item['id']}", image="", related=[], recs={},
            external={"tmdb": int(item["id"])},
        ))
    return records


def load(directory: Path) -> list[dict]:
    path = directory / "raw" / "tmdb_5000_movies.csv"
    if not path.exists():
        return []
    return to_records(path.read_bytes().decode("utf-8-sig"))
