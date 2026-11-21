"""Full TVmaze show index (CC BY-SA), cached per page."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path

from ..config import BLOCKED_TAGS, TVMAZE_URL
from ..util import RateLimiter, clean, http, log, norm, number

SKIPPED_TYPES = {"News", "Sports", "Award Show"}


def page_path(directory: Path, page: int) -> Path:
    return directory / "raw" / "tvmaze" / f"page-{page:04d}.json"


def crawl(directory: Path, max_pages: int = 1000, progress=log) -> int:
    done = directory / "raw" / "tvmaze.done"
    if done.exists():
        return int(done.read_text(encoding="utf-8"))
    limiter = RateLimiter(100)
    pages = 0
    for page in range(max_pages):
        path = page_path(directory, page)
        if path.exists():
            pages += 1
            continue
        try:
            body, _ = http(TVMAZE_URL.format(page), limiter=limiter)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                break
            raise
        payload = json.loads(body)
        if not isinstance(payload, list):
            raise ValueError("TVmaze returned an unexpected payload.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        pages += 1
        if page % 50 == 0:
            progress(f"TVmaze: page {page} fetched")
    done.write_text(str(pages), encoding="utf-8")
    progress(f"TVmaze: crawl complete, {pages} pages.")
    return pages


def load_raw(directory: Path) -> list[dict]:
    shows = []
    for path in sorted((directory / "raw" / "tvmaze").glob("page-*.json")):
        shows.extend(json.loads(path.read_text(encoding="utf-8")))
    return shows


def to_records(shows: list[dict]) -> list[dict]:
    records = []
    for item in shows:
        title = clean(item.get("name"))
        synopsis = clean(item.get("summary"))
        genres = sorted({norm(genre) for genre in item.get("genres") or [] if norm(genre)})
        if not title or item.get("type") in SKIPPED_TYPES or (len(synopsis) < 40 and not genres):
            continue
        channel = item.get("network") or item.get("webChannel") or {}
        country = (channel.get("country") or {}).get("code") or ""
        show_type = norm(item.get("type") or "")
        premiered, ended = item.get("premiered") or "", item.get("ended") or ""
        image = (item.get("image") or {}).get("medium") or ""
        records.append(dict(
            id=f"tv:tvmaze:{item['id']}", kind="tv", title=title, aliases=[],
            year=int(number(premiered[:4])), end_year=int(number(ended[:4])),
            synopsis=synopsis, genres=genres, tags={f"{show_type} series": 0.6} if show_type and show_type != "scripted" else {},
            studios=[norm(channel.get("name"))] if channel.get("name") else [], creators=[],
            source="", format=item.get("type") or "TV", episodes=0,
            runtime=int(number(item.get("averageRuntime") or item.get("runtime"))),
            country=country, language=item.get("language") or "", network=clean(channel.get("name")),
            rating=round(number((item.get("rating") or {}).get("average")), 2),
            popularity=int(number(item.get("weight"))), favourites=0,
            adult=bool(set(genres) & BLOCKED_TAGS), url=item.get("url") or "", image=image,
            related=[], recs={},
            external={"tvmaze": item["id"], **{key: value for key, value in (item.get("externals") or {}).items() if value}},
        ))
    return records
