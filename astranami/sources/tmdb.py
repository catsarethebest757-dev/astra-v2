"""Optional TMDB API source (set TMDB_API_KEY): keywords, credits, and recommendation edges for TV and movies."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
from pathlib import Path

from ..config import BLOCKED_TAGS, TMDB_API
from ..util import RateLimiter, clean, http, log, norm, number

PAGES = 250


def api_key() -> str:
    return os.environ.get("TMDB_API_KEY", "").strip()


def _get(path: str, key: str, limiter: RateLimiter, **params) -> dict:
    headers = {}
    if len(key) > 40:
        headers["Authorization"] = f"Bearer {key}"
    else:
        params["api_key"] = key
    body, _ = http(f"{TMDB_API}{path}?{urllib.parse.urlencode(params)}", headers=headers, limiter=limiter)
    return json.loads(body)


def crawl(directory: Path, pages: int = PAGES, progress=log) -> int:
    key = api_key()
    if not key:
        return 0
    limiter = RateLimiter(1800)
    root = directory / "raw" / "tmdb"
    total = 0
    for media in ("tv", "movie"):
        folder = root / media
        folder.mkdir(parents=True, exist_ok=True)
        ids: list[int] = []
        for page in range(1, pages + 1):
            listing = folder / f"discover-{page:03d}.json"
            if listing.exists():
                data = json.loads(listing.read_text(encoding="utf-8"))
            else:
                data = _get(f"/discover/{media}", key, limiter, sort_by="vote_count.desc", page=page,
                            include_adult="false")
                listing.write_text(json.dumps(data), encoding="utf-8")
            ids.extend(item["id"] for item in data.get("results", []))
            if page >= int(data.get("total_pages", 0)):
                break
        for position, identifier in enumerate(dict.fromkeys(ids)):
            path = folder / f"{identifier}.json"
            if path.exists():
                continue
            try:
                details = _get(f"/{media}/{identifier}", key, limiter,
                               append_to_response="keywords,recommendations,credits,external_ids")
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    continue
                raise
            path.write_text(json.dumps(details), encoding="utf-8")
            if position % 1000 == 0:
                progress(f"TMDB {media}: {position:,}/{len(ids):,} details")
        total += len(ids)
    return total


def to_records(directory: Path) -> list[dict]:
    records = []
    for media in ("tv", "movie"):
        for path in sorted((directory / "raw" / "tmdb" / media).glob("[0-9]*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            title = clean(item.get("name") or item.get("title"))
            if not title:
                continue
            keywords_block = item.get("keywords") or {}
            keywords = [norm(entry["name"]) for entry in keywords_block.get("results", keywords_block.get("keywords", []))]
            genres = sorted({norm(entry["name"]) for entry in item.get("genres", []) if norm(entry["name"])})
            crew = (item.get("credits") or {}).get("crew", [])
            creators = {f"creator:{norm(person['name'])}" for person in item.get("created_by", [])}
            for person in crew:
                job = person.get("job", "")
                if job == "Director":
                    creators.add(f"director:{norm(person['name'])}")
                elif job in {"Screenplay", "Writer"}:
                    creators.add(f"writer:{norm(person['name'])}")
                elif job == "Original Music Composer":
                    creators.add(f"music:{norm(person['name'])}")
            recs = {}
            for rank, entry in enumerate((item.get("recommendations") or {}).get("results", [])[:20]):
                recs[f"{media}:tmdb:{entry['id']}"] = max(1, 20 - rank)
            date = item.get("first_air_date") or item.get("release_date") or ""
            companies = item.get("networks") or item.get("production_companies") or []
            external = {"tmdb": item["id"]}
            imdb = (item.get("external_ids") or {}).get("imdb_id") or item.get("imdb_id")
            if imdb:
                external["imdb"] = imdb
            run_times = item.get("episode_run_time") or []
            runtime = item.get("runtime") or (run_times[0] if run_times else 0)
            records.append(dict(
                id=f"{media}:tmdb:{item['id']}", kind=media, title=title,
                aliases=[clean(item.get("original_name") or item.get("original_title"))]
                if clean(item.get("original_name") or item.get("original_title")) not in {"", title} else [],
                year=int(number(date[:4])), end_year=int(number((item.get("last_air_date") or "")[:4])),
                synopsis=clean(f"{item.get('overview') or ''} {item.get('tagline') or ''}"),
                genres=genres, tags={keyword: round(max(0.4, 0.9 - 0.02 * position), 3)
                                     for position, keyword in enumerate(keywords[:30]) if keyword},
                studios=[norm(company["name"]) for company in companies[:4]], creators=sorted(creators),
                source="", format="MOVIE" if media == "movie" else (item.get("type") or "TV"),
                episodes=int(number(item.get("number_of_episodes"))) if media == "tv" else 1,
                runtime=int(number(runtime)), country=(item.get("origin_country") or [""])[0],
                language=item.get("original_language") or "", network=companies[0]["name"] if companies else "",
                rating=round(number(item.get("vote_average")), 2), popularity=int(number(item.get("vote_count"))),
                favourites=0, adult=bool(item.get("adult")) or bool(set(keywords) & BLOCKED_TAGS),
                url=f"https://www.themoviedb.org/{media}/{item['id']}",
                image=f"https://image.tmdb.org/t/p/w342{item['poster_path']}" if item.get("poster_path") else "",
                related=[], recs=recs, external=external,
            ))
    return records
