"""Resumable AniList crawl: every anime with weighted tags, creators, relations, and community recommendations."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import ANILIST_URL, BLOCKED_TAGS
from ..util import RateLimiter, clean, http, log, norm, number

QUERY = """query($ids:[Int]){Page(page:1,perPage:50){
media(type:ANIME,id_in:$ids){id idMal siteUrl title{romaji english native} synonyms
description(asHtml:false) genres tags{name rank isGeneralSpoiler isMediaSpoiler category}
studios(isMain:true){nodes{name}} staff(perPage:10,sort:RELEVANCE){edges{role node{id name{full}}}}
source format episodes duration seasonYear startDate{year} endDate{year} averageScore meanScore popularity
favourites isAdult countryOfOrigin coverImage{large}
relations{edges{relationType node{id type}}}
recommendations(perPage:25,sort:RATING_DESC){nodes{rating mediaRecommendation{id}}}}}}"""

FRANCHISE_RELATIONS = {"SEQUEL", "PREQUEL", "PARENT", "SIDE_STORY", "SPIN_OFF", "ALTERNATIVE",
                       "SUMMARY", "COMPILATION", "CONTAINS"}
CREATOR_ROLES = {"director": "director", "series composition": "writer", "script": "writer",
                 "original creator": "creator", "original story": "creator", "music": "music",
                 "character design": "design", "chief animation director": "design"}


def raw_path(directory: Path) -> Path:
    return directory / "raw" / "anilist.jsonl"


def done_path(directory: Path) -> Path:
    return directory / "raw" / "anilist.done"


def load_raw(directory: Path) -> dict[int, dict]:
    media: dict[int, dict] = {}
    path = raw_path(directory)
    if not path.exists():
        return media
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            media[item["id"]] = item
    return media


def crawl(directory: Path, anilist_ids: list[int], progress=log) -> int:
    """Fetch the given anime IDs in batches of 50 (AniList caps page depth at 5000 entries, so IDs come from
    the offline database), appending to a JSONL checkpoint. Safe to interrupt and rerun."""
    if done_path(directory).exists():
        return len(load_raw(directory))
    path = raw_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_file = directory / "raw" / "anilist.missing"
    unavailable = set(json.loads(missing_file.read_text(encoding="utf-8"))) if missing_file.exists() else set()
    fetched = set(load_raw(directory))
    pending = [identifier for identifier in sorted(set(anilist_ids)) if identifier not in fetched | unavailable]
    progress(f"AniList: {len(fetched):,} cached, {len(pending):,} to fetch (~{len(pending) / 50 / 27:.0f} min).")
    limiter = RateLimiter(27)
    started = time.monotonic()
    for start in range(0, len(pending), 50):
        batch = pending[start:start + 50]
        payload = json.dumps({"query": QUERY, "variables": {"ids": batch}}).encode()
        body, headers = http(ANILIST_URL, data=payload, limiter=limiter,
                             headers={"Content-Type": "application/json"})
        response = json.loads(body)
        if response.get("errors"):
            raise RuntimeError(f"AniList error: {response['errors'][:1]}")
        media = response["data"]["Page"]["media"]
        with path.open("a", encoding="utf-8") as handle:
            for item in media:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        unavailable |= set(batch) - {item["id"] for item in media}
        missing_file.write_text(json.dumps(sorted(unavailable)), encoding="utf-8")
        if number(headers.get("X-RateLimit-Remaining"), 10) <= 1:
            time.sleep(62)
        if (start // 50) % 20 == 0:
            progress(f"AniList: {start + len(batch):,}/{len(pending):,} ({time.monotonic() - started:.0f}s)")
    count = len(load_raw(directory))
    done_path(directory).write_text(str(count), encoding="utf-8")
    progress(f"AniList: crawl complete, {count:,} anime.")
    return count


def _creators(item: dict) -> list[str]:
    output = []
    for edge in (item.get("staff") or {}).get("edges", []):
        role = norm(edge.get("role", ""))
        for prefix, label in CREATOR_ROLES.items():
            if role.startswith(prefix):
                output.append(f"{label}:{norm(edge['node']['name']['full'])}")
                break
    return sorted(set(output))


def to_records(media: dict[int, dict]) -> list[dict]:
    records = []
    for item in media.values():
        titles = item.get("title") or {}
        title = clean(titles.get("english") or titles.get("romaji") or titles.get("native"))
        if not title:
            continue
        tags = {}
        for tag in item.get("tags") or []:
            if tag.get("isGeneralSpoiler") or tag.get("isMediaSpoiler"):
                continue
            name = norm(tag["name"])
            if name:
                tags[name] = round(number(tag.get("rank"), 50) / 100, 3)
        genres = sorted({norm(genre) for genre in item.get("genres") or [] if norm(genre)})
        aliases = [clean(titles.get(key)) for key in ("romaji", "english", "native")]
        aliases += [clean(alias) for alias in item.get("synonyms") or []]
        related = sorted({f"anime:al:{edge['node']['id']}" for edge in (item.get("relations") or {}).get("edges", [])
                          if edge["relationType"] in FRANCHISE_RELATIONS and edge["node"]["type"] == "ANIME"})
        recommendations = {}
        for node in (item.get("recommendations") or {}).get("nodes", []):
            target = node.get("mediaRecommendation")
            if target and number(node.get("rating")) > 0:
                recommendations[f"anime:al:{target['id']}"] = int(node["rating"])
        year = int(number(item.get("seasonYear")) or number((item.get("startDate") or {}).get("year")))
        score = number(item.get("averageScore") or item.get("meanScore")) / 10
        all_tags = set(tags) | set(genres)
        records.append(dict(
            id=f"anime:al:{item['id']}", kind="anime", title=title,
            aliases=sorted({alias for alias in aliases if alias and alias != title}),
            year=year, end_year=int(number((item.get("endDate") or {}).get("year"))),
            synopsis=clean(item.get("description")), genres=genres, tags=tags,
            studios=sorted({norm(node["name"]) for node in (item.get("studios") or {}).get("nodes", [])}),
            creators=_creators(item), source=norm(item.get("source") or ""), format=item.get("format") or "",
            episodes=int(number(item.get("episodes"))), runtime=int(number(item.get("duration"))),
            country=item.get("countryOfOrigin") or "", language="", network="",
            rating=round(score, 2), popularity=int(number(item.get("popularity"))),
            favourites=int(number(item.get("favourites"))),
            adult=bool(item.get("isAdult")) or bool(all_tags & BLOCKED_TAGS),
            url=item.get("siteUrl") or f"https://anilist.co/anime/{item['id']}",
            image=((item.get("coverImage") or {}).get("large") or ""),
            related=related, recs=recommendations,
            external={"anilist": item["id"], **({"mal": item["idMal"]} if item.get("idMal") else {})},
        ))
    return records
