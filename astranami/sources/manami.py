"""Anime Offline Database: extra synonyms from MAL, Kitsu, AniDB, etc., keyed to AniList IDs."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import MANAMI_URL
from ..util import clean, download, log

ANILIST_SOURCE = re.compile(r"anilist\.co/anime/(\d+)")


def fetch(directory: Path) -> Path:
    path = directory / "raw" / "manami.json"
    download(MANAMI_URL, path, max_bytes=300 * 1024 * 1024)
    return path


def anilist_ids(directory: Path) -> list[int]:
    payload = json.loads((directory / "raw" / "manami.json").read_text(encoding="utf-8"))
    output = set()
    for item in payload.get("data", []):
        for source in item.get("sources", []):
            match = ANILIST_SOURCE.search(source)
            if match:
                output.add(int(match.group(1)))
    return sorted(output)


def synonyms_by_anilist(directory: Path) -> dict[str, list[str]]:
    path = directory / "raw" / "manami.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        log(f"manami: unreadable snapshot ignored ({error})")
        return {}
    output = {}
    for item in payload.get("data", []):
        for source in item.get("sources", []):
            match = ANILIST_SOURCE.search(source)
            if match:
                names = [clean(item.get("title"))] + [clean(name) for name in item.get("synonyms", [])]
                output[f"anime:al:{match.group(1)}"] = [name for name in names if name][:40]
                break
    return output


def merge_synonyms(records: list[dict], synonyms: dict[str, list[str]]) -> None:
    for record in records:
        extra = synonyms.get(record["id"])
        if extra:
            record["aliases"] = sorted(set(record["aliases"]) | {name for name in extra if name != record["title"]})
