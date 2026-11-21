from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(".aftercredits")
USER_AGENT = "Astranami/2.0 (local recommendation engine)"

ANILIST_URL = "https://graphql.anilist.co"
MANAMI_URL = (
    "https://github.com/manami-project/anime-offline-database/"
    "releases/latest/download/anime-offline-database-minified.json"
)
TVMAZE_URL = "https://api.tvmaze.com/shows?page={}"
TMDB_API = "https://api.themoviedb.org/3"
MOVIE_CSV_URL = (
    "https://raw.githubusercontent.com/vamshi121/"
    "TMDB-5000-Movie-Dataset/main/tmdb_5000_movies.csv"
)

KINDS = ("anime", "tv", "movie")
ASPECTS = ("plot", "themes", "tone", "whole")
BLOCKED_TAGS = {"hentai", "pornography", "pornographic", "explicit sex", "erotica"}

ENCODERS = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "jina5": "jinaai/jina-embeddings-v5-text-small",
}

SOURCES = [
    {"name": "AniList GraphQL API", "url": "https://anilist.co",
     "terms": "AniList API terms; non-commercial use, attribute AniList"},
    {"name": "Anime Offline Database", "url": "https://github.com/manami-project/anime-offline-database",
     "terms": "ODbL 1.0 + DbCL 1.0"},
    {"name": "TVmaze", "url": "https://www.tvmaze.com/api",
     "terms": "CC BY-SA 4.0; attribution and ShareAlike apply"},
    {"name": "TMDB 5000 historical dataset mirror", "url": "https://github.com/vamshi121/TMDB-5000-Movie-Dataset",
     "terms": "Historical TMDB data; this product uses TMDB data but is not endorsed or certified by TMDB"},
]


@dataclass
class BuildConfig:
    encoder: str = "qwen3"
    batch_size: int = 32
    max_tokens: int = 384
    holdout_fraction: float = 0.10
    ranker_fraction: float = 0.15
    seed: int = 13
    adapter_dim: int = 256
    adapter_epochs: int = 12
    ranker_epochs: int = 30
    candidates: int = 800
    tvmaze_min_weight: int = 0
    extra: dict = field(default_factory=dict)
