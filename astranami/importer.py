"""Build a catalog and feature index from exported title and vector files, without an embedding model.

Inputs:
- titles: JSON lines (optionally gzipped) with id, media, name, alt, year, format, genres, tags, synopsis, ...
- keys + vectors: a JSON list of title ids and a matching .npy of unit vectors (used as the "whole" aspect)
- training directory (optional): titles.jsonl (key, fam, text), pairs.jsonl (a, b, w)

The plot, themes, and tone aspects are latent-semantic vectors computed here from each title's own text,
so every aspect is distinct even though only one learned vector per title is supplied.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .catalog import blank, clean_edges, cluster_franchises, derive_priors, validate
from .features.text import MOOD_TAGS, _era, _length
from .index import Index
from .store import save_json
from .util import atomic_json, log, norm, sha, tokens, unit

MEDIA = {"anime": "anime", "series": "tv", "tv": "tv", "movie": "movie", "film": "movie"}
NOTE_LINES = {"World rules": "world", "Appeal": "appeal", "Tone": "tone", "Story themes": "story_themes",
              "Audience": "audience"}
NOTE_PATTERN = re.compile(r"^(%s): (.*)$" % "|".join(NOTE_LINES), re.M)
SOURCE_URL = {"anilist": "https://anilist.co/anime/{}", "mal": "https://myanimelist.net/anime/{}",
              "tvmaze": "https://www.tvmaze.com/shows/{}", "tmdb": "https://www.themoviedb.org/movie/{}"}
ASPECT_WIDTH = 192


def _lines(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical(key: str) -> str:
    """Training files prefix movie ids with 'film:'; the title export does not."""
    return key[5:] if key.startswith("film:") else key


def _url(source: str, number: str, kind: str) -> str:
    if source == "tmdb" and kind == "tv":
        return f"https://www.themoviedb.org/tv/{number}"
    return SOURCE_URL[source].format(number) if source in SOURCE_URL else ""


def _record(row: dict) -> dict:
    identifier = row["id"]
    source, _, number = identifier.partition(":")
    names = [name for name in [row.get("original") or "", *(row.get("alt") or [])] if name and name != row["name"]]
    score = float(row.get("score") or 0)
    return blank(
        identifier, row["name"], MEDIA[row["media"]], aliases=names, year=int(row.get("year") or 0),
        synopsis=row.get("synopsis") or "",
        genres=sorted({norm(genre) for genre in row.get("genres") or [] if norm(genre)}),
        tags={norm(name): float(min(max(weight, 0.0), 1.0)) for name, weight in (row.get("tags") or {}).items()
              if norm(name)},
        studios=list(row.get("studios") or []), source=source, format=(row.get("format") or "").upper(),
        episodes=int(row.get("episodes") or 0), runtime=int(row.get("runtime") or 0),
        country=row.get("country") or "", language=row.get("language") or "",
        rating=round(min(max(score / 10, 0.0), 10.0), 2), popularity=max(int(row.get("popularity") or 0), 0),
        image=row.get("image") or "", url=_url(source, number, MEDIA[row["media"]]),
    )


def _apply_training(records: list[dict], training: Path, progress) -> dict:
    """Attach franchise families, curated notes, and recommendation pairs. Returns counts for the log."""
    by_id = {record["id"]: record for record in records}
    families = defaultdict(list)
    notes = 0
    for row in _lines(training / "titles.jsonl"):
        record = by_id.get(canonical(row["key"]))
        if record is None:
            continue
        families[row.get("fam") or row["key"]].append(record["id"])
        found = {NOTE_LINES[name]: text.strip() for name, text in NOTE_PATTERN.findall(row.get("text") or "")}
        if found:
            record["notes"] = found
            notes += 1
    for members in families.values():
        for member in members[1:]:
            by_id[member]["related"].append(members[0])
    pairs = 0
    if (training / "pairs.jsonl").exists():
        for row in _lines(training / "pairs.jsonl"):
            left, right = by_id.get(canonical(row["a"])), by_id.get(canonical(row["b"]))
            if left is None or right is None or left is right:
                continue
            votes = max(1, int(round(10 * float(row.get("w", 0.5)))))
            left["recs"][right["id"]] = max(left["recs"].get(right["id"], 0), votes)
            pairs += 1
    progress(f"Import: {sum(len(m) > 1 for m in families.values()):,} multi-title families, "
             f"{notes:,} titles with curated notes, {pairs:,} recommendation pairs")
    return {"notes": notes, "pairs": pairs}


def aspect_documents(record: dict) -> dict[str, list[str]]:
    """Token lists per aspect. Tags repeat with their weight so strong tags count for more."""
    notes = record.get("notes", {})
    labels = [f"genre {name}" for name in record["genres"]]
    weighted = []
    for name, weight in record["tags"].items():
        weighted.extend(["tag_" + "_".join(tokens(name))] * max(1, int(round(3 * weight))))
    genre_terms = ["genre_" + "_".join(tokens(name)) for name in record["genres"]]
    moods = ["tag_" + "_".join(tokens(name)) for name in record["tags"] if name.lower() in MOOD_TAGS]
    shape = tokens(f"{_era(record['year'])} {_length(record)} {record['kind']}")
    return {
        "plot": tokens(f"{record['synopsis'][:2400]} {notes.get('world', '')}"),
        "themes": weighted + genre_terms + tokens(notes.get("story_themes", "")),
        "tone": (genre_terms + moods * 2 + ["shape_" + term for term in shape]
                 + tokens(f"{notes.get('tone', '')} {notes.get('tone', '')} {notes.get('appeal', '')} "
                          f"{notes.get('audience', '')}")),
    }


def latent_vectors(documents: list[list[str]], width: int = ASPECT_WIDTH, min_count: int = 3) -> np.ndarray:
    """TF-IDF followed by truncated SVD; unit float16 rows. Titles with no usable text get a zero row."""
    counts = Counter(term for document in documents for term in set(document))
    vocabulary = {term: position for position, term in enumerate(
        sorted(term for term, count in counts.items() if min_count <= count <= 0.4 * len(documents)))}
    rows, columns, values = [], [], []
    for position, document in enumerate(documents):
        for term, count in Counter(term for term in document if term in vocabulary).items():
            rows.append(position)
            columns.append(vocabulary[term])
            values.append(1 + np.log(count))
    shape = (len(documents), max(len(vocabulary), 1))
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)
    frequency = np.bincount(matrix.indices, minlength=shape[1])
    matrix = matrix @ sparse.diags(np.log((1 + len(documents)) / (1 + frequency)).astype(np.float32) + 1)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    matrix = sparse.diags(1 / np.maximum(norms, 1e-9)) @ matrix
    rank = min(width, min(shape) - 1)
    if rank < 1:
        return np.zeros((len(documents), width), dtype=np.float16)
    left, singular, _ = svds(matrix.astype(np.float64), k=rank, random_state=0)
    dense = np.zeros((len(documents), width), dtype=np.float32)
    dense[:, :rank] = (left * singular)[:, ::-1]
    return unit(dense).astype(np.float16)


def import_catalog(directory: Path, titles: Path, keys: Path, vectors: Path, training: Path | None = None,
                   progress=log) -> Index:
    directory = Path(directory)
    progress("Import: reading titles and vectors")
    vector_keys = json.loads(Path(keys).read_text(encoding="utf-8"))
    matrix = np.load(vectors, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or len(vector_keys) != matrix.shape[0]:
        raise ValueError("vector keys and vectors must have the same number of rows.")
    row_of = {canonical(key): position for position, key in enumerate(vector_keys)}
    records, skipped = [], 0
    for row in _lines(Path(titles)):
        if row.get("media") not in MEDIA or row["id"] not in row_of or not str(row.get("name") or "").strip():
            skipped += 1
            continue
        records.append(_record(row))
    if skipped:
        progress(f"Import: skipped {skipped:,} titles without a vector, a name, or a known media type")
    stats = _apply_training(records, Path(training), progress) if training else {}
    clean_edges(records)
    cluster_franchises(records)
    derive_priors(records)
    validate(records)
    index = Index(records)
    whole = np.asarray(matrix[[row_of[record["id"]] for record in records]], dtype=np.float32)
    index.aspects["whole"] = unit(whole).astype(np.float16)
    documents = [aspect_documents(record) for record in records]
    for aspect in ("plot", "themes", "tone"):
        progress(f"Import: computing the {aspect} aspect")
        # Same width as the supplied vectors: the taste adapter treats its input as four equal blocks.
        index.aspects[aspect] = latent_vectors([document[aspect] for document in documents], whole.shape[1])
    index.build_sparse()
    index.set_graph(index.all_edges())
    progress("Import: saving the catalog and index")
    atomic_json(directory / "catalog.json", records)
    atomic_json(directory / "catalog_meta.json", {
        "counts": dict(Counter(record["kind"] for record in records)),
        "rec_edges": sum(len(record["recs"]) for record in records),
        "franchises": len({record["franchise"] for record in records}),
        "imported": True, **stats,
    })
    # Models trained on whatever catalog was here before describe different titles.
    (directory / "index" / "adapter.npy").unlink(missing_ok=True)
    shutil.rmtree(directory / "model", ignore_errors=True)
    index.save(directory / "index")
    save_json(directory / "index", "meta", {"encoder": "imported", "catalog": sha("\n".join(r["id"] for r in records))})
    progress(f"Import: saved {index.size:,} titles.")
    return index
