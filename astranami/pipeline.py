"""Bootstrap sources, build features, train, and load a serving engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .catalog import build as build_catalog
from .catalog import load as load_catalog
from .config import BuildConfig
from .engine import Engine
from .index import Index
from .learn.adapter import project, train_adapter
from .learn.pairs import adjacency, make_split
from .learn.ranker import load_ranker, save_ranker, train_ranker
from .sources import anilist, manami, movies, tmdb, tvmaze
from .store import load_json, save_array, save_json
from .util import log, sha


def bootstrap(directory: Path, progress=log, tv_pages: int = 1000) -> None:
    directory = Path(directory)
    progress("Fetching the anime offline database")
    manami.fetch(directory)
    progress("Crawling AniList")
    anilist.crawl(directory, manami.anilist_ids(directory), progress=progress)
    progress("Crawling TVmaze")
    tvmaze.crawl(directory, max_pages=tv_pages, progress=progress)
    progress("Fetching the movie catalog")
    try:
        movies.fetch(directory)
    except Exception as error:
        progress(f"Movie CSV unavailable ({error})")
    if tmdb.api_key():
        progress("Crawling TMDB")
        tmdb.crawl(directory, progress=progress)
    progress("Building the catalog")
    build_catalog(directory)


def build_features(directory: Path, config: BuildConfig | None = None, progress=log, force: bool = False) -> None:
    config = config or BuildConfig()
    directory = Path(directory)
    index_dir = directory / "index"
    if not force and (index_dir / "aspect-whole.npy").exists():
        progress("Feature index already exists.")
        return
    records = load_catalog(directory) if (directory / "catalog.json").exists() else build_catalog(directory)
    index = Index(records)
    from .features.embed import Encoder
    encoder = Encoder(config.encoder, cache_dir=directory / "cache", batch_size=config.batch_size,
                      max_tokens=config.max_tokens)
    progress(f"Encoding aspects with {encoder.model_id}")
    index.embed(encoder)
    index.build_sparse(encoder)
    index.save(index_dir)
    save_json(index_dir, "meta", {"encoder": config.encoder, "catalog": sha("\n".join(record["id"] for record in records))})
    progress(f"Saved features for {index.size:,} titles.")


def _load_index(directory: Path) -> tuple[Index, str]:
    records = load_catalog(directory)
    index = Index.load(records, directory / "index")
    encoder = "qwen3"
    if (directory / "index" / "meta.json").exists():
        encoder = load_json(directory / "index", "meta").get("encoder", encoder)
    return index, encoder


def train_models(directory: Path, config: BuildConfig | None = None, progress=log, refit: bool = True,
                 run_eval: bool = True) -> None:
    config = config or BuildConfig()
    directory = Path(directory)
    progress("Training: loading the catalog and feature index")
    index, encoder_name = _load_index(directory)
    progress(f"Training: loaded {index.size:,} titles; preparing recommendation splits")
    edges = index.all_edges()
    if len(edges) < 8:
        raise ValueError("Need recommendation edges before training. Crawl AniList (and TMDB, if you have a key) first.")
    split = make_split(index, edges, config.holdout_fraction, config.ranker_fraction, config.seed)
    progress(f"Training the held-out taste adapter on {len(split.adapter_edges):,} edges")
    adapter = train_adapter(index, split.adapter_edges, epochs=config.adapter_epochs,
                            output_dim=config.adapter_dim, seed=config.seed, progress=progress)
    progress("Training: projecting the held-out adapter onto all titles")
    index.adapter = project(adapter, index)
    progress("Training: preparing the held-out recommendation graph")
    index.set_graph(split.ranker_graph)
    progress("Training the held-out listwise ranker")
    ranker = train_ranker(index, split.ranker_seeds, split.ranker_graph, epochs=config.ranker_epochs,
                          seed=config.seed, progress=progress, label_edges=split.cold_graph)
    model_dir = directory / "model"
    save_ranker(model_dir / "ranker-heldout.npz", ranker)
    save_array(model_dir, "adapter-heldout", index.adapter.astype(np.float16))
    ranker_enabled = True
    if run_eval:
        progress("Scoring held-out seeds against the baselines")
        from .evaluate import evaluate_split, write_report
        report = evaluate_split(index, split, ranker, sample_seed=config.seed, progress=progress)
        write_report(directory / "report.json", report)
        ranker_enabled = all(
            report[mode]["engine"]["ndcg@10"] >= report[mode]["heuristic"]["ndcg@10"]
            for mode in ("cold", "warm")
        )
        if not ranker_enabled:
            progress("Held-out ranker underperformed; serving with the stronger heuristic scorer")
    if refit:
        progress("Refitting the adapter and ranker on the full graph for serving")
        index.set_graph(edges)
        adapter = train_adapter(index, edges, epochs=config.adapter_epochs, output_dim=config.adapter_dim,
                                seed=config.seed, progress=progress)
        progress("Training: projecting the final adapter onto all titles")
        index.adapter = project(adapter, index)
        if ranker_enabled:
            nodes = np.array(sorted(adjacency(edges)), dtype=np.int64)
            ranker = train_ranker(index, nodes, edges, epochs=config.ranker_epochs, seed=config.seed,
                                  progress=progress)
        else:
            progress("Training: skipped ranker refit because held-out evaluation rejected it")
    index.set_graph(edges)
    progress("Training: saving the final index and model")
    index.save(directory / "index")
    save_ranker(model_dir / "ranker.npz", ranker)
    save_json(model_dir, "meta", {
        "encoder": encoder_name,
        "catalog": sha("\n".join(record["id"] for record in index.records)),
        "refit": refit,
        "ranker_enabled": ranker_enabled,
    })
    progress("Training complete.")


def load_engine(directory: Path, progress=log) -> Engine:
    directory = Path(directory)
    if not (directory / "catalog.json").exists():
        raise FileNotFoundError("No catalog yet. Run: python -m astranami bootstrap")
    if not (directory / "index" / "aspect-whole.npy").exists():
        raise FileNotFoundError("No feature index yet. Run: python -m astranami build")
    progress("Stage 4/4: reading the catalog and feature index")
    index, encoder_name = _load_index(directory)
    progress(f"Stage 4/4: loaded {index.size:,} titles and recommendation graph")
    ranker_path = directory / "model" / "ranker.npz"
    model_meta = directory / "model" / "meta.json"
    enabled = load_json(directory / "model", "meta").get("ranker_enabled", True) if model_meta.exists() else True
    ranker = None
    if ranker_path.exists() and enabled:
        try:
            ranker = load_ranker(ranker_path)
        except ValueError as error:
            progress(f"Stage 4/4: {error} Using the heuristic scorer until then.")
    if ranker_path.exists() and not enabled:
        progress("Stage 4/4: using the heuristic scorer selected by held-out evaluation")
    engine = Engine(index, ranker, encoder_name=encoder_name)
    progress("Stage 4/4: engine ready")
    return engine


def evaluate_saved(directory: Path, config: BuildConfig | None = None) -> dict:
    config = config or BuildConfig()
    directory = Path(directory)
    index, _ = _load_index(directory)
    heldout = directory / "model" / "adapter-heldout.npy"
    ranker_path = directory / "model" / "ranker-heldout.npz"
    if not heldout.exists() or not ranker_path.exists():
        raise FileNotFoundError("No held-out checkpoint. Run: python -m astranami train")
    adapter = np.load(heldout)
    if adapter.shape[0] != index.size:
        raise ValueError("Held-out adapter does not match this catalog. Retrain.")
    index.adapter = np.asarray(adapter, dtype=np.float32)
    ranker = load_ranker(ranker_path)
    split = make_split(index, index.all_edges(), config.holdout_fraction, config.ranker_fraction, config.seed)
    from .evaluate import evaluate_split, write_report
    report = evaluate_split(index, split, ranker, sample_seed=config.seed)
    write_report(directory / "report.json", report)
    return report


def prepare(directory: Path, config: BuildConfig | None = None, progress=log, train: bool = True) -> Engine:
    directory = Path(directory)
    config = config or BuildConfig()
    if not (directory / "catalog.json").exists():
        progress("Stage 1/4: sources and catalog")
        bootstrap(directory, progress=progress)
    else:
        progress("Stage 1/4: catalog already built")
    if not (directory / "index" / "aspect-whole.npy").exists():
        progress("Stage 2/4: aspect embeddings and sparse features")
        build_features(directory, config, progress=progress)
    else:
        progress("Stage 2/4: feature index already built")
    if train and not (directory / "model" / "ranker.npz").exists():
        imported = load_json(directory / "index", "meta").get("encoder") == "imported"
        if imported and len(_load_index(directory)[0].all_edges()) < 8:
            progress("Stage 3/4: no recommendation pairs were imported; using the heuristic scorer")
        else:
            progress("Stage 3/4: training the taste adapter and listwise ranker")
            train_models(directory, config, progress=progress)
    else:
        progress("Stage 3/4: model already on disk" if (directory / "model" / "ranker.npz").exists()
                 else "Stage 3/4: training skipped")
    progress("Stage 4/4: loading the engine")
    return load_engine(directory, progress=progress)
