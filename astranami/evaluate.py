"""Held-out recommendation benchmark against lexical, embedding, graph, and legacy-fusion baselines."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .engine import Engine
from .index import Index, dense_scores, top_k
from .learn.pairs import Split, adjacency, relevant
from .learn.ranker import Ranker
from .util import atomic_json, log


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def ranking_metrics(ranked: list[int], grades: dict[int, float], cutoff: int) -> tuple[float, float]:
    chosen = ranked[:cutoff]
    gains = [grades.get(item, 0.0) for item in chosen]
    ideal = sorted(grades.values(), reverse=True)[:cutoff]
    idcg = _dcg(ideal)
    ndcg = _dcg(gains) / idcg if idcg else 0.0
    recall = sum(item in grades for item in chosen) / max(len(grades), 1)
    return ndcg, recall


def reciprocal(ranked: list[int], grades: dict[int, float]) -> float:
    for position, item in enumerate(ranked):
        if item in grades:
            return 1.0 / (position + 1)
    return 0.0


def _rrf(channels: list[np.ndarray], eligible: np.ndarray) -> np.ndarray:
    score = np.zeros(len(eligible), dtype=np.float64)
    for values in channels:
        candidates = np.flatnonzero(eligible & np.isfinite(values) & (values > 1e-8))
        if not len(candidates):
            continue
        order = candidates[np.argsort(-values[candidates], kind="stable")]
        score[order] += 1.0 / (60.0 + np.arange(1, len(order) + 1))
    return score


def _baseline(index: Index, seed: int, method: str, eligible: np.ndarray) -> list[int]:
    if method == "bm25":
        scores = index.bm25.item_scores(seed)
    elif method == "tags":
        scores = np.asarray((index.tags.soft @ index.tags.soft[seed].T).todense()).ravel()
    elif method == "embed":
        scores = dense_scores(index, "whole", np.asarray(index.aspects["whole"][seed], dtype=np.float32))
    elif method == "graph":
        scores = index.graph.personalized_scores(seed) if index.graph is not None else np.zeros(index.size)
    elif method == "adapter":
        scores = index.adapter @ index.adapter[seed] if index.adapter is not None else np.zeros(index.size)
    elif method == "legacy_rrf":
        scores = _rrf([
            index.bm25.item_scores(seed),
            np.asarray((index.tags.matrix @ index.tags.matrix[seed].T).todense()).ravel(),
            dense_scores(index, "whole", np.asarray(index.aspects["whole"][seed], dtype=np.float32)),
        ], eligible)
    else:
        raise ValueError(f"Unknown baseline {method}.")
    return top_k(np.asarray(scores, dtype=np.float64), 50, eligible).astype(int).tolist()


def _eligible(index: Index, seed: int, hidden: set[int]) -> np.ndarray:
    mask = ~index.adult
    mask = mask & (index.franchise != index.franchise[seed])
    mask[seed] = False
    if hidden:
        mask[np.fromiter(hidden, dtype=np.int64, count=len(hidden))] = False
    return mask


def _summarize(rows: list[tuple]) -> dict:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "ndcg@10": round(float(values[:, 0].mean()), 4),
        "recall@10": round(float(values[:, 1].mean()), 4),
        "recall@50": round(float(values[:, 2].mean()), 4),
        "mrr": round(float(values[:, 3].mean()), 4),
        "latency_p50_ms": round(float(np.percentile(values[:, 4], 50)), 2),
        "latency_p95_ms": round(float(np.percentile(values[:, 5], 95)), 2),
    }


def _quality(index: Index, rankings: list[list[int]]) -> dict:
    top = [item for ranking in rankings for item in ranking[:10]]
    if not top:
        return {"coverage": 0.0, "novelty": 0.0, "diversity": 0.0}
    coverage = len(set(top)) / index.size
    novelty = float(np.mean(-np.log2(index.pop[top] + 0.01)))
    diversities = []
    vectors = index.adapter if index.adapter is not None else index.dense32("whole")
    for ranking in rankings:
        picked = ranking[:10]
        if len(picked) < 2:
            continue
        block = np.asarray(vectors[picked], dtype=np.float32)
        similarity = block @ block.T
        diversities.append(float((1 - similarity).sum() / (len(picked) * (len(picked) - 1))))
    return {
        "coverage": round(coverage, 4),
        "novelty": round(novelty, 4),
        "diversity": round(float(np.mean(diversities)) if diversities else 0.0, 4),
    }


def evaluate_split(index: Index, split: Split, ranker: Ranker | None = None, seed_limit: int = 400,
                   sample_seed: int = 13, progress=log) -> dict:
    """Cold start hides every test edge. Warm start reveals half and asks the model to rank the rest."""
    rng = np.random.default_rng(sample_seed)
    seeds = np.array(split.test_seeds)
    if len(seeds) > seed_limit:
        seeds = np.sort(rng.choice(seeds, seed_limit, replace=False))
    if len(seeds) == 0:
        raise ValueError("The split has no test seeds.")
    full_edges = index.all_edges()
    full = adjacency(full_edges)
    methods = ["bm25", "tags", "embed", "graph", "legacy_rrf", "engine"]
    if index.adapter is not None:
        methods.insert(4, "adapter")
    if ranker is not None:
        methods.insert(-1, "heuristic")
    engine = Engine(index, ranker)
    heuristic = Engine(index) if ranker is not None else engine
    report = {"seeds": int(len(seeds)), "cold": {}, "warm": {}, "lists": {}}
    cold_lists: list[list[int]] = []
    try:
        for mode, edges, revealed in (
            ("cold", split.cold_graph, {}),
            ("warm", split.warm_graph, split.warm_visible),
        ):
            progress(f"Evaluation: preparing {mode} graph")
            index.set_graph(edges)
            per_method = {method: [] for method in methods}
            for seed_number, seed in enumerate(seeds, start=1):
                seed = int(seed)
                hidden = set(revealed.get(seed, ()))
                grades = relevant(index, seed, full.get(seed, {}), hidden)
                if not grades:
                    continue
                eligible = _eligible(index, seed, hidden)
                for method in methods:
                    started = time.perf_counter()
                    if method == "engine":
                        ranked = engine.rank_seed(seed, limit=50, eligible=eligible)
                    elif method == "heuristic":
                        ranked = heuristic.rank_seed(seed, limit=50, eligible=eligible)
                    else:
                        ranked = _baseline(index, seed, method, eligible)
                    elapsed = (time.perf_counter() - started) * 1000
                    ndcg, recall10 = ranking_metrics(ranked, grades, 10)
                    _, recall50 = ranking_metrics(ranked, grades, 50)
                    per_method[method].append((ndcg, recall10, recall50, reciprocal(ranked, grades), elapsed, elapsed))
                    if mode == "cold" and method == "engine":
                        cold_lists.append(ranked)
                if seed_number % 25 == 0 or seed_number == len(seeds):
                    progress(f"Evaluation: {mode} seeds {seed_number:,}/{len(seeds):,}")
            report[mode] = {method: _summarize(rows) for method, rows in per_method.items() if rows}
            progress(f"{mode}: " + ", ".join(
                f"{method} nDCG {metrics['ndcg@10']:.3f}" for method, metrics in report[mode].items()))
    finally:
        progress("Evaluation: restoring the full recommendation graph")
        index.set_graph(full_edges)
    report["lists"] = _quality(index, cold_lists)
    cold = report["cold"]
    report["cold_winner"] = max(cold, key=lambda method: (cold[method]["ndcg@10"], method == "engine")) if cold else ""
    report["note"] = ("Held-out AniList/TMDB recommendation edges. legacy_rrf is reciprocal-rank fusion of "
                      "BM25, tag cosine, and the whole-aspect embedding, the shape of the previous engine. "
                      "This is a win on this benchmark, not a claim about every public ranker.")
    _print_table(report)
    return report


def _print_table(report: dict) -> None:
    header = f"{'method':<14} {'nDCG@10':>8} {'R@10':>8} {'R@50':>8} {'MRR':>8} {'p95 ms':>8}"
    for mode in ("cold", "warm"):
        log(f"\n{mode}  ({report['seeds']} seeds)")
        log(header)
        for method, metrics in report[mode].items():
            log(f"{method:<14} {metrics['ndcg@10']:8.4f} {metrics['recall@10']:8.4f} "
                f"{metrics['recall@50']:8.4f} {metrics['mrr']:8.4f} {metrics['latency_p95_ms']:8.1f}")
    lists = report["lists"]
    log(f"engine cold lists: coverage {lists['coverage']:.4f}, novelty {lists['novelty']:.3f}, "
        f"diversity {lists['diversity']:.3f}, winner {report['cold_winner']}")


def write_report(path: Path, report: dict) -> None:
    atomic_json(path, report)
    log(f"Wrote {path}")
