"""Score the engine and simple baselines against an external judged file (seed -> relevant titles)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ASPECTS
from .engine import Engine
from .evaluate import ranking_metrics, reciprocal
from .importer import canonical
from .index import Index, top_k
from .util import log


def _baseline(index: Index, seed: int, method: str) -> np.ndarray:
    if method == "bm25":
        return index.bm25.item_scores(seed)
    if method == "tags":
        return np.asarray((index.tags.soft @ index.tags.soft[seed].T).todense()).ravel()
    if method == "adapter":
        return index.adapter @ index.adapter[seed]
    matrix = index.dense32(method)
    return matrix @ matrix[seed]


def benchmark_file(index: Index, path: Path, ranker=None, limit: int | None = None, progress=log) -> dict:
    """Each group in the file lists seeds with relevant ids; candidates come from that group's pool if given.

    A seed's own franchise is never a candidate. Only seeds and relevant titles present in the catalog count.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pools = data.get("pools", {})
    methods = [*ASPECTS, "bm25", "tags"] + (["adapter"] if index.adapter is not None else []) + ["heuristic"]
    engines = {"heuristic": Engine(index)}
    if ranker is not None:
        methods.append("engine")
        engines["engine"] = Engine(index, ranker)
    report = {}
    for group, cases in data.items():
        if group == "pools" or not isinstance(cases, list) or not cases or "relevant" not in cases[0]:
            continue
        pool = np.ones(index.size, dtype=bool)
        if group in pools:
            pool[:] = False
            pool[[index.ids[key] for key in map(canonical, pools[group]) if key in index.ids]] = True
        pool &= ~index.adult
        rows = {method: [] for method in methods}
        for case in cases[:limit]:
            seed = index.ids.get(canonical(case["seed"]))
            if seed is None:
                continue
            grades = {index.ids[key]: 1.0 for key in map(canonical, case["relevant"]) if key in index.ids}
            eligible = pool & (index.franchise != index.franchise[seed])
            grades = {item: grade for item, grade in grades.items() if eligible[item]}
            if not grades:
                continue
            for method in methods:
                started = time.perf_counter()
                if method in engines:
                    ranked = engines[method].rank_seed(seed, limit=50, eligible=eligible)
                else:
                    scores = np.asarray(_baseline(index, seed, method), dtype=np.float64)
                    ranked = top_k(scores, 50, eligible).astype(int).tolist()
                elapsed = (time.perf_counter() - started) * 1000
                ndcg, recall10 = ranking_metrics(ranked, grades, 10)
                rows[method].append((ndcg, recall10, ranking_metrics(ranked, grades, 50)[1],
                                     reciprocal(ranked, grades), elapsed))
        report[group] = {"seeds": len(rows[methods[0]])}
        for method in methods:
            values = np.asarray(rows[method], dtype=np.float64)
            if len(values):
                report[group][method] = {
                    "ndcg@10": round(float(values[:, 0].mean()), 4), "recall@10": round(float(values[:, 1].mean()), 4),
                    "recall@50": round(float(values[:, 2].mean()), 4), "mrr": round(float(values[:, 3].mean()), 4),
                    "latency_p50_ms": round(float(np.percentile(values[:, 4], 50)), 1)}
        progress(f"{group} ({report[group]['seeds']} seeds): " + ", ".join(
            f"{method} {report[group][method]['ndcg@10']:.3f}" for method in methods if method in report[group]))
    return report
