"""How much neighbor quality smaller vector formats would cost, measured against the stored vectors."""
from __future__ import annotations

import numpy as np

from .config import ASPECTS
from .index import Index


def _neighbors(scores: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    scores[np.arange(len(rows)), rows] = -np.inf
    return np.argpartition(-scores, k - 1, axis=1)[:, :k]


def _recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(np.intersect1d(a, b)) / len(b) for a, b in zip(found, truth)]))


def compare_formats(index: Index, samples: int = 300, k: int = 20, seed: int = 0) -> dict:
    """Recall of each format's top-k neighbors against the stored vectors' top-k, per aspect, with its size.

    The index already stores float16, so that is the baseline; float32 originals are not kept.
    """
    k = max(1, min(k, index.size - 1))
    rows = np.random.default_rng(seed).choice(index.size, min(samples, index.size), replace=False)
    report = {"titles": index.size, "samples": int(len(rows)), "k": k, "aspects": {}}
    for aspect in ASPECTS:
        exact = index.dense32(aspect)
        width = exact.shape[1]
        truth = _neighbors(exact[rows] @ exact.T, rows, k)
        scale = float(np.abs(exact).max()) or 1.0
        int8 = np.round(exact / scale * 127).astype(np.int8).astype(np.float32)
        signs = np.where(exact >= 0, 1.0, -1.0).astype(np.float32)
        formats = {"int8": (int8, width), "binary": (signs, (width + 7) // 8)}
        entry = {"float16 (stored)": {"recall": 1.0, "megabytes": round(index.size * 2 * width / 1e6, 2)}}
        for name, (matrix, bytes_per_row) in formats.items():
            entry[name] = {"recall": round(_recall(_neighbors(matrix[rows] @ matrix.T, rows, k), truth), 4),
                           "megabytes": round(index.size * bytes_per_row / 1e6, 2)}
        # Binary search that rescores a 4x shortlist with int8 vectors: the usual two-stage setup.
        depth = min(4 * k, index.size - 1)
        shortlist = _neighbors(signs[rows] @ signs.T, rows, depth)
        rescored = np.stack([found[np.argsort(-(int8[found] @ int8[row]))[:k]] for found, row in zip(shortlist, rows)])
        entry["binary+int8 rescore"] = {"recall": round(_recall(rescored, truth), 4),
                                        "megabytes": round(index.size * ((width + 7) // 8 + width) / 1e6, 2)}
        report["aspects"][aspect] = entry
    return report
