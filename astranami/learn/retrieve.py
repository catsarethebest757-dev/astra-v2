"""Batched nearest-neighbor lookup used while building ranker training lists."""
from __future__ import annotations

import numpy as np


def batch_neighbors(matrix: np.ndarray, seeds: np.ndarray, k: int, progress=None, label: str = "neighbors") -> np.ndarray:
    """For each seed row, the k other rows with the highest dot product. Returns [len(seeds), k]."""
    import torch

    seeds = np.asarray(seeds, dtype=np.int64)
    if len(seeds) == 0:
        return np.zeros((0, 0), dtype=np.int64)
    k = min(k, max(int(matrix.shape[0]) - 1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        mat = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)).to(device=device, dtype=dtype)
    except RuntimeError:
        device, dtype = torch.device("cpu"), torch.float32
        mat = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    output = np.empty((len(seeds), k), dtype=np.int64)
    step = 128 if device.type == "cuda" else 32
    if progress:
        progress(f"Ranker: finding {label} for {len(seeds):,} seeds on {device}")
    with torch.no_grad():
        for start in range(0, len(seeds), step):
            batch = torch.from_numpy(seeds[start:start + step]).to(device)
            scores = mat[batch].float() @ mat.float().T
            scores[torch.arange(len(batch), device=device), batch] = -1e9
            output[start:start + len(batch)] = torch.topk(scores, k).indices.cpu().numpy()
            if progress and ((start // step + 1) % max(1, (len(seeds) // step) // 10) == 0 or start + step >= len(seeds)):
                progress(f"Ranker: {label} neighbors {min(start + step, len(seeds)):,}/{len(seeds):,}")
    return output
