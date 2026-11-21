"""Small helpers to persist numpy/scipy artifacts without pickle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .util import atomic_write


def save_array(directory: Path, name: str, array: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{name}.part.npy"
    np.save(temporary, array)
    temporary.replace(directory / f"{name}.npy")


def load_array(directory: Path, name: str, mmap: bool = False) -> np.ndarray:
    return np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)


def save_sparse(directory: Path, name: str, matrix) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{name}.part.npz"
    sparse.save_npz(temporary, sparse.csr_matrix(matrix))
    temporary.replace(directory / f"{name}.npz")


def load_sparse(directory: Path, name: str) -> sparse.csr_matrix:
    return sparse.load_npz(directory / f"{name}.npz").tocsr()


def save_json(directory: Path, name: str, value) -> None:
    atomic_write(directory / f"{name}.json", json.dumps(value, ensure_ascii=False))


def load_json(directory: Path, name: str):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
