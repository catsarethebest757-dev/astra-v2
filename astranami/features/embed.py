"""Dense encoder with a content-addressed, append-only on-disk cache (re-embeds only changed texts)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ENCODERS
from ..util import atomic_write, log, sha, unit

QUERY_INSTRUCTION = ("Given a viewer's description of what they want to watch, retrieve anime, TV shows, "
                     "or movies whose story, themes, and tone match it")


class Encoder:
    def __init__(self, name: str = "qwen3", cache_dir: Path | None = None, batch_size: int = 32,
                 max_tokens: int = 384, device: str | None = None):
        if name not in ENCODERS:
            raise ValueError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}.")
        self.name = name
        self.model_id = ENCODERS[name]
        self.cache_dir = cache_dir / f"emb-{name}" if cache_dir else None
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = device
        self.model = None
        self._keys: dict[str, int] | None = None
        self._shards: list[np.ndarray] = []

    def _load(self):
        if self.model is not None:
            return self.model
        import torch
        from sentence_transformers import SentenceTransformer
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        log(f"Loading encoder {self.model_id} on {device}...")
        self.model = SentenceTransformer(self.model_id, device=device, model_kwargs=kwargs,
                                         trust_remote_code=self.name != "qwen3")
        self.model.max_seq_length = self.max_tokens
        return self.model

    def _raw_encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        model = self._load()
        if self.name == "jina5":
            kwargs = {"task": "retrieval", "prompt_name": "query" if query else "document"}
        else:
            kwargs = {"prompt": f"Instruct: {QUERY_INSTRUCTION}\nQuery: "} if query else {}
        vectors = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False, **kwargs)
        return np.asarray(vectors, dtype=np.float16)

    def _open_cache(self) -> None:
        if self._keys is not None:
            return
        self._keys, self._shards = {}, []
        if self.cache_dir is None or not (self.cache_dir / "keys.json").exists():
            return
        entries = json.loads((self.cache_dir / "keys.json").read_text(encoding="utf-8"))
        offset = 0
        for shard_name, shard_keys in entries:
            path = self.cache_dir / shard_name
            if not path.exists():
                break
            vectors = np.load(path)
            if len(vectors) != len(shard_keys):
                break
            self._shards.append(vectors)
            for position, key in enumerate(shard_keys):
                self._keys[key] = offset + position
            offset += len(vectors)

    def _append_shard(self, keys: list[str], vectors: np.ndarray) -> None:
        offset = sum(len(shard) for shard in self._shards)
        self._shards.append(vectors)
        for position, key in enumerate(keys):
            self._keys[key] = offset + position
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys_path = self.cache_dir / "keys.json"
        entries = json.loads(keys_path.read_text(encoding="utf-8")) if keys_path.exists() else []
        name = f"shard-{len(entries):05d}.npy"
        np.save(self.cache_dir / name, vectors)
        entries.append([name, keys])
        atomic_write(keys_path, json.dumps(entries))

    def encode_documents(self, texts: list[str], chunk: int = 4096) -> np.ndarray:
        """Encode passages, reusing cached vectors for identical texts. Returns float16 unit vectors."""
        self._open_cache()
        keys = [sha(f"{self.max_tokens}|{text}") for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self._keys))
        if missing:
            text_by_key = dict(zip(keys, texts))
            missing.sort(key=lambda key: len(text_by_key[key]))
            log(f"Encoding {len(missing):,} passages with {self.model_id} ({len(set(keys)) - len(missing):,} cached)...")
            for start in range(0, len(missing), chunk):
                batch = missing[start:start + chunk]
                self._append_shard(batch, self._raw_encode([text_by_key[key] for key in batch]))
                log(f"  encoded {min(start + chunk, len(missing)):,}/{len(missing):,}")
        matrix = np.concatenate(self._shards) if len(self._shards) > 1 else self._shards[0]
        self._shards = [matrix]
        return matrix[[self._keys[key] for key in keys]]

    def encode_query(self, text: str) -> np.ndarray:
        return unit(self._raw_encode([text], query=True).astype(np.float32))[0]
