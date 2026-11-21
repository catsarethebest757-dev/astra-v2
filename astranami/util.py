from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .config import USER_AGENT

STOP = set(
    "a an and are as at be been but by for from had has have he her his how in is it its not of on or she "
    "that the their them they this to was were what when which who will with you your into after before "
    "while where there then than also only just about over under more most very can one two".split()
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def relative_path(value) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a path relative to the current directory, without '..'.")
    return path


def clean(value) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    text = re.sub(r"\(Source:[^)]*\)|\[Written by[^\]]*\]|Source:\s*\S+\s*$", " ", text, flags=re.I)
    return " ".join(text.split())


def norm(value) -> str:
    text = unicodedata.normalize("NFKD", clean(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def tokens(value) -> list[str]:
    words = [word for word in norm(value).split() if word not in STOP and len(word) > 1]
    return words + [left + "~" + right for left, right in zip(words, words[1:])]


def number(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def sha(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class RateLimiter:
    """Minimum spacing between calls, adjustable when a server reports its own limit."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def http(url: str, *, data: bytes | None = None, headers: dict | None = None, limiter: RateLimiter | None = None,
         retries: int = 6, timeout: int = 90, max_bytes: int = 200 * 1024 * 1024) -> tuple[bytes, dict]:
    """HTTP GET/POST with retry, Retry-After support, and a size cap. Raises HTTPError for non-retryable codes."""
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
    request_headers.update(headers or {})
    context = ssl_context()
    last_error: Exception | None = None
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        request = urllib.request.Request(url, data=data, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                chunks, total = [], 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Response from {url} exceeds {max_bytes} bytes.")
                    chunks.append(chunk)
                return b"".join(chunks), dict(response.headers.items())
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 404 or error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            retry_after = number(error.headers.get("Retry-After"), 0.0) if error.headers else 0.0
            time.sleep(max(retry_after, 2.0 ** attempt) + 0.5)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error
            time.sleep(2.0 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def download(url: str, path: Path, max_bytes: int = 200 * 1024 * 1024) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    body, _ = http(url, max_bytes=max_bytes)
    atomic_write(path, body)
    return body
