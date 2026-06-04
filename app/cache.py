"""
Tiny persistent cache for AI responses.

AI calls go through a real browser and take many seconds (~30-40s for an
auto-fill). Many are effectively deterministic for a given input — the
title/composer for a filename never changes — so they are cached to disk and
returned instantly on a repeat.

Usage:
    from app.cache import ai_cache
    hit = ai_cache.get("suggest_meta", filename)
    if hit is None:
        hit = ...call AI...
        ai_cache.set("suggest_meta", filename, hit)

Keyed by (namespace, key); values are any JSON-serialisable object. Backed by
a single JSON file, loaded once and rewritten on each set (the volume is tiny).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

_CACHE_PATH = Path(os.environ.get(
    "AI_CACHE_PATH",
    str(Path(__file__).parent.parent / ".cache" / "ai_cache.json"),
))


class _AICache:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _flush(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def _norm(key: str) -> str:
        return str(key).strip().lower()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(namespace, {}).get(self._norm(key))

    def set(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(namespace, {})[self._norm(key)] = value
            self._flush()

    def delete(self, namespace: str, key: str) -> bool:
        with self._lock:
            ns = self._data.get(namespace, {})
            if self._norm(key) in ns:
                del ns[self._norm(key)]
                self._flush()
                return True
            return False

    def stats(self) -> dict:
        with self._lock:
            return {ns: len(items) for ns, items in self._data.items()}


ai_cache = _AICache(_CACHE_PATH)
