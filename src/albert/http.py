"""HTTP client with per-host rate limiting, retries, and an append-only raw cache.

Every response is stored under ``data/raw/<venue>/<key>.json.zst`` (zstd-compressed JSON,
about 20x smaller than plain JSON) along with the request that produced it. Legacy plain
``<key>.json`` files are still read, and ``compact_raw`` converts them losslessly. The key is a hash of the full URL (path + sorted params), so different endpoints
never collide. Immutable requests (``cache=True``) are served from disk on re-runs. Live
requests (``cache=False``) always hit the network and are written to a new timestamped file,
so a raw file referenced by an earlier run is never overwritten.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import zstandard

RETRY_STATUSES = {429, 500, 502, 503, 504}
RAW_SUFFIX = ".json.zst"
_ZSTD_LEVEL = 10


def write_raw(path: Path, record: dict[str, Any]) -> None:
    """Atomically write one raw record as zstd-compressed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, separators=(",", ":")).encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data))
    tmp.replace(path)


def read_raw(path: Path) -> dict[str, Any]:
    """Read a raw record, compressed (``.json.zst``) or legacy plain ``.json``.

    Manifests written before compression list ``.json`` paths; those resolve to the
    ``.json.zst`` sibling when the plain file has been compacted.
    """
    if path.name.endswith(".json") and not path.exists():
        path = path.with_name(path.name + ".zst")
    data = path.read_bytes()
    if path.name.endswith(".zst"):
        data = zstandard.ZstdDecompressor().decompress(data)
    return json.loads(data)


def compact_raw(raw_dir: Path, *, min_age_seconds: float = 60.0) -> tuple[int, int, int]:
    """Convert legacy plain ``.json`` raw files to ``.json.zst``, verifying each round trip
    before deleting the original. Files younger than ``min_age_seconds`` are skipped so an
    ingest that is writing right now is left alone. Returns (files, bytes_before, bytes_after).

    If a ``.json.zst`` with the same key already exists (the same request was re-fetched by
    an older process that could not see the compressed cache), the plain file is dropped when
    its response is identical, and otherwise kept as ``<key>-dup-<mtime>.json.zst`` so no
    distinct response is ever lost.
    """
    files = before = after = 0
    now = time.time()
    for src in sorted(raw_dir.rglob("*.json")):
        mtime = src.stat().st_mtime
        if now - mtime < min_age_seconds:
            continue
        data = src.read_bytes()
        dst = src.with_name(src.name + ".zst")
        if dst.exists():
            if read_raw(dst).get("response") == json.loads(data).get("response"):
                before += len(data)
                src.unlink()
                files += 1
                continue
            stamp = datetime.fromtimestamp(mtime, UTC).strftime("%Y%m%dT%H%M%SZ")
            dst = src.with_name(f"{src.stem}-dup-{stamp}{RAW_SUFFIX}")
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data))
        tmp.replace(dst)
        if zstandard.ZstdDecompressor().decompress(dst.read_bytes()) != data:
            raise RuntimeError(f"round trip mismatch for {src}; original kept")
        before += len(data)
        after += dst.stat().st_size
        src.unlink()
        files += 1
    return files, before, after


@dataclass
class Fetched:
    url: str
    params: dict[str, Any]
    fetched_at: datetime
    raw_path: Path | None   # None for live responses that were not recorded
    payload: Any
    from_cache: bool


class HttpError(RuntimeError):
    pass


class CachedClient:
    def __init__(
        self,
        venue: str,
        base_url: str,
        raw_dir: Path,
        *,
        min_interval: float = 0.1,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        record_live: bool = True,
        user_agent: str = "albert-research/0.2",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.record_live = record_live
        self.venue = venue
        self.base_url = base_url.rstrip("/")
        self.raw_dir = raw_dir / venue
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_request = 0.0
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            # Extra headers (e.g. auth) are sent but never stored: raw records keep only the
            # URL and params, and the cache key ignores headers.
            headers={"User-Agent": user_agent, **(headers or {})},
        )

    def get(self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True,
            tag: str | None = None) -> Fetched:
        """GET ``path``. ``tag`` is mixed into the cache key only (e.g. a date for daily refresh)."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{self.base_url}{path}"
        key = self.cache_key(url, params if tag is None else {**params, "__tag": tag})
        if cache:
            cached = next((p for p in (self.raw_dir / f"{key}{RAW_SUFFIX}",
                                       self.raw_dir / f"{key}.json") if p.exists()), None)
            if cached is not None:
                record = read_raw(cached)
                return Fetched(
                    url=url,
                    params=params,
                    fetched_at=datetime.fromisoformat(record["fetched_at"]),
                    raw_path=cached,
                    payload=record["response"],
                    from_cache=True,
                )
        payload = self._request(url, params)
        fetched_at = datetime.now(UTC)
        if not cache and not self.record_live:
            return Fetched(url, params, fetched_at, None, payload, from_cache=False)
        name = key if cache else f"{key}-{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        raw_path = self.raw_dir / f"{name}{RAW_SUFFIX}"
        record = {
            "venue": self.venue,
            "url": url,
            "params": params,
            "fetched_at": fetched_at.isoformat(),
            "response": payload,
        }
        write_raw(raw_path, record)
        return Fetched(url, params, fetched_at, raw_path, payload, from_cache=False)

    @staticmethod
    def cache_key(url: str, params: dict[str, Any]) -> str:
        canonical = f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def _request(self, url: str, params: dict[str, Any]) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as error:
                if attempt == self.max_retries:
                    raise HttpError(f"{url}: {error}") from error
                self._sleep(delay)
                delay *= 2
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                self._sleep(wait)
                delay *= 2
                continue
            if response.status_code >= 400:
                raise HttpError(f"{url} {params} -> {response.status_code}: {response.text[:300]}")
            # JSON APIs return parsed JSON; CSV/text endpoints (e.g. IEM) return the raw text.
            if "json" in response.headers.get("content-type", "") or response.text.lstrip()[:1] in ("[", "{"):
                return response.json()
            return response.text
        raise HttpError(f"{url}: retries exhausted")

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()
