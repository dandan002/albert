"""HTTP client with per-host rate limiting, retries, and an append-only raw cache.

Every response is stored under ``data/raw/<venue>/<key>.json`` along with the request that
produced it. The key is a hash of the full URL (path + sorted params), so different endpoints
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

RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class Fetched:
    url: str
    params: dict[str, Any]
    fetched_at: datetime
    raw_path: Path
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
    ) -> None:
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
            headers={"User-Agent": "albert-research/0.2"},
        )

    def get(self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True,
            tag: str | None = None) -> Fetched:
        """GET ``path``. ``tag`` is mixed into the cache key only (e.g. a date for daily refresh)."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{self.base_url}{path}"
        key = self.cache_key(url, params if tag is None else {**params, "__tag": tag})
        if cache:
            cached = self.raw_dir / f"{key}.json"
            if cached.exists():
                record = json.loads(cached.read_text())
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
        name = key if cache else f"{key}-{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        raw_path = self.raw_dir / f"{name}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "venue": self.venue,
            "url": url,
            "params": params,
            "fetched_at": fetched_at.isoformat(),
            "response": payload,
        }
        tmp = raw_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, separators=(",", ":")))
        tmp.replace(raw_path)
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
            return response.json()
        raise HttpError(f"{url}: retries exhausted")

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()
