import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from albert.cache import raw_cache_path, sanitize_segment


ALLOWED_ENDPOINT_SEGMENTS = {
    "live_markets",
    "historical_markets",
    "live_trades",
    "historical_trades",
    "live_candlesticks",
    "historical_candlesticks",
    "live_orderbooks",
    "historical_cutoff",
}

RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class FetchedPage:
    source: str
    endpoint: str
    endpoint_segment: str
    parameters: dict[str, object]
    fetched_at: datetime
    covered_start: datetime
    covered_end: datetime
    raw_path: Path
    payload: dict[str, object]


class KalshiClient:
    def __init__(
        self,
        cache_root: Path,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        timeout: float = 10.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self._cache_root = cache_root
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._http = http_client or httpx.Client(timeout=timeout)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def get_json_page(
        self,
        *,
        endpoint_segment: str,
        path: str,
        parameters: Mapping[str, object],
        covered_start: datetime,
        covered_end: datetime,
    ) -> FetchedPage:
        endpoint_segment = self._validated_endpoint_segment(endpoint_segment)
        request_parameters = dict(parameters)
        payload = self._request_json(path, request_parameters)
        fetched_at = self._aware_utc(self._now(), "fetched_at")
        raw_path = raw_cache_path(
            self._cache_root,
            "kalshi",
            endpoint_segment,
            request_parameters,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return FetchedPage(
            source="kalshi",
            endpoint=path,
            endpoint_segment=endpoint_segment,
            parameters=request_parameters,
            fetched_at=fetched_at,
            covered_start=self._aware_utc(covered_start, "covered_start"),
            covered_end=self._aware_utc(covered_end, "covered_end"),
            raw_path=raw_path,
            payload=payload,
        )

    def iter_cursor_pages(
        self,
        *,
        endpoint_segment: str,
        path: str,
        parameters: Mapping[str, object],
        collection_key: str,
        covered_start: datetime,
        covered_end: datetime,
    ) -> Iterator[FetchedPage]:
        cursor: object | None = None
        while True:
            page_parameters = dict(parameters)
            if cursor:
                page_parameters["cursor"] = cursor
            page = self.get_json_page(
                endpoint_segment=endpoint_segment,
                path=path,
                parameters=page_parameters,
                covered_start=covered_start,
                covered_end=covered_end,
            )
            if collection_key not in page.payload:
                raise ValueError(f"payload missing collection key: {collection_key}")
            yield page
            cursor = page.payload.get("cursor")
            if cursor is None or cursor == "":
                break

    def _request_json(
        self,
        path: str,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        last_error: Exception | None = None
        url = self._url_for(path)
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._http.get(url, params=parameters)
                if response.status_code in RETRY_STATUSES and attempt < self._max_retries:
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Kalshi response JSON must be an object")
                return payload
            except httpx.TransportError as error:
                last_error = error
                if attempt == self._max_retries:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Kalshi request failed without a response")

    def _url_for(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("endpoint path must start with /")
        return f"{self._base_url}{path}"

    @staticmethod
    def _validated_endpoint_segment(endpoint_segment: str) -> str:
        try:
            safe_segment = sanitize_segment(endpoint_segment)
        except ValueError as error:
            raise ValueError(
                f"unsafe Kalshi endpoint segment: {endpoint_segment!r}"
            ) from error
        if safe_segment not in ALLOWED_ENDPOINT_SEGMENTS:
            raise ValueError(f"unsupported Kalshi endpoint segment: {endpoint_segment}")
        return safe_segment

    @staticmethod
    def _aware_utc(value: datetime, field_name: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value.astimezone(timezone.utc)


__all__ = [
    "ALLOWED_ENDPOINT_SEGMENTS",
    "FetchedPage",
    "KalshiClient",
]
