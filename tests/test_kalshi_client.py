import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from albert.cache import raw_cache_path
from albert.kalshi.client import FetchedPage, KalshiClient
from albert.kalshi.client import ALLOWED_ENDPOINT_SEGMENTS


UTC = timezone.utc
BASE_URL = "https://kalshi.test/trade-api/v2"


def _dt(hour: int) -> datetime:
    return datetime(2026, 4, 1, tzinfo=UTC) + timedelta(hours=hour)


def test_allowed_endpoint_segments_are_phase_two_contract() -> None:
    assert ALLOWED_ENDPOINT_SEGMENTS == {
        "live_markets",
        "historical_markets",
        "live_trades",
        "historical_trades",
        "live_candlesticks",
        "historical_candlesticks",
        "live_orderbooks",
        "historical_cutoff",
    }


def test_get_json_page_writes_raw_cache_with_safe_endpoint_segment(
    cache_root: Path,
) -> None:
    payload = {"markets": [{"ticker": "KXBTC15M-26APR011200"}], "cursor": None}
    params = {
        "series_ticker": "KXBTC15M",
        "min_close_ts": "2026-04-01T00:00:00Z",
        "max_close_ts": "2026-04-02T00:00:00Z",
    }
    fetched_at = _dt(12)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets"
        assert dict(request.url.params) == params
        return httpx.Response(200, json=payload)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = KalshiClient(
        cache_root=cache_root,
        base_url=BASE_URL,
        http_client=http,
        now=lambda: fetched_at,
    )

    page = client.get_json_page(
        endpoint_segment="live_markets",
        path="/markets",
        parameters=params,
        covered_start=_dt(0),
        covered_end=_dt(24),
    )

    expected_raw_path = raw_cache_path(cache_root, "kalshi", "live_markets", params)
    assert isinstance(page, FetchedPage)
    assert page.source == "kalshi"
    assert page.endpoint == "/markets"
    assert page.endpoint_segment == "live_markets"
    assert page.parameters == params
    assert page.fetched_at == fetched_at
    assert page.covered_start == _dt(0)
    assert page.covered_end == _dt(24)
    assert page.raw_path == expected_raw_path
    assert page.payload == payload
    assert json.loads(expected_raw_path.read_text(encoding="utf-8")) == payload


def test_iter_cursor_pages_keeps_filters_and_adds_cursor(cache_root: Path) -> None:
    observed_queries: list[dict[str, str]] = []
    params = {
        "series_ticker": "KXBTC15M",
        "min_close_ts": "2026-04-01T00:00:00Z",
        "max_close_ts": "2026-04-02T00:00:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        query = dict(request.url.params)
        observed_queries.append(query)
        if "cursor" not in query:
            return httpx.Response(
                200,
                json={
                    "markets": [{"ticker": "KXBTC15M-26APR011200"}],
                    "cursor": "page-2",
                },
            )
        return httpx.Response(
            200,
            json={
                "markets": [{"ticker": "KXBTC15M-26APR011215"}],
                "cursor": None,
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = KalshiClient(cache_root=cache_root, base_url=BASE_URL, http_client=http)

    pages = list(
        client.iter_cursor_pages(
            endpoint_segment="live_markets",
            path="/markets",
            parameters=params,
            collection_key="markets",
            covered_start=_dt(0),
            covered_end=_dt(24),
        )
    )

    assert len(pages) == 2
    assert observed_queries[0] == params
    assert observed_queries[1] == {**params, "cursor": "page-2"}
    assert params == {
        "series_ticker": "KXBTC15M",
        "min_close_ts": "2026-04-01T00:00:00Z",
        "max_close_ts": "2026-04-02T00:00:00Z",
    }


def test_retries_429_then_succeeds_with_bounded_attempts(cache_root: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"markets": [], "cursor": None})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = KalshiClient(
        cache_root=cache_root,
        base_url=BASE_URL,
        http_client=http,
        max_retries=3,
    )

    page = client.get_json_page(
        endpoint_segment="live_markets",
        path="/markets",
        parameters={"series_ticker": "KXBTC15M"},
        covered_start=_dt(0),
        covered_end=_dt(1),
    )

    assert attempts == 3
    assert page.payload == {"markets": [], "cursor": None}


@pytest.mark.parametrize(
    "unsafe_endpoint",
    ["../live_markets", "live/markets", "live markets", "markets.."],
)
def test_rejects_unsafe_endpoint_segment(
    cache_root: Path,
    unsafe_endpoint: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("unsafe endpoint segments must be rejected before HTTP")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = KalshiClient(cache_root=cache_root, base_url=BASE_URL, http_client=http)

    with pytest.raises(ValueError, match="endpoint"):
        client.get_json_page(
            endpoint_segment=unsafe_endpoint,
            path="/markets",
            parameters={"series_ticker": "KXBTC15M"},
            covered_start=_dt(0),
            covered_end=_dt(1),
        )
