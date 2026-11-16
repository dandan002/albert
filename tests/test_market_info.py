"""leadlag_fast.market_info: retries rate limits and never silently drops a ticker (a dropped
ticker silently shrank the in-play HF sample). Payload: a recorded GET /markets/{ticker}."""

import json
from pathlib import Path

import httpx
import pytest

from albert.research.leadlag_fast import market_info

PAYLOAD = json.loads((Path(__file__).parent / "fixtures" / "kalshi" / "market_single.json").read_text())
TICKER = PAYLOAD["market"]["ticker"]


def _transport(statuses: list[int], seen: list[str]):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        code = statuses.pop(0) if statuses else 200
        return httpx.Response(code, json=PAYLOAD if code == 200 else {"error": "too many requests"})
    return httpx.MockTransport(handler)


def test_rate_limited_ticker_is_retried_and_cached(tmp_path):
    seen: list[str] = []
    cache = tmp_path / "meta.json"
    got = market_info([TICKER], cache, transport=_transport([429, 429, 503], seen), sleep=lambda s: None)
    assert len(seen) == 4 and seen[-1].endswith(TICKER)
    assert got[TICKER]["expected_expiration_time"] == PAYLOAD["market"]["expected_expiration_time"]
    assert got[TICKER]["result"] == PAYLOAD["market"]["result"]
    assert json.loads(cache.read_text()) == got
    # cached: no request on the next call
    market_info([TICKER], cache, transport=_transport([], seen), sleep=lambda s: None)
    assert len(seen) == 4


def test_persistent_failure_raises_after_saving_progress(tmp_path):
    cache = tmp_path / "meta.json"
    other = TICKER.rsplit("-", 1)[0] + "-ZZZ"
    statuses = [200] + [429] * 10          # first ticker succeeds, the second never does
    with pytest.raises(RuntimeError, match="missing for 1 of 2"):
        market_info(sorted([TICKER, other]), cache, retries=3, transport=_transport(statuses, []),
                    sleep=lambda s: None)
    assert list(json.loads(cache.read_text())) == [min(TICKER, other)]
