import httpx

from albert.http import CachedClient


def make(tmp_path, handler):
    return CachedClient("v", "https://example.test", tmp_path, min_interval=0,
                        transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_cache_key_includes_path():
    a = CachedClient.cache_key("https://x/historical/cutoff", {})
    b = CachedClient.cache_key("https://x/markets", {})
    assert a != b


def test_cached_requests_are_reused_and_live_ones_append(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"n": len(calls)})

    c = make(tmp_path, handler)
    first = c.get("/a", {"x": 1})
    again = c.get("/a", {"x": 1})
    assert again.from_cache and again.payload == first.payload and len(calls) == 1
    live1 = c.get("/b", cache=False)
    live2 = c.get("/b", cache=False)
    assert live1.raw_path != live2.raw_path and live1.raw_path.exists() and live2.raw_path.exists()


def test_retries_429_then_succeeds(tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"ok": True})

    assert make(tmp_path, handler).get("/r").payload == {"ok": True}
    assert state["n"] == 3
