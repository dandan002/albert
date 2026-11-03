import json
import os

import httpx

from albert.http import CachedClient, compact_raw, read_raw


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


def test_raw_files_are_compressed_and_legacy_json_still_served(tmp_path):
    def handler(request):
        raise AssertionError("should be served from cache")

    c = make(tmp_path, lambda r: httpx.Response(200, json={"v": 1}))
    fresh = c.get("/z", {"q": 2})
    assert fresh.raw_path.name.endswith(".json.zst")
    assert read_raw(fresh.raw_path)["response"] == {"v": 1}
    # A legacy plain-JSON file (written before compression) is still a cache hit.
    key = CachedClient.cache_key("https://example.test/old", {})
    legacy = tmp_path / "v" / f"{key}.json"
    legacy.write_text(json.dumps({"fetched_at": "2026-11-01T00:00:00+00:00", "response": [1]}))
    assert make(tmp_path, handler).get("/old").payload == [1]


def test_compact_raw_round_trips_and_keeps_manifest_paths_readable(tmp_path):
    legacy = tmp_path / "v" / "abc.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"fetched_at": "2026-11-01T00:00:00+00:00", "response": {"a": [1] * 500}}))
    os.utime(legacy, (0, 0))
    files, before, after = compact_raw(tmp_path)
    assert files == 1 and after < before and not legacy.exists()
    assert read_raw(legacy)["response"] == {"a": [1] * 500}  # old manifest path still resolves
    fresh = tmp_path / "v" / "new.json"
    fresh.write_text("{}")
    assert compact_raw(tmp_path)[0] == 0 and fresh.exists()  # too young: skipped


def test_compact_raw_drops_identical_refetch_and_keeps_distinct_one(tmp_path):
    from albert.http import write_raw

    d = tmp_path / "v"
    write_raw(d / "k1.json.zst", {"fetched_at": "a", "response": [1]})
    write_raw(d / "k2.json.zst", {"fetched_at": "a", "response": [1]})
    (d / "k1.json").write_text(json.dumps({"fetched_at": "b", "response": [1]}))
    (d / "k2.json").write_text(json.dumps({"fetched_at": "b", "response": [2]}))
    for f in d.glob("*.json"):
        os.utime(f, (0, 0))
    assert compact_raw(tmp_path)[0] == 2
    assert read_raw(d / "k1.json.zst")["fetched_at"] == "a"
    dups = list(d.glob("k2-dup-*.json.zst"))
    assert len(dups) == 1 and read_raw(dups[0])["response"] == [2]
    assert not list(d.glob("*.json"))
