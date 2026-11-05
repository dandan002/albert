import base64
import json
from collections import defaultdict
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from albert.stream import HourlyWriter, KalshiBook, read_stream, rest_levels
from albert.venues.kalshi_auth import KalshiSigner, load_private_key

FIX = Path(__file__).parent / "fixtures"


def _kalshi_frames() -> list[dict]:
    lines = (FIX / "kalshi" / "ws_orderbook.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_recorded_frames_are_contiguous_and_rebuild_a_book():
    frames = _kalshi_frames()
    book_msgs = [f for f in frames if f["type"] in ("orderbook_snapshot", "orderbook_delta")]
    assert book_msgs[0]["type"] == "orderbook_snapshot"
    assert len(book_msgs) > 10
    assert all(isinstance(f["seq"], int) and f["sid"] == book_msgs[0]["sid"] for f in book_msgs)

    snap = book_msgs[0]["msg"]
    expected: dict[tuple[str, float], float] = defaultdict(float)
    for side in ("yes", "no"):
        for p, q in snap.get(f"{side}_dollars_fp") or []:
            expected[(side, float(p))] += float(q)
    for f in book_msgs[1:]:
        m = f["msg"]
        expected[(m["side"], float(m["price_dollars"]))] += float(m["delta_fp"])

    book = KalshiBook()
    for f in book_msgs:
        book.apply(f["type"], f["msg"])
    ticker = snap["market_ticker"]
    for side in ("yes", "no"):
        got = dict(book.levels(ticker, side, 1000))
        want = {p: round(q, 6) for (s, p), q in expected.items() if s == side and round(q, 6) > 0}
        assert got == pytest.approx(want)
        assert all(q > 0 for q in got.values())


def test_rest_levels_matches_orderbook_fixture():
    payload = json.loads((FIX / "kalshi" / "orderbook.json").read_text())
    yes = rest_levels(payload, "yes", 3)
    assert len(yes) == 3 and yes == sorted(yes, reverse=True)


def test_hourly_writer_round_trip_and_never_overwrites(tmp_path):
    ns = 1_793_848_273_000_000_000  # 2026-11-05 03:11 UTC
    w = HourlyWriter(tmp_path, "kalshi", "k0")
    w.write(ns, '{"a":1}')
    w.write(ns + 3_600 * 10**9, '{"b":2}')  # next hour -> new file
    w.close()
    assert [c.lines for c in w.closed] == [1, 1]
    first = Path(w.closed[0].path)
    assert first.name == "03-k0.jsonl.zst"
    assert read_stream(first) == [{"recv_ns": ns, "conn": "k0", "frame": '{"a":1}'}]

    w2 = HourlyWriter(tmp_path, "kalshi", "k0")  # a restart in the same hour
    w2.write(ns, '{"c":3}')
    w2.close()
    assert Path(w2.closed[0].path).name == "03-k0-1.jsonl.zst"
    assert read_stream(first)[0]["frame"] == '{"a":1}'


def test_signer_accepts_bare_base64_key_and_signs_pss():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.TraditionalOpenSSL,
                            serialization.NoEncryption())
    loaded = load_private_key(base64.b64encode(der).decode())
    h = KalshiSigner("key-id", loaded).headers("get", "/trade-api/ws/v2?x=1", now_ms=1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "key-id" and h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), b"1700000000000GET/trade-api/ws/v2",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())
