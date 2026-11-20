"""archive-stream / restore-stream against an in-memory S3 stand-in that verifies the PUT checksum
the way S3 does (the real bucket was exercised by hand on 2026-11-20)."""

import base64
import hashlib
import io
import json
from datetime import date

import pytest
from botocore.exceptions import ClientError

from albert import archive
from albert.archive import (
    MARKER,
    STORAGE_CLASS,
    ArchivedError,
    archive_day,
    check_local,
    closed_days,
    require_day,
    restore_day,
)


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.classes: dict[str, str] = {}

    def put_object(self, *, Bucket, Key, Body, ChecksumSHA256, StorageClass="STANDARD"):
        data = Body.read()
        assert base64.b64encode(hashlib.sha256(data).digest()).decode() == ChecksumSHA256
        self.objects[Key], self.classes[Key] = data, StorageClass

    def head_object(self, *, Bucket, Key, ChecksumMode=None):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data = self.objects[Key]
        return {"ContentLength": len(data),
                "ChecksumSHA256": base64.b64encode(hashlib.sha256(data).digest()).decode()}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


def _day(root, venue="kalshi", day="2026-11-10", files=("00-k0.jsonl.zst", "01-k0.jsonl.zst")):
    d = root / venue / day
    d.mkdir(parents=True)
    for i, n in enumerate(files):
        (d / n).write_bytes(f"{n} payload {i}".encode() * 50)
    (d / "02-k0.jsonl.zst.part").write_bytes(b"still being written")
    return d


@pytest.fixture(autouse=True)
def _clear_cache():
    archive._incomplete.cache_clear()


def test_closed_days_keeps_open_days(tmp_path):
    for day in ("2026-11-10", "2026-11-16", "2026-11-17"):
        _day(tmp_path, day=day)
    (tmp_path / "pm" / "2026-11-01").mkdir(parents=True)          # empty day: nothing to archive
    got = [(v, d.name) for v, d in closed_days(tmp_path, 4, today=date(2026, 11, 20))]
    assert got == [("kalshi", "2026-11-10"), ("kalshi", "2026-11-16")]


def test_archive_then_restore_round_trip(tmp_path):
    s3, d = FakeS3(), _day(tmp_path)
    before = {p.name: p.read_bytes() for p in d.glob("*.jsonl.zst")}
    r = archive_day(s3, "b", "kalshi", d, delete_local=True)
    assert (r.uploaded, r.already, r.deleted, r.errors) == (2, 0, 2, [])
    assert set(s3.objects) == {"raw/stream/kalshi/2026-11-10/00-k0.jsonl.zst",
                               "raw/stream/kalshi/2026-11-10/01-k0.jsonl.zst"}
    assert set(s3.classes.values()) == {STORAGE_CLASS}
    assert not list(d.glob("*.jsonl.zst"))
    assert (d / "02-k0.jsonl.zst.part").exists()                  # partial files are never touched
    marker = json.loads((d / MARKER).read_text())
    assert [f["name"] for f in marker["files"]] == sorted(before)

    with pytest.raises(ArchivedError, match="restore-stream 2026-11-10 --venue kalshi"):
        check_local(tmp_path / "kalshi", "2026-11-10T05", "2026-11-10T06")
    check_local(tmp_path / "kalshi", "2026-11-11T00", "2026-11-12T00")   # other days are fine
    with pytest.raises(ArchivedError):
        require_day(d)

    assert restore_day(s3, d) == {"restored": 2, "present": 0}
    assert {p.name: p.read_bytes() for p in d.glob("*.jsonl.zst")} == before
    check_local(tmp_path / "kalshi")
    archive._incomplete.cache_clear()
    require_day(d)
    assert restore_day(s3, d) == {"restored": 0, "present": 2}


def test_rerun_is_idempotent_and_never_overwrites(tmp_path):
    s3, d = FakeS3(), _day(tmp_path)
    archive_day(s3, "b", "kalshi", d, delete_local=False)
    r = archive_day(s3, "b", "kalshi", d, delete_local=False)
    assert (r.uploaded, r.already, r.errors) == (0, 2, [])

    key = "raw/stream/kalshi/2026-11-10/00-k0.jsonl.zst"
    s3.objects[key] = b"different bytes"                          # remote copy differs from disk
    r = archive_day(s3, "b", "kalshi", d, delete_local=True)
    assert len(r.errors) == 1 and r.deleted == 0                  # one bad file keeps the whole day
    assert s3.objects[key] == b"different bytes"
    assert len(list(d.glob("*.jsonl.zst"))) == 2


def test_restore_refuses_a_corrupt_download(tmp_path):
    s3, d = FakeS3(), _day(tmp_path)
    archive_day(s3, "b", "kalshi", d, delete_local=True)
    s3.objects["raw/stream/kalshi/2026-11-10/01-k0.jsonl.zst"] = b"tampered"
    with pytest.raises(ValueError, match="do not match"):
        restore_day(s3, d)
    assert not (d / "01-k0.jsonl.zst").exists()
    assert not list(d.glob("01-k0.jsonl.zst.part"))
