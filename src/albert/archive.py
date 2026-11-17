"""Move closed raw stream days to S3 and bring them back.

``data/raw/stream/<venue>/<date>/`` is append-only, so a day is archived only once no recorder can
still write into it (``date <= today - keep_days``, UTC). Each file goes up with a single PUT that
carries its SHA-256, which S3 verifies on receipt; a HEAD then re-checks size and checksum. Only
after every file of the day has passed is the local copy deleted, and the day's directory keeps an
``_archived.json`` marker listing each file (name, size, sha256) and its key. The bucket policy
grants no delete, and an existing key with different content is never overwritten.

Readers call :func:`check_local` so a backtest over an archived day fails loudly instead of
silently running on fewer files; ``albert restore-stream <date>`` brings a day back byte-identical.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

MARKER = "_archived.json"
VENUES = ("kalshi", "pm")
PREFIX = "raw/stream"


class ArchivedError(FileNotFoundError):
    """A requested stream day is in S3, not on disk."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def key_for(venue: str, day: str, name: str) -> str:
    return f"{PREFIX}/{venue}/{day}/{name}"


def closed_days(stream_root: Path, keep_days: int, today: date | None = None) -> list[tuple[str, Path]]:
    """(venue, day dir) for every day old enough to archive that still has local stream files."""
    today = today or datetime.now(UTC).date()
    last = (today - timedelta(days=keep_days)).isoformat()
    out = []
    for venue in VENUES:
        for d in sorted((stream_root / venue).glob("????-??-??")):
            if d.is_dir() and d.name <= last and any(d.glob("*.jsonl.zst")):
                out.append((venue, d))
    return out


def _read_marker(day_dir: Path) -> dict[str, Any] | None:
    p = day_dir / MARKER
    return json.loads(p.read_text()) if p.exists() else None


def _remote(client, bucket: str, key: str) -> dict[str, Any] | None:
    from botocore.exceptions import ClientError

    try:
        return client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


@dataclass
class DayResult:
    venue: str
    day: str
    files: list[dict[str, Any]] = field(default_factory=list)
    uploaded: int = 0
    already: int = 0
    bytes: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)


def archive_day(client, bucket: str, venue: str, day_dir: Path, *, delete_local: bool,
                workers: int = 8) -> DayResult:
    """Upload every stream file of one closed day, verify each, then (optionally) delete locally."""
    res = DayResult(venue, day_dir.name)
    paths = sorted(p for p in day_dir.iterdir() if p.is_file() and p.name != MARKER and not p.name.endswith(".part"))

    def one(p: Path) -> dict[str, Any]:
        size, digest = p.stat().st_size, sha256_file(p)
        key = key_for(venue, day_dir.name, p.name)
        head = _remote(client, bucket, key)
        if head is None:
            with open(p, "rb") as f:
                client.put_object(Bucket=bucket, Key=key, Body=f, ChecksumSHA256=digest)
            head, status = _remote(client, bucket, key), "uploaded"
        else:
            status = "already"
        ok = head is not None and head.get("ContentLength") == size and head.get("ChecksumSHA256") == digest
        return {"name": p.name, "size": size, "sha256": digest, "key": key, "status": status, "ok": ok}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, paths))
    marker = _read_marker(day_dir) or {"venue": venue, "day": day_dir.name, "bucket": bucket, "files": []}
    known = {r["name"]: r for r in marker["files"]}
    for r in rows:
        if not r["ok"]:
            res.errors.append(f"{r['key']}: remote size/checksum differs from local; kept local, not overwritten")
            continue
        res.uploaded += r["status"] == "uploaded"
        res.already += r["status"] == "already"
        res.bytes += r["size"]
        known[r["name"]] = {k: r[k] for k in ("name", "size", "sha256", "key")}
        res.files.append(known[r["name"]])
    marker["files"] = sorted(known.values(), key=lambda r: r["name"])
    marker["bucket"] = bucket
    marker["archived_at"] = datetime.now(UTC).isoformat()
    tmp = day_dir / (MARKER + ".part")
    tmp.write_text(json.dumps(marker, indent=1))
    os.replace(tmp, day_dir / MARKER)          # the marker lands before any local file goes
    if delete_local and not res.errors:
        for r in res.files:
            (day_dir / r["name"]).unlink()
            res.deleted += 1
    return res


def restore_day(client, day_dir: Path, *, workers: int = 8) -> dict[str, int]:
    """Download every file listed in a day's marker that is missing locally, verified by SHA-256.
    An existing local file is never replaced; it must match the marker."""
    marker = _read_marker(day_dir)
    if marker is None:
        raise FileNotFoundError(f"{day_dir} has no {MARKER}: nothing archived for that day")
    bucket = marker["bucket"]

    def one(r: dict[str, Any]) -> str:
        dest = day_dir / r["name"]
        if dest.exists():
            if dest.stat().st_size != r["size"] or sha256_file(dest) != r["sha256"]:
                raise ValueError(f"{dest} exists and differs from the archived copy")
            return "present"
        part = dest.with_name(dest.name + ".part")
        h = hashlib.sha256()
        body = client.get_object(Bucket=bucket, Key=r["key"])["Body"]
        with open(part, "wb") as f:
            for chunk in iter(lambda: body.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
        if base64.b64encode(h.digest()).decode() != r["sha256"] or part.stat().st_size != r["size"]:
            part.unlink()
            raise ValueError(f"{r['key']}: downloaded bytes do not match the marker")
        os.replace(part, dest)
        return "restored"

    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(one, marker["files"]))
    return {"restored": got.count("restored"), "present": got.count("present")}


def check_local(venue_root: Path, start: str | None = None, end: str | None = None) -> None:
    """Raise :class:`ArchivedError` if a day overlapping ``[start, end)`` (``YYYY-MM-DDTHH``) has
    been archived and is not fully restored, so readers never run silently on missing hours."""
    missing = []
    for marker in sorted(venue_root.glob(f"*/{MARKER}")):
        d = marker.parent.name
        if (start is not None and f"{d}T23" < start) or (end is not None and f"{d}T00" >= end):
            continue
        names = [r["name"] for r in json.loads(marker.read_text())["files"]]
        if any(not (marker.parent / n).exists() for n in names):
            missing.append(d)
    if missing:
        raise ArchivedError(f"{venue_root.name} stream days {', '.join(missing)} are archived in S3; "
                            f"run `albert restore-stream {missing[0]}` (one per day) first")
