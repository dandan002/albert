"""Cross-venue market matching.

Only *exact* matches (same event, same outcome, same resolution rule) are used for trading. Fuzzy
candidates are written separately for manual review and are never traded.

Observed conventions (2026-11):
* Kalshi game markets look like ``KXNBAGAME-26OCT30LALGSW-LAL``: series, US-local date, both team
  codes concatenated, and the team the YES side backs.
* Polymarket intl moneylines look like ``nba-lal-gsw-2026-10-30``: same codes, same local date.
  The outcomes are listed in slug order. Spreads and totals carry a suffix after the date.
* Polymarket US events reuse the intl event slug, sometimes behind a product prefix
  (``aec-nba-lal-gsw-2026-10-30``). The long side is a team name or "Yes".
"""

import re
from datetime import datetime

import polars as pl

# Kalshi game series -> Polymarket slug prefix (two-way sports only; soccer has draws).
KALSHI_GAME_LEAGUES = {
    "KXNBAGAME": "nba",
    "KXNFLGAME": "nfl",
    "KXNHLGAME": "nhl",
    "KXMLBGAME": "mlb",
    "KXNCAAFGAME": "cfb",
    "KXNCAAMBGAME": "cbb",
    "KXWNBAGAME": "wnba",
}

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
KALSHI_GAME_RE = re.compile(r"^(?P<series>KX[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
                            r"(?P<teams>[A-Z0-9]+)-(?P<pick>[A-Z0-9]+)$")
PM_GAME_RE = re.compile(r"^(?P<league>[a-z0-9]+)-(?P<a>[a-z0-9]+)-(?P<b>[a-z0-9]+)-"
                        r"(?P<date>\d{4}-\d{2}-\d{2})$")

MATCH_SCHEMA = {
    "kalshi_ticker": pl.String,
    "pm_condition_id": pl.String,
    "pm_outcome_index": pl.Int64,
    "pm_event_slug": pl.String,
    "method": pl.String,
}

PMUS_MATCH_SCHEMA = {
    "pmus_slug": pl.String,
    "pm_condition_id": pl.String,
    "pm_outcome_index": pl.Int64,
    "pm_event_slug": pl.String,
    "method": pl.String,
}


def parse_kalshi_game(ticker: str) -> dict | None:
    m = KALSHI_GAME_RE.match(ticker)
    if not m or m["series"] not in KALSHI_GAME_LEAGUES or m["mon"] not in MONTHS:
        return None
    date = f"20{m['yy']}-{MONTHS[m['mon']]:02d}-{m['dd']}"
    return {"league": KALSHI_GAME_LEAGUES[m["series"]], "date": date,
            "teams": m["teams"].lower(), "pick": m["pick"].lower()}


def parse_pm_game(slug: str) -> dict | None:
    m = PM_GAME_RE.match(slug)
    if not m:
        return None
    return {"league": m["league"], "a": m["a"], "b": m["b"], "date": m["date"]}


def match_kalshi_pm(kalshi: pl.DataFrame, pm: pl.DataFrame) -> pl.DataFrame:
    """Exact moneyline matches between Kalshi game markets and Polymarket intl moneylines."""
    index: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
    for row in pm.select("slug", "condition_id", "event_slug").iter_rows(named=True):
        g = parse_pm_game(row["slug"] or "")
        if g and row["slug"] == row["event_slug"]:
            index[(g["league"], g["date"], g["a"] + g["b"])] = (
                row["condition_id"], g["a"], g["b"], row["event_slug"])
    rows = []
    for ticker in kalshi["ticker"].to_list():
        g = parse_kalshi_game(ticker)
        if not g:
            continue
        hit = index.get((g["league"], g["date"], g["teams"]))
        if not hit:
            continue
        condition_id, a, b, event_slug = hit
        if g["pick"] not in (a, b):
            continue
        rows.append({"kalshi_ticker": ticker, "pm_condition_id": condition_id,
                     "pm_outcome_index": 0 if g["pick"] == a else 1,
                     "pm_event_slug": event_slug, "method": "exact_game"})
    return pl.DataFrame(rows, schema=MATCH_SCHEMA)


def _strip_prefix(event_slug: str, known: set[str]) -> str | None:
    if event_slug in known:
        return event_slug
    rest = event_slug.split("-", 1)[1] if "-" in event_slug else None
    return rest if rest in known else None


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def match_pmus_pm(pmus: pl.DataFrame, pm: pl.DataFrame) -> pl.DataFrame:
    """Polymarket US markets matched to the intl market and outcome with the same meaning.

    * Moneylines: the US long side (a team name) equals one of the intl moneyline's two
      outcomes. The event is found by slug (minus the US product prefix) or, because team
      codes differ between the two sites (``ny`` vs ``nyk``), by league + date + the pair of
      team names.
    * Yes/No markets: the US title is contained in the intl question within the same event.
    """
    by_event: dict[str, list[dict]] = {}
    by_game: dict[tuple[str, str, frozenset[str]], dict] = {}
    for row in pm.select("condition_id", "slug", "event_slug", "question", "outcomes").iter_rows(
            named=True):
        by_event.setdefault(row["event_slug"], []).append(row)
        g = parse_pm_game(row["slug"] or "")
        if g and row["slug"] == row["event_slug"] and len(row["outcomes"]) == 2:
            by_game[(g["league"], g["date"], frozenset(_norm(o) for o in row["outcomes"]))] = row
    known = set(by_event)
    rows = []
    for us in pmus.select("slug", "event_slug", "long_side", "short_side", "title",
                          "question").iter_rows(named=True):
        long_side, short_side = _norm(us["long_side"]), _norm(us["short_side"])
        event = _strip_prefix(us["event_slug"] or us["slug"] or "", known)
        hit = None
        if long_side not in ("yes", "no", ""):
            moneyline = [c for c in by_event.get(event, []) if c["slug"] == event] if event else []
            g = parse_pm_game(us["slug"].split("-", 1)[1] if "-" in us["slug"] else "")
            if not moneyline and g:
                game = by_game.get((g["league"], g["date"], frozenset({long_side, short_side})))
                moneyline = [game] if game else []
            for c in moneyline:
                outcomes = [_norm(o) for o in c["outcomes"]]
                if long_side in outcomes and len(outcomes) == 2:
                    hit = (c, outcomes.index(long_side),
                           "exact_moneyline" if c["slug"] == event else "game_names")
                    break
        elif event:
            title = _norm(us["title"])
            for c in by_event[event]:
                if title and title in _norm(c["question"]) and [_norm(o) for o in c["outcomes"]] == ["yes", "no"]:
                    hit = (c, 0 if long_side != "no" else 1, "event_title")
                    break
        if hit:
            c, idx, method = hit
            rows.append({"pmus_slug": us["slug"], "pm_condition_id": c["condition_id"],
                         "pm_outcome_index": idx, "pm_event_slug": c["event_slug"], "method": method})
    return pl.DataFrame(rows, schema=PMUS_MATCH_SCHEMA)


def fuzzy_candidates(kalshi: pl.DataFrame, pm: pl.DataFrame, *, min_score: float = 0.6,
                     max_days: float = 2.0) -> pl.DataFrame:
    """Title-similarity candidates for non-game markets. For manual review only, never traded."""
    def tokens(text: str | None) -> frozenset[str]:
        return frozenset(t for t in _norm(text).split() if len(t) > 2)

    pm_rows = [(r["condition_id"], tokens(r["question"]), r["end_date"], r["question"])
               for r in pm.select("condition_id", "question", "end_date").iter_rows(named=True)]
    rows = []
    for k in kalshi.select("ticker", "title", "yes_sub_title", "close_time").iter_rows(named=True):
        if parse_kalshi_game(k["ticker"]):
            continue
        kt = tokens(f"{k['title']} {k['yes_sub_title']}")
        if not kt:
            continue
        best = None
        for cid, pt, end, question in pm_rows:
            if not pt or end is None or k["close_time"] is None:
                continue
            if abs((end - k["close_time"]).total_seconds()) > max_days * 86400:
                continue
            score = len(kt & pt) / len(kt | pt)
            if score >= min_score and (best is None or score > best[0]):
                best = (score, cid, question)
        if best:
            rows.append({"kalshi_ticker": k["ticker"], "kalshi_title": k["title"],
                         "pm_condition_id": best[1], "pm_question": best[2], "score": best[0]})
    return pl.DataFrame(rows, schema={"kalshi_ticker": pl.String, "kalshi_title": pl.String,
                                      "pm_condition_id": pl.String, "pm_question": pl.String,
                                      "score": pl.Float64})


def match_all(tables_dir, *, fuzzy: bool = False) -> dict[str, int]:
    from albert.store import read_table, write_table

    pm = read_table(tables_dir, "pm_markets")
    counts = {}
    frames = []
    for name in ("kalshi_markets", "kalshi_open_markets"):
        try:
            frames.append(read_table(tables_dir, name).select("ticker", "title", "yes_sub_title",
                                                               "close_time"))
        except FileNotFoundError:
            pass
    try:
        pm = pl.concat([pm, read_table(tables_dir, "pm_open_markets")], how="diagonal_relaxed")
    except FileNotFoundError:
        pass
    pm = pm.unique("condition_id")
    if frames:
        kalshi = pl.concat(frames).unique("ticker")
        km = match_kalshi_pm(kalshi, pm)
        write_table(tables_dir, "match_kalshi_pm", km, merge=False)
        counts["match_kalshi_pm"] = km.height
        if fuzzy:
            fz = fuzzy_candidates(kalshi, pm)
            write_table(tables_dir, "match_kalshi_pm_fuzzy", fz, merge=False)
            counts["match_kalshi_pm_fuzzy"] = fz.height
    us_frames = []
    for name in ("pmus_markets", "pmus_open_markets"):
        try:
            us_frames.append(read_table(tables_dir, name))
        except FileNotFoundError:
            pass
    if us_frames:
        us = pl.concat(us_frames, how="diagonal_relaxed").unique("slug")
        um = match_pmus_pm(us, pm)
        write_table(tables_dir, "match_pmus_pm", um, merge=False)
        counts["match_pmus_pm"] = um.height
    return counts


def utc_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")
