"""Cross-event / cross-series linked contracts on Kalshi: are logically linked prices consistent?

Question (2026-11-17): markets that resolve on the same fact, or on logically linked facts, but sit
in *different events or series* (so the within-event set arbitrage of ``arb_l2`` never sees them):
do they trade at prices that violate the logical bound by more than the cost of trading both legs,
and can that be harvested (taker lock, or a resting order on the rich leg hedged by the cheap leg)?

**Links** (``build_links``) are defined only from rules text and ticker structure, never from
outcomes. A link ``A => B`` means "A resolves YES implies B resolves YES" (B may be a basket of
mutually exclusive markets, or the NO side of one market), so ``P(A) <= P(B)``. A same-fact
duplicate is two links, one each way. Families (each one checked on the rules text of both legs):

* ``spread_win``      "T wins by more than k" => "T wins" (same game; full game, 1H, F5)
* ``tt_total``        "T scores over k" => "total over m", m <= k (also 1H team => team / 1H total)
* ``period_total``    1H / 2H / F5 total over k => full-game total over m, m <= k
* ``first_any``       first (team) goalscorer / touchdown => that player 1+ goals / touchdowns
* ``prop_cross``      player stat ladders across series (hits => total bases, HR => RBI, ...)
* ``protected``       "(Protected)" prop <=> the same unprotected prop (differ only on early exits)
* ``finish_ladder``   golf / racing: win => top 3/5 => top 10 => top 20 => makes the cut
* ``exact_win``       exact set score / method of victory / correct score => that side wins
* ``chart_dup``       Billboard "#1"/"#2" in two chart series (identical rules)
* ``mention_say``     a word in a Trump speech (``KXTRUMPMENTION*``) => "Trump says X before T"
* ``say_nested``      "says X in window W1" => "says X in window W2" when W1 is inside W2
* ``mention_dup``     the same speech and word listed in two mention series
* ``rain_weekend``    rain on day d (``KXRAIN``) => rain on the weekend containing d
* ``wx_hilo``         daily low >= lo => NOT daily high <= hi (hi < lo), same station and day

**Quotes**: Kalshi hourly candles (bid/ask *close* = the book at the period end, so two legs at the
same ``end_ts`` are simultaneous snapshots), cached ones plus a seeded fetch of the missing legs
(``fetch``), carried forward over the hours the API omits (no change), dropped at/after each
market's close. Gap per contract for ``A => B``: ``bid_A - cost(B)`` (sell A = buy A-NO at
``1 - bid_A``; buy B at its ask); net of Kalshi taker fees on both legs (0.07 P(1-P), rounded up to
the cent per order).

**Strategies** (one position per link, capped $100 book, both legs' capital counted):

* ``taker_snap``: take both legs at the snapshot where the net gap is >= 0 (zero latency: an upper
  bound);
* ``taker_persist``: take at the *next* hourly snapshot, only if the gap is still there (the honest
  version with hourly data);
* ``taker_minute1``: the hourly gaps re-checked on 1-minute candles; take both legs one minute after
  the snapshot, at that minute's quotes, only if the net gap is still there (the honest taker);
* ``maker``: rest a sell on the rich leg A at ``L = cost(B) + fees + margin`` for one hour; filled
  only if a print in that hour trades strictly above L (candle high); then buy B at the hour's
  closing ask (legging risk included). Maker fees per series.

Every position is held to settlement and paid on the actual ``settlement_value`` of each leg, so a
link that did not resolve consistently (basis risk) shows up as a loss. ``settle_check`` reports,
for every link in the universe, how often the implication failed at settlement.

Dev = decision before 2026-10-01, holdout = October; t by game/event cluster. Sharpe / vol from
``albert.research.vol_metrics`` on the capped-book ledgers (marks from the same candles).

    python -m albert.research.linked links | plan | fetch | settle | panel | gaps | sim | minutes | protbasis | report | all
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

OUT = Path("data/research/linked")
TABLES = Path("data/tables")
HOLDOUT_FROM = datetime(2026, 10, 1, tzinfo=UTC)
ET = "America/New_York"
MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "category", "title", "yes_sub_title", "rules_primary",
               "open_time", "close_time", "settlement_ts", "result", "settlement_value", "fee_type", "fee_multiplier",
               "expected_expiration_time"]
LINK_SCHEMA = {"link_id": pl.String, "family": pl.String, "kind": pl.String, "a": pl.String,
               "b": pl.List(pl.String), "b_side": pl.String, "cluster": pl.String, "detail": pl.String}

# ---- small parsers (pure; unit-tested) -----------------------------------------------------------
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_FULL_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"])}


def game_key(event_ticker: str) -> str:
    """``KXNFLSPREAD-26OCT18NONYG`` -> ``26OCT18NONYG`` (the part shared by every series of a game)."""
    return event_ticker.split("-", 1)[1] if "-" in event_ticker else ""


def suffix(ticker: str, event_ticker: str) -> str:
    """Market part of a ticker: ``KXNFLSPREAD-26OCT18NONYG-NO10`` -> ``NO10``."""
    return ticker[len(event_ticker) + 1:] if ticker.startswith(event_ticker + "-") else ""


def code_date(code: str) -> date | None:
    """Leading ``YYMONDD`` of an event key: ``26OCT02`` / ``26SEP051840LAAPIT`` -> 2026-10-02."""
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})", code or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def text_date(s: str) -> date | None:
    """First ``Mon[th] d, yyyy`` in a text: 'originally scheduled for Oct 2, 2026' -> 2026-10-02."""
    m = re.search(r"\b([A-Z][a-z]{2,8})\.? (\d{1,2}),? (\d{4})", s or "")
    if not m:
        return None
    k = m.group(1).lower()
    mon = _FULL_MONTHS.get(k) or next((v for n, v in _FULL_MONTHS.items() if n.startswith(k[:3])), None)
    return date(int(m.group(3)), mon, int(m.group(2))) if mon else None


def over_strike(s: str) -> float | None:
    """'Houston over 10.5 points' / 'Over 0.5 1H goals' / 'wins by more than 2.5 goals' -> 10.5 / 0.5 / 2.5."""
    m = re.search(r"(?:[Oo]ver|more than)\s+(\d+(?:\.\d+)?)", s or "")
    return float(m.group(1)) if m else None


def team_code(sfx: str) -> str:
    """Spread / team-total suffix -> team code: ``NO10`` -> ``NO``, ``LAA2`` -> ``LAA``."""
    return re.sub(r"\d+$", "", sfx)


def ladder_k(sfx: str) -> tuple[str, int] | None:
    """Player-prop suffix ``CHCABREGMAN3-2`` -> ('CHCABREGMAN3', 2)."""
    m = re.match(r"^(.+)-(\d+)$", sfx or "")
    return (m.group(1), int(m.group(2))) if m else None


def word_set(sub_title: str) -> frozenset[str]:
    """Mention word alternatives: 'Deport / Deported (5+ times)' -> {'deport', 'deported'}."""
    s = re.sub(r"\(\s*\d+\+\s*times?\s*\)", "", sub_title or "")
    return frozenset(w.strip().lower() for w in s.split("/") if w.strip())


def et_date(t: datetime) -> date:
    """US Eastern calendar date of a UTC timestamp."""
    from zoneinfo import ZoneInfo
    return t.astimezone(ZoneInfo(ET)).date()


def say_window(rules: str, open_time: datetime | None) -> tuple[datetime | None, datetime | None]:
    """UTC window of a ``KXTRUMPSAY*`` market: explicit 'after D at H ET' or the open time, to
    'before D at 12:00am ET'."""
    def et(d: date | None, hm: str = "12:00am") -> datetime | None:
        if d is None:
            return None
        m = re.match(r"(\d{1,2}):(\d{2})\s*([ap]m)", hm.lower())
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0) if m else 0
        mi = int(m.group(2)) if m else 0
        return pl.Series([datetime(d.year, d.month, d.day, h, mi)]).dt.replace_time_zone(ET).dt.convert_time_zone(
            "UTC").item()
    a = re.search(r"after (\w+ \d{1,2}, \d{4}) at ([\d:]+\s*[ap]m) ET", rules or "")
    b = re.search(r"before (\w+ \d{1,2}, \d{4}) at ([\d:]+\s*[ap]m) ET", rules or "")
    start = et(text_date(a.group(1)), a.group(2)) if a else open_time
    end = et(text_date(b.group(1)), b.group(2)) if b else None
    return start, end


def score_winner(sfx: str) -> str | None:
    """Correct-score suffix ``CFC0TOT1`` -> winning team code ``TOT`` (``TIE`` for a draw)."""
    m = re.match(r"^([A-Z]+?)(\d+)([A-Z]+)(\d+)$", sfx or "")
    if not m:
        return None
    a, ga, b, gb = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    return a if ga > gb else b if gb > ga else "TIE"


def wx_range(sub_title: str) -> tuple[float, float] | None:
    """Weather bracket in whole degrees: '78° to 79°' -> (78, 79); '83° or above' -> (83, inf);
    '68° or below' -> (-inf, 68)."""
    s = sub_title or ""
    m = re.match(r"^\s*(-?\d+)°? to (-?\d+)°?", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^\s*(-?\d+)°? or above", s)
    if m:
        return float(m.group(1)), math.inf
    m = re.match(r"^\s*(-?\d+)°? or below", s)
    if m:
        return -math.inf, float(m.group(1))
    return None


# ---- market table --------------------------------------------------------------------------------
def load_markets(series: list[str] | None = None, *, regex: str | None = None, tables: Path = TABLES) -> pl.DataFrame:
    """Settled binary markets of the given series (lazy scan + filter: the table is 1.3M rows)."""
    lf = pl.scan_parquet(tables / "kalshi_markets.parquet").filter(
        pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary"))
    if series is not None:
        lf = lf.filter(pl.col("series_ticker").is_in(series))
    if regex is not None:
        lf = lf.filter(pl.col("series_ticker").str.contains(regex))
    return with_keys(lf.select(MARKET_COLS).collect())


def with_keys(m: pl.DataFrame) -> pl.DataFrame:
    """Add ``gkey`` (event key shared across a game's series) and ``sfx`` (market part of the ticker)."""
    return m.with_columns(
        pl.col("event_ticker").map_elements(game_key, return_dtype=pl.String).alias("gkey"),
        pl.struct("ticker", "event_ticker").map_elements(lambda r: suffix(r["ticker"], r["event_ticker"]),
                                                         return_dtype=pl.String).alias("sfx"))


def _series_names(tables: Path = TABLES) -> set[str]:
    return set(pl.scan_parquet(tables / "kalshi_markets.parquet").select(pl.col("series_ticker").unique())
               .collect()["series_ticker"])


def _links(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=LINK_SCHEMA)
    df = pl.DataFrame(rows, schema=LINK_SCHEMA)
    return df.with_columns((pl.col("a") + ">" + pl.col("b").list.join("+") + ":" + pl.col("b_side")).alias("link_id")
                           ).unique("link_id", keep="first", maintain_order=True)


def _link(family: str, a: str, b: list[str], cluster: str, detail: str = "", *, kind: str = "implies",
          b_side: str = "yes") -> dict[str, Any]:
    return {"link_id": "", "family": family, "kind": kind, "a": a, "b": b, "b_side": b_side, "cluster": cluster,
            "detail": detail}


def _same_sched(ra: str, rb: str) -> bool:
    """Both rules name the same 'originally scheduled for <date>' (or neither names one)."""
    da = re.search(r"originally scheduled for (\w+\.? \d{1,2},? \d{4})", ra or "")
    db = re.search(r"originally scheduled for (\w+\.? \d{1,2},? \d{4})", rb or "")
    if not da or not db:
        return da is None and db is None
    return text_date(da.group(1)) == text_date(db.group(1))


# ---- families ------------------------------------------------------------------------------------
RX_SPREAD = re.compile(r"wins (?:the \w+ (?:half|quarter|innings?) )?by (?:more than|over) \d", re.IGNORECASE)
RX_WIN = re.compile(r"\bwins\b|\bwinner\b", re.IGNORECASE)


def spread_win_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    pairs = []
    for s in names:
        for spread_sfx, win_sfx in (("SPREAD", "GAME"), ("1HSPREAD", "1H"), ("F5SPREAD", "F5")):
            if s.endswith(spread_sfx) and not (spread_sfx == "SPREAD" and re.search(r"(1H|2H|\dQ|F5)SPREAD$", s)):
                w = s[: -len(spread_sfx)] + win_sfx
                if w in names:
                    pairs.append((s, w))
    rows = []
    for sa, sb in sorted(pairs):
        m = loader([sa, sb])
        sp = m.filter(pl.col("series_ticker") == sa).with_columns(
            pl.col("sfx").map_elements(team_code, return_dtype=pl.String).alias("team"))
        wn = m.filter((pl.col("series_ticker") == sb) & (pl.col("sfx") != "TIE")).rename(
            {"ticker": "b", "rules_primary": "rb", "sfx": "team"}).select("gkey", "team", "b", "rb")
        j = sp.join(wn, on=["gkey", "team"], how="inner")
        for r in j.iter_rows(named=True):
            if RX_SPREAD.search(r["rules_primary"] or "") and RX_WIN.search(r["rb"] or "") \
                    and not re.search(r"by (?:more than|over)", r["rb"] or "") and _same_sched(r["rules_primary"], r["rb"]):
                rows.append(_link("spread_win", r["ticker"], [r["b"]], f"{sb}|{r['gkey']}",
                                  f"{r['yes_sub_title']} => {r['team']} wins"))
    return _links(rows)


def _over_pairs(family: str, sa: str, sb: str, by_team: bool, rx_a: str, rx_b: str, loader=None) -> list[dict[str, Any]]:
    loader = loader or load_markets
    """``A over k`` => ``B over m`` for the largest B strike m <= k (same game, same team if team-level)."""
    m = loader([sa, sb]).with_columns(
        pl.col("yes_sub_title").map_elements(over_strike, return_dtype=pl.Float64).alias("k"))
    a = m.filter((pl.col("series_ticker") == sa) & pl.col("k").is_not_null())
    b = m.filter((pl.col("series_ticker") == sb) & pl.col("k").is_not_null())
    keys = ["gkey"]
    if by_team:
        a = a.with_columns(pl.col("sfx").map_elements(team_code, return_dtype=pl.String).alias("team"))
        b = b.with_columns(pl.col("sfx").map_elements(team_code, return_dtype=pl.String).alias("team"))
        keys.append("team")
    elif "TEAMTOTAL" in sa:
        a = a.with_columns(pl.col("sfx").map_elements(team_code, return_dtype=pl.String).alias("team"))
    bb = b.select(*keys, pl.col("k").alias("kb"), pl.col("ticker").alias("b"), pl.col("rules_primary").alias("rb"))
    j = a.join(bb, on=keys, how="inner").filter(pl.col("kb") <= pl.col("k"))
    j = j.sort("kb").group_by("ticker", maintain_order=True).last()
    rows = []
    for r in j.iter_rows(named=True):
        if re.search(rx_a, r["rules_primary"] or "", re.IGNORECASE) and re.search(rx_b, r["rb"] or "", re.IGNORECASE) \
                and _same_sched(r["rules_primary"], r["rb"]):
            rows.append(_link(family, r["ticker"], [r["b"]], f"{sb}|{r['gkey']}", f"over {r['k']} => over {r['kb']}"))
    return rows


def total_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows: list[dict[str, Any]] = []
    team = r"\bscores? (?:over|more than) \d"
    tot = r"collectively score more"
    for s in sorted(names):
        if s.endswith("TEAMTOTAL") and not re.search(r"(1H|2H)TEAMTOTAL$", s):
            b = s[: -len("TEAMTOTAL")] + "TOTAL"
            if b in names:
                rows += _over_pairs("tt_total", s, b, False, team, tot, loader)
        if s.endswith("1HTEAMTOTAL"):
            base = s[: -len("1HTEAMTOTAL")]
            if base + "TEAMTOTAL" in names:
                rows += _over_pairs("period_total", s, base + "TEAMTOTAL", True, r"1st half", team, loader)
            if base + "1HTOTAL" in names:
                rows += _over_pairs("tt_total", s, base + "1HTOTAL", False, team, tot, loader)
        for per in ("1HTOTAL", "2HTOTAL", "F5TOTAL"):
            if s.endswith(per) and not s.endswith("TEAM" + per):
                b = s[: -len(per)] + "TOTAL"
                if b in names and b != s:
                    rows += _over_pairs("period_total", s, b, False, r"(1st|2nd) half|first 5 innings", tot, loader)
    return _links(rows)


def first_any_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    specs = [(s, s.replace("FIRSTGOAL", "GOAL"), "goal") for s in names if s.endswith("FIRSTGOAL")]
    specs += [("KXNFLFIRSTTD", "KXNFLTD", "td"), ("KXNFLTEAMFIRSTTD", "KXNFLTD", "td"),
              ("KXNFLFIRSTTD", "KXNFLTEAMFIRSTTD", "team")]
    for sa, sb, kind in specs:
        if sa not in names or sb not in names:
            continue
        m = loader([sa, sb])
        a = m.filter(pl.col("series_ticker") == sa).with_columns(pl.col("sfx").str.replace_all("-", "").alias("player"))
        b = m.filter(pl.col("series_ticker") == sb)
        if kind == "team":
            b = b.with_columns(pl.col("sfx").str.replace_all("-", "").alias("player"))
        else:
            b = b.filter(pl.col("sfx").str.ends_with("-1")).with_columns(
                pl.col("sfx").str.replace(r"-1$", "").str.replace_all("-", "").alias("player"))
        bb = b.select("gkey", "player", pl.col("ticker").alias("b"), pl.col("rules_primary").alias("rb"),
                      pl.col("yes_sub_title").alias("nb"))
        j = a.join(bb, on=["gkey", "player"], how="inner")
        for r in j.iter_rows(named=True):
            name = (r["yes_sub_title"] or "").strip()
            if not name or not (r["nb"] or "").startswith(name) or not _same_sched(r["rules_primary"], r["rb"]):
                continue
            ok_a = re.search(r"(first|1st) (goal|touchdown)", r["rules_primary"] or "", re.IGNORECASE)
            ok_b = re.search(r"at least 1 goal|1\+ goals|at least 1\+ touchdowns|first touchdown for",
                             r["rb"] or "", re.IGNORECASE)
            if ok_a and ok_b:
                rows.append(_link("first_any", r["ticker"], [r["b"]], f"{sb}|{r['gkey']}", f"{name}: {sa} => {sb}"))
    return _links(rows)


# (A series, B series, k -> implied B k) for player-stat ladders; X >= k implies Y >= f(k)
PROP_RULES = [
    ("KXMLBHIT", "KXMLBTB", lambda k: k, "hits <= total bases"),
    ("KXMLBHIT", "KXMLBHRR", lambda k: k, "hits <= hits+runs+RBIs"),
    ("KXMLBHR", "KXMLBRBI", lambda k: k, "every home run is an RBI"),
    ("KXMLBHR", "KXMLBTB", lambda k: 4 * k, "a home run is 4 total bases"),
    ("KXMLBHR", "KXMLBHRR", lambda k: 3 * k, "a home run is a hit, a run and an RBI"),
    ("KXMLBRBI", "KXMLBHRR", lambda k: k, "RBIs <= hits+runs+RBIs"),
    ("KXNBAPTS", "KXNBAPRA", lambda k: k, "points <= points+rebounds+assists"),
    ("KXNBAREB", "KXNBAPRA", lambda k: k, "rebounds <= PRA"),
    ("KXNBAAST", "KXNBAPRA", lambda k: k, "assists <= PRA"),
    ("KXNBA3PT", "KXNBAPTS", lambda k: 3 * k, "a three is 3 points"),
    ("KXNHLGOAL", "KXNHLPTS", lambda k: k, "goals <= points"),
    ("KXNHLAST", "KXNHLPTS", lambda k: k, "assists <= points"),
]


def prop_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    for sa, sb, f, why in PROP_RULES:
        if sa not in names or sb not in names:
            continue
        m = loader([sa, sb]).with_columns(
            pl.col("sfx").map_elements(lambda s: (ladder_k(s) or ("", -1))[0], return_dtype=pl.String).alias("player"),
            pl.col("sfx").map_elements(lambda s: (ladder_k(s) or ("", -1))[1], return_dtype=pl.Int64).alias("k"))
        a = m.filter((pl.col("series_ticker") == sa) & (pl.col("k") > 0))
        a = a.with_columns(pl.col("k").map_elements(f, return_dtype=pl.Int64).alias("need"))
        b = m.filter((pl.col("series_ticker") == sb) & (pl.col("k") > 0)).select(
            "gkey", "player", pl.col("k").alias("kb"), pl.col("ticker").alias("b"), pl.col("rules_primary").alias("rb"))
        j = a.join(b, on=["gkey", "player"], how="inner").filter(pl.col("kb") <= pl.col("need"))
        j = j.sort("kb").group_by("ticker", maintain_order=True).last()
        for r in j.iter_rows(named=True):
            if re.search(r"records? \d+\+|at least", r["rules_primary"] or "") and re.search(r"records? \d+\+", r["rb"] or "") \
                    and _same_sched(r["rules_primary"], r["rb"]):
                rows.append(_link("prop_cross", r["ticker"], [r["b"]], f"{sb}|{r['gkey']}",
                                  f"{sa} {r['k']}+ => {sb} {r['kb']}+ ({why})"))
    return _links(rows)


def protected_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    for s in sorted(names):
        if not s.endswith("PROT") or s[:-4] not in names:
            continue
        m = loader([s, s[:-4]])
        a = m.filter(pl.col("series_ticker") == s)
        b = m.filter(pl.col("series_ticker") == s[:-4]).select("gkey", "sfx", pl.col("ticker").alias("b"),
                                                               pl.col("rules_primary").alias("rb"))
        j = a.join(b, on=["gkey", "sfx"], how="inner")
        for r in j.iter_rows(named=True):
            # same player, stat and threshold: rules_primary differ at most in wording of "records"/"scores"
            na = re.sub(r"\b(records|scores at least)\b", "", r["rules_primary"] or "")
            nb = re.sub(r"\b(records|scores at least)\b", "", r["rb"] or "")
            if re.sub(r"\s+", " ", na).strip() != re.sub(r"\s+", " ", nb).strip():
                continue
            cl = f"{s[:-4]}|{r['gkey']}"
            rows.append(_link("protected", r["ticker"], [r["b"]], cl, r["yes_sub_title"], kind="equal"))
            rows.append(_link("protected", r["b"], [r["ticker"]], cl, r["yes_sub_title"], kind="equal"))
    return _links(rows)


LADDERS = [
    ["KXPGATOUR", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20", "KXPGAMAKECUT"],
    ["KXPGAR1LEAD", "KXPGAR1TOP5", "KXPGAR1TOP10"], ["KXPGAR2LEAD", "KXPGAR2TOP10"], ["KXPGAR3LEAD", "KXPGAR3TOP10"],
    ["KXDPWORLDTOUR", "KXDPWTTOP5", "KXDPWTTOP10", "KXDPWTTOP20", "KXDPWORLDTOURMAKECUT"],
    ["KXLIVTOUR", "KXLIVTOP5", "KXLIVTOP10"],
    ["KXNASCARRACE", "KXNASCARTOP3", "KXNASCARTOP5", "KXNASCARTOP10", "KXNASCARTOP20"],
    ["KXF1RACE", "KXF1RACEPODIUM", "KXF1TOP5", "KXF1TOP10"], ["KXF1SPRINTTOP5", "KXF1SPRINTTOP10"],
    ["KXINDYCARRACE", "KXINDYCARTOP3", "KXINDYCARTOP5", "KXINDYCARTOP10"],
]
RX_FINISH = re.compile(r"\bwins\b|finishes (?:in )?(?:first|the top \d+)|top \d+|made the cut|is the leader", re.IGNORECASE)


def ladder_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    for lad in LADDERS:
        lad = [s for s in lad if s in names]
        if len(lad) < 2:
            continue
        m = loader(lad)
        for sa, sb in itertools.pairwise(lad):
            a = m.filter(pl.col("series_ticker") == sa)
            b = m.filter(pl.col("series_ticker") == sb).select("gkey", "sfx", pl.col("ticker").alias("b"),
                                                               pl.col("rules_primary").alias("rb"),
                                                               pl.col("yes_sub_title").alias("nb"))
            j = a.join(b, on=["gkey", "sfx"], how="inner")
            for r in j.iter_rows(named=True):
                if (r["yes_sub_title"] or "") != (r["nb"] or "") and not (r["nb"] or "").startswith(r["yes_sub_title"] or "?"):
                    continue
                if RX_FINISH.search(r["rules_primary"] or "") and RX_FINISH.search(r["rb"] or ""):
                    rows.append(_link("finish_ladder", r["ticker"], [r["b"]], f"{lad[0]}|{r['gkey']}",
                                      f"{r['yes_sub_title']}: {sa} => {sb}"))
    return _links(rows)


def exact_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    specs = [(s, s.replace("EXACTMATCH", "MATCH"), "set") for s in names if s.endswith("EXACTMATCH")]
    specs += [("KXUFCMOV", "KXUFCFIGHT", "mov")]
    specs += [(s, s[: -len("SCORE")] + "GAME", "score") for s in names
              if s.endswith("SCORE") and not s.endswith("1HSCORE") and s[: -len("SCORE")] + "GAME" in names]
    for sa, sb, kind in specs:
        if sa not in names or sb not in names:
            continue
        m = loader([sa, sb])
        a = m.filter(pl.col("series_ticker") == sa)
        if kind == "set":
            a = a.with_columns(pl.col("sfx").str.replace(r"\d{2}$", "").alias("side"))
        elif kind == "mov":
            a = a.with_columns(pl.col("sfx").str.replace(r"(DEC|KOTKODQ|SUB|KO|TKO)$", "").alias("side")).filter(
                pl.col("side") != pl.col("sfx"))
        else:
            a = a.with_columns(pl.col("sfx").map_elements(lambda s: score_winner(s) or "", return_dtype=pl.String)
                               .alias("side")).filter(pl.col("side") != "")
        b = m.filter(pl.col("series_ticker") == sb).select("gkey", pl.col("sfx").alias("side"), pl.col("ticker").alias("b"),
                                                           pl.col("rules_primary").alias("rb"))
        j = a.join(b, on=["gkey", "side"], how="inner")
        for r in j.iter_rows(named=True):
            if not _same_sched(r["rules_primary"], r["rb"]):
                continue
            ra = r["rules_primary"] or ""
            ok = (kind == "set" and re.search(r"set score of \d-\d", ra)) or \
                 (kind == "mov" and re.search(r"wins by (Decision|KO|Submission)", ra, re.IGNORECASE)) or \
                 (kind == "score" and re.search(r"score|draw|wins \d", (r["yes_sub_title"] or "") + ra, re.IGNORECASE))
            if ok:
                rows.append(_link("exact_win", r["ticker"], [r["b"]], f"{sb}|{r['gkey']}",
                                  f"{r['yes_sub_title']} => {r['side']}"))
    return _links(rows)


def chart_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    for sa, sb in (("KXTOPSONG", "KXBBCHARTPOSITIONSONG"), ("KXBILLBOARDRUNNERUPSONG", "KXBBCHARTPOSITIONSONG"),
                   ("KXTOPALBUM", "KXBBCHARTPOSITIONALBUM"), ("KXBILLBOARDRUNNERUPALBUM", "KXBBCHARTPOSITIONALBUM")):
        if sa not in names or sb not in names:
            continue
        m = loader([sa, sb])
        a = m.filter(pl.col("series_ticker") == sa)
        b = m.filter(pl.col("series_ticker") == sb).select("title", pl.col("ticker").alias("b"), pl.col("rules_primary").alias("rb"))
        j = a.join(b, on="title", how="inner")
        for r in j.iter_rows(named=True):
            pos = re.search(r"is (?:ranked )?#(\d)", r["rules_primary"] or "")
            posb = re.search(r"is (?:ranked )?#(\d)", r["rb"] or "")
            wk, wkb = text_date((r["rules_primary"] or "").split("Week of")[-1]), text_date((r["rb"] or "").split("Week of")[-1])
            if pos and posb and pos.group(1) == posb.group(1) and wk == wkb and wk is not None:
                cl = f"{sb}|{r['b']}"
                rows.append(_link("chart_dup", r["ticker"], [r["b"]], cl, r["title"], kind="equal"))
                rows.append(_link("chart_dup", r["b"], [r["ticker"]], cl, r["title"], kind="equal"))
    return _links(rows)


SAY_SERIES = ["KXTRUMPSAY", "KXTRUMPSAYMONTH", "KXTRUMPSAYCOMPANY", "KXTRUMPSAYCOUNTRY", "KXTRUMPSAYNICKNAME"]
MENTION_SERIES = ["KXTRUMPMENTION", "KXTRUMPMENTIONB"]


def mention_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    m = loader([s for s in SAY_SERIES + MENTION_SERIES if s in names])
    say = m.filter(pl.col("series_ticker").is_in(SAY_SERIES))
    men = m.filter(pl.col("series_ticker").is_in(MENTION_SERIES))
    sw = []
    for r in say.iter_rows(named=True):
        st, en = say_window(r["rules_primary"], r["open_time"])
        if st and en and re.search(r"is stated by Donald Trump", r["rules_primary"] or ""):
            sw.append({**r, "w": word_set(r["yes_sub_title"]), "st": st, "en": en})
    rows = []
    for r in men.iter_rows(named=True):
        if not re.search(r"Donald Trump says", r["rules_primary"] or ""):
            continue
        w = word_set(r["yes_sub_title"])
        d = code_date(r["gkey"]) or text_date(r["title"])
        if not w or d is None:
            continue
        # the speech's ET calendar day must sit strictly inside the window (a whole day of margin)
        for s in sw:
            st_d, en_d = et_date(s["st"]), et_date(s["en"])
            if st_d < d < en_d and w <= s["w"]:
                rows.append(_link("mention_say", r["ticker"], [s["ticker"]], f"TRUMP|{s['event_ticker']}",
                                  f"{r['yes_sub_title']} @ {d} => {s['yes_sub_title']} in ({st_d}, {en_d})"))
    # nested say windows (weekly inside monthly), same or wider word set
    for a in sw:
        for b in sw:
            if a["ticker"] != b["ticker"] and a["st"] >= b["st"] and a["en"] <= b["en"] and a["w"] <= b["w"] \
                    and (a["st"], a["en"]) != (b["st"], b["en"]):
                rows.append(_link("say_nested", a["ticker"], [b["ticker"]], f"TRUMP|{b['event_ticker']}",
                                  f"{a['yes_sub_title']} [{a['st']:%m-%d}, {a['en']:%m-%d}) => [{b['st']:%m-%d}, {b['en']:%m-%d})"))
            elif a["ticker"] < b["ticker"] and (a["st"], a["en"]) == (b["st"], b["en"]) and a["w"] == b["w"] \
                    and a["series_ticker"] != b["series_ticker"]:
                cl = f"TRUMP|{b['event_ticker']}"
                rows.append(_link("say_nested", a["ticker"], [b["ticker"]], cl, f"dup {a['yes_sub_title']}", kind="equal"))
                rows.append(_link("say_nested", b["ticker"], [a["ticker"]], cl, f"dup {a['yes_sub_title']}", kind="equal"))
    # the same speech and word in two mention series (identical title)
    j = men.join(men, on=["title", "yes_sub_title"], suffix="_b").filter(
        (pl.col("series_ticker") < pl.col("series_ticker_b"))
        & (pl.col("gkey").str.slice(0, 7) == pl.col("gkey_b").str.slice(0, 7)))
    for r in j.iter_rows(named=True):
        cl = f"TRUMP|{r['event_ticker']}"
        rows.append(_link("mention_dup", r["ticker"], [r["ticker_b"]], cl, f"{r['title'][:60]}: {r['yes_sub_title']}",
                          kind="equal"))
        rows.append(_link("mention_dup", r["ticker_b"], [r["ticker"]], cl, f"{r['title'][:60]}: {r['yes_sub_title']}",
                          kind="equal"))
    return _links(rows)


def weather_links(names: set[str], loader=None) -> pl.DataFrame:
    loader = loader or load_markets
    rows = []
    # rain on a day => rain on the weekend that contains it (same CLI station)
    m = loader(["KXRAIN", "KXRAINWKND"])
    day = m.filter(pl.col("series_ticker") == "KXRAIN")
    wk = m.filter(pl.col("series_ticker") == "KXRAINWKND")
    st_rx = re.compile(r"precipitation at (CLI[A-Z]+)")
    wkr = []
    for r in wk.iter_rows(named=True):
        g = re.search(r"within (\w+ \d{1,2}, \d{4}) through (\w+ \d{1,2}, \d{4})", r["rules_primary"] or "")
        s = st_rx.search(r["rules_primary"] or "")
        if g and s:
            wkr.append((s.group(1), text_date(g.group(1)), text_date(g.group(2)), r["ticker"], r["event_ticker"]))
    for r in day.iter_rows(named=True):
        s = st_rx.search(r["rules_primary"] or "")
        d = text_date((r["rules_primary"] or "").split(" in ", 2)[-1]) or code_date(r["gkey"])
        if not s or d is None:
            continue
        for st, d0, d1, tk, ev in wkr:
            if st == s.group(1) and d0 and d1 and d0 <= d <= d1:
                rows.append(_link("rain_weekend", r["ticker"], [tk], f"RAIN|{ev}", f"{st} {d} in [{d0}, {d1}]"))
    # daily low >= lo  =>  NOT (daily high <= hi) for hi < lo, same CLI station and day
    hi_s = sorted(s for s in names if re.fullmatch(r"KXHIGHT?[A-Z]+", s))
    lo_s = sorted(s for s in names if re.fullmatch(r"KXLOWT[A-Z]+", s))
    m = loader(hi_s + lo_s).with_columns(
        pl.col("rules_primary").str.extract(r"\((CLI[A-Z]+)\)").alias("station"),
        pl.col("rules_primary").str.extract(r"for (\w+ \d{1,2}, \d{4})").alias("day"),
        pl.col("yes_sub_title").map_elements(lambda s: list(wx_range(s) or (math.nan, math.nan)),
                                             return_dtype=pl.List(pl.Float64)).alias("rng"))
    m = m.filter(pl.col("station").is_not_null() & pl.col("day").is_not_null()).with_columns(
        pl.col("rng").list.get(0).alias("lo"), pl.col("rng").list.get(1).alias("hi"))
    lows = m.filter(pl.col("series_ticker").is_in(lo_s) & pl.col("rules_primary").str.contains("minimum temperature")
                    & pl.col("lo").is_finite())
    highs = m.filter(pl.col("series_ticker").is_in(hi_s) & pl.col("rules_primary").str.contains("maximum temperature")
                     & pl.col("hi").is_finite()).select("station", "day", pl.col("hi").alias("hb"),
                                                         pl.col("ticker").alias("b"))
    j = lows.join(highs, on=["station", "day"], how="inner").filter(pl.col("hb") < pl.col("lo"))
    for r in j.iter_rows(named=True):
        rows.append(_link("wx_hilo", r["ticker"], [r["b"]], f"WX|{r['station']}|{r['day']}",
                          f"low >= {r['lo']:.0f} => high not <= {r['hb']:.0f}", b_side="no"))
    return _links(rows)


FAMILIES = {
    "spread_win": spread_win_links, "totals": total_links, "first_any": first_any_links, "prop_cross": prop_links,
    "protected": protected_links, "finish_ladder": ladder_links, "exact_win": exact_links, "chart_dup": chart_links,
    "mentions": mention_links, "weather": weather_links,
}


def build_links(families: list[str] | None = None, loader=None, names: set[str] | None = None) -> pl.DataFrame:
    names = names if names is not None else _series_names()
    frames = []
    for name, fn in FAMILIES.items():
        if families and name not in families:
            continue
        df = fn(names, loader)
        print(f"  {name}: {df.height} links", flush=True)
        frames.append(df)
    return pl.concat(frames).unique("link_id", keep="first", maintain_order=True)


# ---- candles: caches, fetch plan, fetch ----------------------------------------------------------
CUTOFF = datetime(2026, 9, 18, tzinfo=UTC)        # /historical/cutoff market_settled_ts at fetch time
CACHES = [  # (name, glob, OHLC?) in priority order after our own fetch
    ("xh", "data/research/xcat/hourly/part-*.parquet", True),
    ("kc", "data/tables/kalshi_candles.parquet", False),
    ("sm", "data/research/pm_smile/candles_fetched.parquet", False),
    ("hz", "data/research/pm_hazard/candles.parquet", False),
]
WINDOW_DAYS = 7          # hourly candles over the last week of each market's life
MAX_BATCH_TICKERS = 50
MAX_BATCH_CANDLES = 10000
# clusters (games / tournaments / weeks) drawn per stratum (= family x B series), and per family
SAMPLE_PER_STRATUM = {"prop_cross": 40, "protected": 30, "first_any": 25, "spread_win": 20, "period_total": 25,
                      "tt_total": 20, "exact_win": 20, "finish_ladder": 1000, "chart_dup": 1000, "rain_weekend": 1000}
SAMPLE_PER_FAMILY = {"spread_win": 300, "period_total": 200, "tt_total": 160, "exact_win": 200, "first_any": 150,
                     "prop_cross": 110, "protected": 60}
HIST_FAMILIES = ("chart_dup", "rain_weekend")    # also fetched one by one before the cutoff
SEED = 20261117


def cached_tickers() -> set[str]:
    out: set[str] = set()
    for _, g, _ in CACHES:
        if list(Path(".").glob(g)):
            out |= set(pl.scan_parquet(g).select("ticker").unique().collect()["ticker"])
    for p in (OUT / "hourly").glob("part-*.parquet"):
        out |= set(pl.read_parquet(p, columns=["ticker"])["ticker"])
    return out


def leg_markets(links: pl.DataFrame) -> pl.DataFrame:
    legs = pl.concat([links.select(pl.col("a").alias("ticker")), links.select(pl.col("b").explode().alias("ticker"))]).unique()
    cols = [c for c in MARKET_COLS if c not in ("rules_primary", "title")]
    return (pl.scan_parquet(TABLES / "kalshi_markets.parquet").select(cols)
            .join(legs.lazy(), on="ticker", how="semi").collect())


def pack_jobs(rows: list[dict[str, Any]], period: int = 60) -> list[dict[str, Any]]:
    """Greedy batches of live markets within the API caps (<= 50 tickers, tickers x span <= 10,000
    periods); markets settled before the cutoff go one per request to the historical endpoint."""
    jobs: list[dict[str, Any]] = [{"kind": "hist", "rows": [r]} for r in rows if r["settlement_ts"] < CUTOFF]
    live = sorted((r for r in rows if r["settlement_ts"] >= CUTOFF), key=lambda r: (r["lo"], r["hi"]))
    cur: list[dict[str, Any]] = []
    lo = hi = None
    for r in live:
        nlo = r["lo"] if lo is None else min(lo, r["lo"])
        nhi = r["hi"] if hi is None else max(hi, r["hi"])
        span = math.ceil((nhi - nlo).total_seconds() / 60 / period) + 1
        if cur and (len(cur) + 1 > MAX_BATCH_TICKERS or span * (len(cur) + 1) > MAX_BATCH_CANDLES):
            jobs.append({"kind": "batch", "rows": cur})
            cur, nlo, nhi = [], r["lo"], r["hi"]
        cur.append(r)
        lo, hi = nlo, nhi
    if cur:
        jobs.append({"kind": "batch", "rows": cur})
    return jobs


def _unit(cluster: str) -> str:
    return cluster.split("|")[-1]


def plan(links: pl.DataFrame | None = None) -> pl.DataFrame:
    """Seeded, outcome-blind sample of clusters whose legs are not cached; writes ``fetch_plan.parquet``
    (one row per market to fetch with its window) and returns it."""
    links = links if links is not None else pl.read_parquet(OUT / "links.parquet")
    have = cached_tickers()
    m = leg_markets(links)
    st = dict(zip(m["ticker"], m["settlement_ts"], strict=True))
    lk = links.with_columns(pl.col("cluster").str.split("|").list.first().alias("stratum"),
                            pl.col("cluster").map_elements(_unit, return_dtype=pl.String).alias("unit"))
    lk = lk.with_columns(
        pl.concat_list(pl.col("a"), pl.col("b")).alias("legs")).with_columns(
        pl.col("legs").list.eval(pl.element().is_in(list(have))).list.all().alias("cached"),
        pl.col("legs").map_elements(lambda xs: min(st.get(x, CUTOFF) for x in xs) >= CUTOFF,
                                    return_dtype=pl.Boolean).alias("live"))
    need = lk.filter(~pl.col("cached") & (pl.col("live") | pl.col("family").is_in(HIST_FAMILIES))
                     & pl.col("family").is_in(list(SAMPLE_PER_STRATUM)))
    pick = []
    for fam in sorted(need["family"].unique()):
        f = need.filter(pl.col("family") == fam)
        units = f.select("stratum", "unit").unique().sort("stratum", "unit")
        units = units.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("stratum").alias("r"))
        units = units.filter(pl.col("r") < SAMPLE_PER_STRATUM[fam])
        cap = SAMPLE_PER_FAMILY.get(fam)
        if cap is not None and units.height > cap:
            units = units.sample(cap, seed=SEED, shuffle=True)
        pick.append(f.join(units.select("stratum", "unit"), on=["stratum", "unit"], how="semi"))
    chosen = pl.concat(pick)
    tick = chosen.select(pl.col("legs").explode().alias("ticker")).unique()
    tick = tick.filter(~pl.col("ticker").is_in(list(have)))
    p = m.join(tick, on="ticker", how="semi").with_columns(
        pl.max_horizontal(pl.col("open_time"), pl.col("close_time") - timedelta(days=WINDOW_DAYS)).dt.truncate("1h").alias("lo"),
        pl.col("close_time").alias("hi"), pl.lit(60).alias("period"))
    p = p.filter(pl.col("hi") > pl.col("lo"))
    p.write_parquet(OUT / "fetch_plan.parquet")
    chosen.select("link_id", "family", "stratum", "unit").write_parquet(OUT / "sample_links.parquet")
    jobs = pack_jobs(p.select("ticker", "event_ticker", "settlement_ts", "lo", "hi", "period").to_dicts())
    by = chosen.group_by("family").agg(pl.len().alias("links"), pl.col("unit").n_unique().alias("units"))
    print(by.sort("family"))
    print(f"plan: {p.height} markets, {len(jobs)} requests "
          f"({sum(j['kind'] == 'hist' for j in jobs)} historical)", flush=True)
    return p


def fetch(s, *, workers: int = 2, min_interval: float = 0.5, save_every: int = 100, max_requests: int | None = None) -> None:
    """Fetch ``fetch_plan.parquet`` (resumable; raw responses cached append-only by ``CachedClient``)."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from albert.research.pm_hazard import _api
    from albert.research.xcat import OHLC_SCHEMA, _run_job

    d = OUT / "hourly"
    d.mkdir(parents=True, exist_ok=True)
    p = pl.read_parquet(OUT / "fetch_plan.parquet")
    done_path = d / "done.parquet"
    done = set(pl.read_parquet(done_path)["ticker"]) if done_path.exists() else set()
    todo = p.filter(~pl.col("ticker").is_in(list(done)))
    api = _api(s, min_interval, nap_cap=5.0, retries=60)
    cut = api.cutoff()["market_settled_ts"]
    if cut != CUTOFF:
        raise RuntimeError(f"historical cutoff moved to {cut}: re-plan")
    jobs = pack_jobs(todo.select("ticker", "event_ticker", "settlement_ts", "lo", "hi", "period").to_dicts())
    if max_requests is not None:
        jobs = jobs[:max_requests]
    print(f"fetch: {p.height} planned, {len(done)} done, {todo.height} to fetch in {len(jobs)} requests", flush=True)
    lock = threading.Lock()
    buf: list[dict[str, Any]] = []
    fin: list[str] = []
    failed: list[str] = []
    part = len(list(d.glob("part-*.parquet")))
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, part
        if buf:
            pl.DataFrame(buf, schema=OHLC_SCHEMA).write_parquet(d / f"part-{part:05d}.parquet")
            part += 1
        buf = []
        pl.DataFrame({"ticker": sorted(done | set(fin) | set(failed))}, schema={"ticker": pl.String}).write_parquet(done_path)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda j: _run_job(api, j), jobs):
            with lock:
                for tk, got in res:
                    if got is None:
                        failed.append(tk)
                        continue
                    buf.extend(got)
                    fin.append(tk)
                n += 1
                if n % save_every == 0:
                    flush()
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(jobs)} requests, {len(fin)} markets, {rate:.2f} req/s", flush=True)
    flush()
    print(f"fetch done: {len(fin)} markets, {len(set(failed))} failed", flush=True)


# ---- quote panel ---------------------------------------------------------------------------------
def load_candles(tickers: list[str]) -> pl.DataFrame:
    """``(ticker, t, bid, ask, high, low, volume)`` from our fetch first, then the caches."""
    frames = []
    own = sorted((OUT / "hourly").glob("part-*.parquet"))
    srcs = ([("own", str(OUT / "hourly" / "part-*.parquet"), True)] if own else []) + CACHES
    for rank, (_, g, ohlc) in enumerate(srcs):
        if not list(Path(".").glob(g)):
            continue
        lf = pl.scan_parquet(g).filter(pl.col("ticker").is_in(tickers))
        if ohlc:
            lf = lf.filter(pl.col("period") == 60).select(
                "ticker", pl.col("end_ts").alias("t"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"),
                pl.col("p_high").alias("high"), pl.col("p_low").alias("low"), "volume")
        else:
            lf = lf.select("ticker", pl.col("end_ts").alias("t"), pl.col("yes_bid_close").alias("bid"),
                           pl.col("yes_ask_close").alias("ask"), pl.lit(None, pl.Float64).alias("high"),
                           pl.lit(None, pl.Float64).alias("low"), "volume")
        frames.append(lf.with_columns(pl.lit(rank).alias("rank"), pl.lit(ohlc).alias("ohlc")))
    if not frames:
        return pl.DataFrame()
    c = pl.concat(frames, how="diagonal_relaxed").collect()
    c = c.with_columns(pl.col("t").cast(pl.Datetime("us", "UTC")))
    # one source per ticker (the best-ranked one that has it): never splice two sources' hours
    best = c.group_by("ticker").agg(pl.col("rank").min().alias("best"))
    c = c.join(best, on="ticker").filter(pl.col("rank") == pl.col("best")).drop("best")
    return c.unique(["ticker", "t"], keep="first").sort("ticker", "t")


def quote_grid(c: pl.DataFrame, m: pl.DataFrame) -> pl.DataFrame:
    """Hourly grid per market from its first candle to the last hour before close: quotes carried
    forward over omitted hours (``real`` = a candle exists, so a trade range is known). Candles
    ending at or after the actual close are dropped."""
    c = c.join(m.select("ticker", "close_time", "open_time"), on="ticker", how="inner").filter(
        (pl.col("t") < pl.col("close_time")) & (pl.col("t") > pl.col("open_time")))
    g = (c.sort("ticker", "t").with_columns(pl.lit(True).alias("real"))
         .drop("close_time", "open_time").upsample("t", every="1h", group_by="ticker", maintain_order=True))
    g = g.with_columns(pl.col("bid", "ask", "rank", "ohlc").forward_fill().over("ticker"),
                       pl.col("volume").fill_null(0.0), pl.col("real").fill_null(False))
    return g.with_columns(pl.col("ticker").forward_fill().over("ticker"))


# ---- gaps ----------------------------------------------------------------------------------------
TAKER = 0.07
MAKER = 0.0175
N_CONTRACTS = 5          # contracts per position (both legs)
EVENT_CAP = 10.0         # open cost per game / event cluster (both legs count)
BANKROLL = 100.0


def fee_expr(n: pl.Expr | float, p: pl.Expr, coef: pl.Expr | float, mult: pl.Expr | float = 1.0) -> pl.Expr:
    """Kalshi fee for one order, rounded up to the cent (the per-order worst case)."""
    raw = pl.lit(coef) * mult * n * p * (1 - p) * 100 if not isinstance(coef, pl.Expr) else coef * mult * n * p * (1 - p) * 100
    return raw.round(9).ceil() / 100


def fee_order(n: float, p: float, coef: float = TAKER, mult: float = 1.0) -> float:
    if n <= 0 or p <= 0 or p >= 1:
        return 0.0
    return math.ceil(round(coef * mult * n * p * (1 - p) * 100, 9)) / 100


def leg_table(links: pl.DataFrame) -> pl.DataFrame:
    """One row per (link, leg): role ``a`` is sold (bought as NO), ``b`` legs are bought on ``b_side``."""
    a = links.select("link_id", pl.col("a").alias("ticker"), pl.lit("a").alias("role"), pl.lit("no").alias("side"))
    b = links.select("link_id", pl.col("b").alias("ticker"), pl.lit("b").alias("role"), pl.col("b_side").alias("side")
                     ).explode("ticker")
    return pl.concat([a, b])


def link_panel(links: pl.DataFrame, grid: pl.DataFrame, m: pl.DataFrame, n: int = N_CONTRACTS) -> pl.DataFrame:
    """Per (link, hour) where every leg is quoted: cost of the lock (buy A-NO + buy B), gross gap
    ``1 - cost``, taker fees for ``n`` contracts per leg and the net gap per contract; plus A's
    trade range and B's cost in the *next* hour for the maker test."""
    legs = leg_table(links)
    nlegs = legs.group_by("link_id").len("nlegs")
    fm = m.select("ticker", pl.col("fee_multiplier").fill_null(1.0).alias("mult"),
                  pl.col("fee_type").str.contains("maker_fee").fill_null(False).alias("mk"))
    x = legs.join(grid, on="ticker", how="inner").join(fm, on="ticker", how="left")
    # price paid for the leg's side; invalid when that side has no quote
    price = (pl.when(pl.col("side") == "yes").then(pl.when((pl.col("ask") > 0) & (pl.col("ask") < 1)).then(pl.col("ask")))
             .otherwise(pl.when((pl.col("bid") > 0) & (pl.col("bid") < 1)).then(1 - pl.col("bid"))))
    x = x.with_columns(price.alias("px")).with_columns(
        fee_expr(n, pl.col("px"), TAKER, pl.col("mult")).alias("fee"))
    agg = x.group_by("link_id", "t").agg(
        pl.len().alias("k"), pl.col("px").is_null().any().alias("bad"), pl.col("px").sum().alias("cost"),
        pl.col("fee").sum().alias("fee"),
        pl.col("px").filter(pl.col("role") == "a").first().alias("a_px"),
        pl.col("px").filter(pl.col("role") == "b").sum().alias("b_cost"),
        pl.col("fee").filter(pl.col("role") == "b").sum().alias("b_fee"),
        pl.col("bid").filter(pl.col("role") == "a").first().alias("a_bid"),
        pl.col("ask").filter(pl.col("role") == "a").first().alias("a_ask"),
        pl.col("high").filter(pl.col("role") == "a").first().alias("a_high"),
        pl.col("real").filter(pl.col("role") == "a").first().alias("a_real"),
        pl.col("mult").filter(pl.col("role") == "a").first().alias("a_mult"),
        pl.col("mk").filter(pl.col("role") == "a").first().alias("a_mk"),
        pl.col("ohlc").filter(pl.col("role") == "a").first().alias("a_ohlc"),
        pl.col("real").all().alias("all_real"))
    agg = agg.join(nlegs, on="link_id").filter((pl.col("k") == pl.col("nlegs")) & ~pl.col("bad")).drop("bad", "k")
    agg = agg.with_columns((1 - pl.col("cost")).alias("gap"),
                           (1 - pl.col("cost") - pl.col("fee") / n).alias("net")).sort("link_id", "t")
    nx = pl.col("t").shift(-1).over("link_id")
    agg = agg.with_columns(
        (nx == pl.col("t") + timedelta(hours=1)).fill_null(False).alias("has_next"),
        pl.col("a_high").shift(-1).over("link_id").alias("a_high_next"),
        pl.col("a_real").shift(-1).over("link_id").alias("a_real_next"),
        pl.col("b_cost").shift(-1).over("link_id").alias("b_cost_next"),
        pl.col("b_fee").shift(-1).over("link_id").alias("b_fee_next"),
        pl.col("net").shift(1).over("link_id").alias("net_prev"),
        (pl.col("t").shift(1).over("link_id") == pl.col("t") - timedelta(hours=1)).fill_null(False).alias("has_prev"))
    return agg.with_columns((pl.col("t") >= HOLDOUT_FROM).alias("holdout"))


def episodes(p: pl.DataFrame, col: str = "net", thr: float = 0.0) -> pl.DataFrame:
    """Runs of consecutive hourly snapshots with ``col >= thr`` per link: start, hours, max value."""
    x = p.select("link_id", "t", col).with_columns((pl.col(col) >= thr).alias("on")).sort("link_id", "t")
    brk = (~pl.col("on").shift(1).over("link_id").fill_null(False)) | (
        pl.col("t").shift(1).over("link_id") != pl.col("t") - timedelta(hours=1)).fill_null(True)
    x = x.filter(pl.col("on")).with_columns(brk.alias("new"))
    x = x.with_columns(pl.col("new").cum_sum().over("link_id").alias("ep"))
    return x.group_by("link_id", "ep").agg(pl.col("t").min().alias("start"), pl.len().alias("hours"),
                                           pl.col(col).max().alias("best"))


# ---- strategies ----------------------------------------------------------------------------------
def _legs_of(links: pl.DataFrame) -> dict[str, tuple[str, list[str], str]]:
    return {r["link_id"]: (r["a"], r["b"], r["b_side"]) for r in links.select("link_id", "a", "b", "b_side").iter_rows(named=True)}


def taker_entries(p: pl.DataFrame, thr: float = 0.0, persist: bool = False) -> pl.DataFrame:
    """First hour per link with net >= thr (``persist``: and net >= 0 an hour earlier too)."""
    f = p.filter(pl.col("net") >= thr)
    if persist:
        f = f.filter(pl.col("has_prev") & (pl.col("net_prev") >= 0))
    return f.sort("t").unique("link_id", keep="first").select(
        "link_id", "t", pl.col("a_px").alias("a_price"), "b_cost", pl.lit("taker").alias("how"))


def maker_entries(p: pl.DataFrame, margin: float = 0.01, reach: float = 0.0, n: int = N_CONTRACTS) -> pl.DataFrame:
    """Rest a YES sell on A at ``L`` = B's cost + taker fee on B + maker fee on A + margin (rounded up
    to the cent), for one hour, only if it neither crosses (L > bid_A) nor sits more than ``reach``
    above A's ask. Filled when a print in the next hour trades strictly above L (candle high; only
    OHLC candles qualify); B is then bought at the close of that hour (legging risk)."""
    f = p.filter(pl.col("a_ohlc") & pl.col("has_next"))
    f = f.with_columns(((pl.col("b_cost") + pl.col("b_fee") / n + margin) * 100).ceil().alias("L0") / 100)
    # maker fee on A at L (only on maker-fee series); one more cent if it would eat the margin
    f = f.with_columns(pl.when(pl.col("a_mk")).then(fee_expr(n, pl.col("L0"), MAKER, pl.col("a_mult")) / n)
                       .otherwise(0.0).alias("mfee"))
    f = f.with_columns(((pl.col("L0") + pl.col("mfee")) * 100).ceil().alias("L") / 100)
    f = f.filter((pl.col("L") > pl.col("a_bid")) & (pl.col("L") <= pl.col("a_ask") + reach + 1e-9) & (pl.col("L") < 1))
    filled = f.filter(pl.col("a_real_next").fill_null(False) & (pl.col("a_high_next") > pl.col("L") + 1e-9)
                      & pl.col("b_cost_next").is_not_null())
    return filled.sort("t").unique("link_id", keep="first").select(
        "link_id", (pl.col("t") + timedelta(hours=1)).alias("t"), (1 - pl.col("L")).alias("a_price"),
        pl.col("b_cost_next").alias("b_cost"), pl.lit("maker").alias("how"))


def positions(entries: pl.DataFrame, links: pl.DataFrame, m: pl.DataFrame, n: int = N_CONTRACTS) -> pl.DataFrame:
    """Leg rows (vol_metrics ledger schema) for each entry: A bought as NO at ``a_price``, each B leg
    at its own price at entry (recomputed from the panel's per-leg prices), held to settlement and
    paid on the actual settlement value."""
    info = {r["ticker"]: r for r in m.select("ticker", "settlement_ts", "settlement_value", "fee_type", "fee_multiplier"
                                               ).iter_rows(named=True)}
    legs = _legs_of(links)
    rows = []
    for e in entries.iter_rows(named=True):
        a, bs, side = legs[e["link_id"]]
        ia = info[a]
        mult_a = ia["fee_multiplier"] or 1.0
        if e["how"] == "maker":
            coef = MAKER if "maker_fee" in (ia["fee_type"] or "") else 0.0
            fa = fee_order(n, 1 - e["a_price"], coef, mult_a)
        else:
            fa = fee_order(n, e["a_price"], TAKER, mult_a)
        struct = f"{e['link_id']}|{e['t'].isoformat()}|{e['how']}"
        rows.append({"struct_id": struct, "link_id": e["link_id"], "ticker": a, "side": "no", "n": n, "t_in": e["t"],
                     "cost_usd": n * e["a_price"] + fa, "settle_ts": max(ia["settlement_ts"], e["t"]),
                     "payoff_usd": n * (1 - ia["settlement_value"]), "fee_usd": fa})
        # B legs: split the B cost by leg prices recorded in e["b_prices"] when present, else evenly
        prices = e.get("b_prices") or [e["b_cost"] / len(bs)] * len(bs)
        for tb, pb in zip(bs, prices, strict=True):
            ib = info[tb]
            fb = fee_order(n, pb, TAKER, ib["fee_multiplier"] or 1.0)
            pay = ib["settlement_value"] if side == "yes" else 1 - ib["settlement_value"]
            rows.append({"struct_id": struct, "link_id": e["link_id"], "ticker": tb, "side": side, "n": n, "t_in": e["t"],
                         "cost_usd": n * pb + fb, "settle_ts": max(ib["settlement_ts"], e["t"]),
                         "payoff_usd": n * pay, "fee_usd": fb})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def capped_book(pos: pl.DataFrame, links: pl.DataFrame, bankroll: float = BANKROLL, event_cap: float = EVENT_CAP
                ) -> pl.DataFrame:
    """Structures in time order on a $``bankroll`` book: taken only while the open cost of all legs
    (this one included) stays within the bankroll and the cluster's open cost within ``event_cap``;
    a structure's capital returns when its last leg settles."""
    import heapq

    if pos.is_empty():
        return pos
    cl = dict(zip(links["link_id"], links["cluster"].map_elements(_unit, return_dtype=pl.String), strict=True))
    s = pos.group_by("struct_id").agg(pl.col("link_id").first(), pl.col("t_in").first(), pl.col("cost_usd").sum(),
                                      pl.col("settle_ts").max()).sort("t_in", "struct_id")
    heap: list[tuple[Any, float, str]] = []
    used, per, keep = 0.0, {}, []
    for r in s.iter_rows(named=True):
        while heap and heap[0][0] <= r["t_in"]:
            _, c, u = heapq.heappop(heap)
            used -= c
            per[u] -= c
        u = cl[r["link_id"]]
        if used + r["cost_usd"] > bankroll + 1e-9 or per.get(u, 0.0) + r["cost_usd"] > event_cap + 1e-9:
            continue
        used += r["cost_usd"]
        per[u] = per.get(u, 0.0) + r["cost_usd"]
        heapq.heappush(heap, (r["settle_ts"], r["cost_usd"], u))
        keep.append(r["struct_id"])
    out = pos.filter(pl.col("struct_id").is_in(keep))
    return out.with_columns(pl.col("link_id").replace_strict(cl, return_dtype=pl.String).alias("event_ticker"))


# ---- settlement consistency (basis risk), over the whole universe ---------------------------------
def settle_check(links: pl.DataFrame | None = None) -> pl.DataFrame:
    """For every link: did ``A => B`` hold at settlement? (A paid more than B -> basis risk.)"""
    links = links if links is not None else pl.read_parquet(OUT / "links.parquet")
    m = leg_markets(links).select("ticker", "settlement_value")
    sv = dict(zip(m["ticker"], m["settlement_value"], strict=True))
    x = links.with_columns(
        pl.col("a").replace_strict(sv, default=None, return_dtype=pl.Float64).alias("va"),
        pl.col("b").list.eval(pl.element().replace_strict(sv, default=None, return_dtype=pl.Float64)).list.sum().alias("vb0"))
    x = x.with_columns(pl.when(pl.col("b_side") == "yes").then(pl.col("vb0")).otherwise(1 - pl.col("vb0")).alias("vb"))
    x = x.with_columns((pl.col("va") > pl.col("vb") + 1e-9).alias("broken"))
    x.select("link_id", "family", "kind", "a", "b", "b_side", "cluster", "detail", "va", "vb", "broken").write_parquet(
        OUT / "settle.parquet")
    out = x.group_by("family", "kind").agg(pl.len().alias("links"), pl.col("cluster").n_unique().alias("clusters"),
                                           pl.col("va").mean().alias("p_a"), pl.col("vb").mean().alias("p_b"),
                                           pl.col("broken").sum().alias("broken"),
                                           pl.col("cluster").filter(pl.col("broken")).n_unique().alias("broken_clusters")
                                           ).sort("family", "kind")
    out.write_csv(OUT / "settle_check.csv")
    x.filter(pl.col("broken")).select("family", "a", pl.col("b").list.join("+"), "b_side", "va", "vb", "detail"
                                      ).write_csv(OUT / "settle_broken.csv")
    return out


PROT_SERIES = ["KXNBA3PTPROT", "KXNBAASTPROT", "KXNBAPTSPROT", "KXNBAREBPROT", "KXNFLPASSTDSPROT", "KXNFLPASSYDSPROT",
               "KXNFLRECPROT", "KXNFLRECYDSPROT", "KXNFLRSHYDSPROT", "KXNFLTDPROT"]


def protected_basis(s) -> dict[str, Any]:
    """The market table keeps only yes/no results, so a protected prop settled at a fair price
    (player left after the first half) never forms a link. Count those divergences from the
    settled listings of every protected series and its unprotected twin (Sep 18 - Oct 31)."""
    from albert.research.pm_hazard import _api
    from albert.venues.kalshi import num

    api = _api(s, 0.5)
    rows: dict[str, tuple[str | None, float | None]] = {}
    lo, hi = CUTOFF, datetime(2026, 11, 1, tzinfo=UTC)
    for ser in PROT_SERIES + [x[:-4] for x in PROT_SERIES]:
        for page in api.series_markets(ser, "settled", min_close=lo, max_close=hi, cache=True):
            for mk in page.payload.get("markets", []):
                rows[mk["ticker"]] = (mk.get("result"), num(mk.get("settlement_value_dollars")))
    pairs, div = 0, []
    for t, (r, v) in rows.items():
        ser, rest = t.split("-", 1)
        if ser.endswith("PROT") and f"{ser[:-4]}-{rest}" in rows:
            pairs += 1
            ru, vu = rows[f"{ser[:-4]}-{rest}"]
            if (r, v) != (ru, vu):
                div.append({"protected": t, "result": r, "value": v, "unprotected_result": ru, "unprotected_value": vu})
    diff = [abs((d["value"] or 0.0) - (d["unprotected_value"] or 0.0)) for d in div]
    out = {"pairs": pairs, "settled_differently": len(div), "share": len(div) / pairs if pairs else None,
           "mean_abs_value_gap": sum(diff) / len(diff) if diff else None,
           "expected_cost_per_contract_selling_protected": sum(diff) / pairs if pairs else None,
           "examples": div[:10]}
    (OUT / "protected_basis.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ---- driver --------------------------------------------------------------------------------------
def analysis_links(links: pl.DataFrame | None = None) -> pl.DataFrame:
    """Links whose every leg has candles (cached, or fetched for the seeded sample)."""
    links = links if links is not None else pl.read_parquet(OUT / "links.parquet")
    have = cached_tickers()
    legs = pl.concat_list(pl.col("a"), pl.col("b"))
    ok = links.filter(legs.list.eval(pl.element().is_in(list(have))).list.all())
    return ok


def build_panels(links: pl.DataFrame, chunk: int = 1000) -> None:
    """Per-family (link, hour) panels under ``OUT/panel``, chunked to bound memory."""
    d = OUT / "panel"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.parquet"):
        old.unlink()
    m = leg_markets(links)
    for fam in sorted(links["family"].unique()):
        f = links.filter(pl.col("family") == fam).sort("cluster")
        for i in range(0, f.height, chunk):
            part = f.slice(i, chunk)
            tick = pl.concat([part["a"], part["b"].explode()]).unique().to_list()
            c = load_candles(tick)
            if c.is_empty():
                continue
            g = quote_grid(c, m.filter(pl.col("ticker").is_in(tick)))
            p = link_panel(part, g, m)
            if p.height:
                p.with_columns(pl.lit(fam).alias("family")).write_parquet(d / f"{fam}-{i // chunk:03d}.parquet")
        print(f"  panel {fam}: {f.height} links", flush=True)


def load_panel(families: list[str] | None = None) -> pl.DataFrame:
    parts = sorted((OUT / "panel").glob("*.parquet"))
    if families:
        parts = [p for p in parts if p.name.split("-")[0] in families]
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed") if parts else pl.DataFrame()


def gap_table(p: pl.DataFrame) -> pl.DataFrame:
    """How often a tradable gap exists: per family x split, share of quoted link-hours with a gross
    gap (bid_A > cost of B) and with a net gap (>= 0 after taker fees on both legs), and episodes."""
    ep = episodes(p).join(p.select("link_id", "family").unique(), on="link_id").with_columns(
        (pl.col("start") >= HOLDOUT_FROM).alias("holdout"))
    a = p.group_by("family", "holdout").agg(
        pl.col("link_id").n_unique().alias("links"), pl.len().alias("link_hours"),
        (pl.col("gap") > 1e-9).mean().alias("gross_share"), (pl.col("net") >= -1e-9).mean().alias("net_share"),
        pl.col("link_id").filter(pl.col("net") >= -1e-9).n_unique().alias("links_with_net_gap"),
        pl.col("gap").median().alias("median_gap"), pl.col("gap").quantile(0.99).alias("p99_gap"))
    e = ep.group_by("family", "holdout").agg(
        pl.len().alias("episodes"), pl.col("hours").median().alias("ep_median_h"), pl.col("hours").quantile(0.9).alias("ep_p90_h"),
        (pl.col("hours") >= 2).mean().alias("ep_ge2h_share"), pl.col("best").median().alias("ep_median_best_net"))
    return a.join(e, on=["family", "holdout"], how="left").sort("family", "holdout")


def pos_pnl(pos: pl.DataFrame, links: pl.DataFrame) -> pl.DataFrame:
    """One row per structure: PnL per contract (both legs, after fees, at settlement)."""
    lk = links.select("link_id", "family", "kind", pl.col("cluster").map_elements(_unit, return_dtype=pl.String).alias("unit"))
    s = pos.group_by("struct_id").agg(pl.col("link_id").first(), pl.col("t_in").first(), pl.col("n").first(),
                                      pl.col("cost_usd").sum(), pl.col("payoff_usd").sum(), pl.col("fee_usd").sum())
    s = s.join(lk, on="link_id").with_columns(
        ((pl.col("payoff_usd") - pl.col("cost_usd")) / pl.col("n")).alias("pnl"),
        (pl.col("payoff_usd") / pl.col("n") < 1 - 1e-9).alias("broke"),
        pl.when(pl.col("t_in") >= HOLDOUT_FROM).then(pl.lit("holdout")).otherwise(pl.lit("dev")).alias("split"))
    return s


STRATEGIES: dict[str, Any] = {
    "taker_snap": lambda p: taker_entries(p, 0.0),
    "taker_snap_1c": lambda p: taker_entries(p, 0.01),
    "taker_persist": lambda p: taker_entries(p, 0.0, persist=True),
    "maker_m1": lambda p: maker_entries(p, 0.01, 0.0),
    "maker_m2_r3": lambda p: maker_entries(p, 0.02, 0.03),
}


def marks_for(tickers: list[str], m: pl.DataFrame) -> pl.DataFrame:
    c = load_candles(tickers)
    if c.is_empty():
        return pl.DataFrame(schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "yes_mid": pl.Float64})
    two = (pl.col("bid") > 0) & (pl.col("ask") < 1) & (pl.col("ask") > pl.col("bid"))
    return c.filter(two).select("ticker", pl.col("t").alias("ts"), ((pl.col("bid") + pl.col("ask")) / 2).alias("yes_mid"))


def strategy_entries(parts: list[Path] | None = None) -> dict[str, pl.DataFrame]:
    """Entries of every strategy, computed panel file by panel file (bounded memory)."""
    parts = parts if parts is not None else sorted((OUT / "panel").glob("*.parquet"))
    out: dict[str, list[pl.DataFrame]] = {k: [] for k in STRATEGIES}
    for path in parts:
        p = pl.read_parquet(path)
        for name, fn in STRATEGIES.items():
            e = fn(p)
            if e.height:
                out[name].append(e)
    return {k: pl.concat(v) if v else pl.DataFrame() for k, v in out.items()}


def run_sims(entries: dict[str, pl.DataFrame], links: pl.DataFrame, m: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per strategy: per-position PnL (unconstrained, one per link) and capped-book ledgers by split."""
    from albert.research.ta_calendar import cl_mean
    from albert.research.vol_metrics import ledger_metrics

    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)
    rows, mrows, allpos = [], [], []

    def row(name: str, fam: str, split: str, g: pl.DataFrame) -> dict[str, Any]:
        mu, t, k = cl_mean(g, "pnl", "unit")
        return {"strategy": name, "family": fam, "split": split, "positions": g.height, "clusters": k,
                "pnl_per_contract": mu, "t": t, "win_share": float((g["pnl"] > 0).mean()) if g.height else None,
                "lock_floor": float(g["floor"].mean()) if g.height else None,
                "broken": int(g["broke"].sum()), "fees_per_contract": float((g["fee_usd"] / g["n"]).mean()) if g.height else None}

    for name, e in entries.items():
        if e.is_empty():
            rows.append({"strategy": name, "family": "ALL", "split": "all", "positions": 0})
            continue
        pos = positions(e, links, m)
        s = pos_pnl(pos, links).with_columns(pl.lit(name).alias("strategy"))
        s = s.join(e.select("link_id", pl.col("t").alias("t_in"), (1 - pl.col("a_price") - pl.col("b_cost")).alias("gap_in")),
                   on=["link_id", "t_in"], how="left").with_columns(
            (pl.col("gap_in") - pl.col("fee_usd") / pl.col("n")).alias("floor"))
        allpos.append(s)
        for (fam, split), g in s.group_by(["family", "split"]):
            rows.append(row(name, fam, split, g))
        for split in ("dev", "holdout", "all"):
            g = s if split == "all" else s.filter(pl.col("split") == split)
            rows.append(row(name, "ALL", split, g))
            led = capped_book(pos.filter(pl.col("struct_id").is_in(g["struct_id"].implode())), links)
            if led.is_empty():
                continue
            led = led.with_columns(pl.lit("linked").alias("strategy"), pl.lit(name).alias("variant"), pl.lit(split).alias("split"))
            led.write_parquet(OUT / "ledgers" / f"ledger_linked_{name}_{split}.parquet")
            met = ledger_metrics(led, marks_for(led["ticker"].unique().to_list(), m))
            mrows.append({"strategy": name, "split": split, **{k: met.get(k) for k in (
                "positions", "events", "pnl_usd", "usd_per_week", "ann_return_pct", "ann_vol_pct", "sharpe", "sharpe_ci90",
                "p_sharpe_pos", "sharpe_ex_top5", "usd_per_week_ex_top5", "max_dd_usd", "weeks_positive_pct",
                "first_day", "last_day")}})
    res = pl.DataFrame(rows, infer_schema_length=None)
    if allpos:
        pl.concat(allpos, how="diagonal_relaxed").write_parquet(OUT / "positions.parquet")
    return res, pl.DataFrame(mrows, infer_schema_length=None)


# ---- minute-level check of the hourly gaps --------------------------------------------------------
MINUTE_BEFORE, MINUTE_AFTER = 30, 60


def minute_plan(entries: pl.DataFrame, links: pl.DataFrame, m: pl.DataFrame, cap: int = 400) -> list[dict[str, Any]]:
    """One job per hourly gap entry: both legs' 1-minute candles in [t - 30 min, t + 60 min]
    (capped by each market's close). Seeded order; at most ``cap`` entries."""
    legs = _legs_of(links)
    info = {r["ticker"]: r for r in m.select("ticker", "event_ticker", "settlement_ts", "close_time").iter_rows(named=True)}
    e = entries.sample(fraction=1.0, shuffle=True, seed=SEED).head(cap)
    jobs = []
    for r in e.iter_rows(named=True):
        a, bs, _ = legs[r["link_id"]]
        rows = []
        for tk in [a, *bs]:
            i = info[tk]
            hi = min(r["t"] + timedelta(minutes=MINUTE_AFTER), i["close_time"])
            rows.append({"ticker": tk, "event_ticker": i["event_ticker"], "settlement_ts": i["settlement_ts"],
                         "lo": r["t"] - timedelta(minutes=MINUTE_BEFORE), "hi": hi, "period": 1})
        if all(x["settlement_ts"] >= CUTOFF for x in rows):
            jobs.append({"kind": "batch", "rows": rows, "link_id": r["link_id"], "t": r["t"]})
        else:
            jobs += [{"kind": "hist", "rows": [x], "link_id": r["link_id"], "t": r["t"]} for x in rows]
    return jobs


def fetch_minutes(s, entries: pl.DataFrame, links: pl.DataFrame, m: pl.DataFrame, cap: int = 400,
                  min_interval: float = 0.5) -> pl.DataFrame:
    from albert.research.pm_hazard import _api
    from albert.research.xcat import OHLC_SCHEMA, _run_job

    api = _api(s, min_interval, nap_cap=5.0, retries=60)
    jobs = minute_plan(entries, links, m, cap)
    print(f"minute check: {len(jobs)} requests", flush=True)
    out = []
    for i, j in enumerate(jobs):
        for tk, got in _run_job(api, j):
            for g in got or []:
                out.append({**g, "link_id": j["link_id"], "t_hour": j["t"]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(jobs)}", flush=True)
    df = pl.DataFrame(out, schema={**OHLC_SCHEMA, "link_id": pl.String, "t_hour": pl.Datetime("us", "UTC")})
    df.write_parquet(OUT / "minute_candles.parquet")
    return df


def minute_episodes(mc: pl.DataFrame, links: pl.DataFrame, m: pl.DataFrame, n: int = N_CONTRACTS) -> pl.DataFrame:
    """Per hourly entry: the net gap on the minute grid around it; how long it lasts, and the net
    gap 1 / 2 / 5 minutes after the snapshot an hourly observer acted on."""
    rows = []
    for (lid, th), g in mc.group_by(["link_id", "t_hour"]):
        lk = links.filter(pl.col("link_id") == lid)
        c = g.select("ticker", pl.col("end_ts").alias("t"), pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask"),
                     pl.lit(None, pl.Float64).alias("high"), pl.lit(None, pl.Float64).alias("low"), "volume",
                     pl.lit(0).alias("rank"), pl.lit(True).alias("ohlc"))
        if c.is_empty():
            continue
        mm = m.filter(pl.col("ticker").is_in(c["ticker"].unique().implode()))
        cc = c.join(mm.select("ticker", "close_time", "open_time"), on="ticker").filter(pl.col("t") < pl.col("close_time"))
        grid = (cc.drop("close_time", "open_time").sort("ticker", "t").with_columns(pl.lit(True).alias("real"))
                .upsample("t", every="1m", group_by="ticker", maintain_order=True))
        grid = grid.with_columns(pl.col("ticker", "bid", "ask", "rank", "ohlc").forward_fill().over("ticker"),
                                 pl.col("volume").fill_null(0.0), pl.col("real").fill_null(False))
        p = link_panel(lk, grid, mm, n).sort("t")
        if p.is_empty():
            continue
        net = dict(zip(p["t"], p["net"], strict=True))
        px = {t: (a, b) for t, a, b in zip(p["t"], p["a_px"], p["b_cost"], strict=True)}
        t1 = th + timedelta(minutes=1)
        on = [t for t, v in net.items() if v >= -1e-9]
        # run of consecutive minutes with a net gap that contains the snapshot minute (or starts after it)
        run, t = 0, th
        while net.get(t, -1) >= -1e-9:
            run += 1
            t += timedelta(minutes=1)
        before, t = 0, th - timedelta(minutes=1)
        while net.get(t, -1) >= -1e-9:
            before += 1
            t -= timedelta(minutes=1)
        rows.append({"link_id": lid, "t_hour": th, "net_at_snapshot": net.get(th), "minutes_on_window": len(on),
                     "run_from_snapshot_min": run, "run_before_snapshot_min": before,
                     **{f"net_plus{k}m": net.get(th + timedelta(minutes=k)) for k in (1, 2, 5, 10)},
                     "a_px_plus1m": px.get(t1, (None, None))[0], "b_cost_plus1m": px.get(t1, (None, None))[1]})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---- report / CLI --------------------------------------------------------------------------------
def _md(df: pl.DataFrame, digits: int = 3) -> str:
    if df.is_empty():
        return "(empty)\n"
    cols = df.columns

    def f(v: Any) -> str:
        if isinstance(v, float):
            return "" if not math.isfinite(v) else f"{v:.{digits}f}"
        if isinstance(v, list):
            return "[" + ", ".join(f(x) for x in v) + "]"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join(lines) + "\n"


def examples(links: pl.DataFrame) -> pl.DataFrame:
    """One concrete ticker pair per family (the first link of a seeded shuffle)."""
    x = links.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("family").alias("r")).filter(pl.col("r") == 0)
    return x.select("family", "kind", "a", pl.col("b").list.join("+").alias("b"), "b_side", "detail").sort("family")


def report() -> Path:
    links = pl.read_parquet(OUT / "links.parquet")
    uni = links.group_by("family", "kind").agg(pl.len().alias("links"), pl.col("cluster").n_unique().alias("clusters")).sort("family")
    parts = ["# Cross-event / cross-series linked contracts on Kalshi\n",
             ("Generated by `python -m albert.research.linked report`. Hourly candle closes; taker fees rounded up per "
              f"order at {N_CONTRACTS} contracts per leg; dev < 2026-10-01 <= holdout; t clustered by game/event.\n"),
             "## Link universe (rules text + ticker structure; settled Aug 15 - Oct 31)\n", _md(uni),
             "\n### One example per family\n", _md(examples(links)),
             "\n## Settlement consistency (basis risk) over the whole universe\n",
             _md(pl.read_csv(OUT / "settle_check.csv"))]
    for name, title in (("gaps.csv", "How often a gap exists (quoted link-hours)"),
                        ("strategies.csv", "Strategies: PnL per contract-pair at settlement"),
                        ("metrics.csv", "Capped $100 book: Sharpe / vol (vol_metrics)"),
                        ("minute_summary.csv", "Minute-level check of hourly taker entries")):
        if (OUT / name).exists():
            parts += [f"\n## {title}\n", _md(pl.read_csv(OUT / name, infer_schema_length=None))]
    if (OUT / "protected_basis.json").exists():
        pb = json.loads((OUT / "protected_basis.json").read_text())
        text = (f"{pb['settled_differently']} of {pb['pairs']} pairs ({100 * pb['share']:.1f}%) settled differently "
                f"(protected at a fair price, unprotected NO); mean value gap {pb['mean_abs_value_gap']:.2f}, i.e. "
                f"{100 * pb['expected_cost_per_contract_selling_protected']:.2f}c per contract expected cost for a lock "
                "that sells the protected leg. These markets are not in the yes/no market table, so the simulated "
                "protected trades above exclude them; subtract this cost.\n")
        parts += ["\n## Hidden basis risk: protected vs unprotected props (settled listings, Sep 18 - Oct 31)\n", text]
    path = OUT / "report.md"
    path.write_text("".join(parts))
    return path


def run_panel() -> None:
    links = analysis_links()
    links.write_parquet(OUT / "analysis_links.parquet")
    print(links.group_by("family").len().sort("family"), flush=True)
    build_panels(links)


def run_gaps() -> pl.DataFrame:
    frames = []
    for path in sorted((OUT / "panel").glob("*.parquet")):
        frames.append(pl.read_parquet(path, columns=["link_id", "family", "t", "gap", "net", "holdout"]))
    p = pl.concat(frames)
    g = gap_table(p)
    g.write_csv(OUT / "gaps.csv")
    return g


def minute_entries(me: pl.DataFrame) -> pl.DataFrame:
    """The hourly observer acts one minute after the snapshot, at that minute's quotes, only if the
    net gap is still there (``taker_minute1``)."""
    f = me.filter(pl.col("net_plus1m").is_not_null() & (pl.col("net_plus1m") >= -1e-9))
    return f.select("link_id", (pl.col("t_hour") + timedelta(minutes=1)).alias("t"),
                    pl.col("a_px_plus1m").alias("a_price"), pl.col("b_cost_plus1m").alias("b_cost"),
                    pl.lit("taker").alias("how"))


def run_strategies() -> None:
    links = pl.read_parquet(OUT / "analysis_links.parquet")
    m = leg_markets(links)
    ent = strategy_entries()
    if (OUT / "minute_episodes.parquet").exists():
        ent["taker_minute1"] = minute_entries(pl.read_parquet(OUT / "minute_episodes.parquet"))
    res, met = run_sims(ent, links, m)
    res.sort("strategy", "family", "split").write_csv(OUT / "strategies.csv")
    met.with_columns(pl.col("sharpe_ci90").cast(pl.List(pl.String)).list.join(" .. ")).write_csv(OUT / "metrics.csv")
    for k, e in ent.items():
        if e.height:
            e.write_parquet(OUT / f"entries_{k}.parquet")


def run_minutes(s, cap: int = 400) -> None:
    links = pl.read_parquet(OUT / "analysis_links.parquet")
    e = pl.read_parquet(OUT / "entries_taker_snap.parquet")
    m = leg_markets(links)
    mc = fetch_minutes(s, e, links, m, cap)
    me = minute_episodes(mc, links, m)
    me.write_parquet(OUT / "minute_episodes.parquet")
    print(minute_summary(me, links), flush=True)


def minute_summary(me: pl.DataFrame, links: pl.DataFrame) -> pl.DataFrame:
    fam = links.select("link_id", "family")
    x = me.join(fam, on="link_id").with_columns((pl.col("t_hour") >= HOLDOUT_FROM).alias("holdout"))
    summ = x.group_by("family").agg(
        pl.len().alias("entries"), (pl.col("net_at_snapshot") >= -1e-9).mean().alias("gap_at_snapshot_minute"),
        pl.col("run_from_snapshot_min").median().alias("median_run_min"),
        (pl.col("run_from_snapshot_min") >= 2).mean().alias("share_lasting_2m"),
        (pl.col("net_plus1m") >= -1e-9).mean().alias("still_gap_plus1m"),
        (pl.col("net_plus5m") >= -1e-9).mean().alias("still_gap_plus5m"),
        pl.col("net_plus1m").median().alias("median_net_plus1m")).sort("family")
    summ.write_csv(OUT / "minute_summary.csv")
    return summ


def main(argv: list[str] | None = None) -> None:
    from albert.config import settings

    ap = argparse.ArgumentParser(prog="python -m albert.research.linked")
    ap.add_argument("step", choices=["links", "plan", "fetch", "panel", "gaps", "sim", "settle", "minutes", "protbasis",
                                     "report", "all"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step in ("links", "all"):
        build_links().write_parquet(OUT / "links.parquet")
    if a.step in ("plan", "all"):
        plan()
    if a.step in ("fetch", "all"):
        fetch(settings(), workers=1)
    if a.step in ("settle", "all"):
        print(settle_check())
    if a.step in ("panel", "all"):
        run_panel()
    if a.step in ("gaps", "all"):
        print(run_gaps())
    if a.step in ("sim", "all"):
        run_strategies()
    if a.step in ("minutes", "all"):
        run_minutes(settings())
    if a.step == "all":
        run_strategies()          # again, now with the minute-delayed entries
    if a.step in ("protbasis", "all"):
        print({k: v for k, v in protected_basis(settings()).items() if k != "examples"})
    if a.step in ("report", "all"):
        print(json.dumps({"report": str(report())}))


if __name__ == "__main__":
    main()
