from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from albert.research import vol_metrics as M


def _ts(d: int, h: int = 12) -> datetime:
    return datetime(2026, 9, d, h, tzinfo=UTC)


def _ledger(rows: list[dict]) -> pl.DataFrame:
    base = {"struct_id": "1", "event_ticker": "EV", "ticker": "T1", "side": "yes", "n": 10}
    return pl.DataFrame([{**base, **r} for r in rows])


MARKS = pl.DataFrame({"ticker": ["T1", "T1", "T1"], "ts": [_ts(1, 6), _ts(1, 20), _ts(2, 20)],
                      "yes_mid": [0.30, 0.50, 0.60]})


def test_mtm_yes_leg_sums_to_realized_and_ignores_pre_fill_marks():
    led = _ledger([{"t_in": _ts(1), "cost_usd": 4.1, "settle_ts": _ts(3), "payoff_usd": 10.0}])
    v = M.mtm_values(led, MARKS).sort("day")
    # day 1: 10 x 0.50 - 4.10; day 2: 10 x (0.60 - 0.50); day 3: settles at 10
    assert v["pnl"].to_list() == pytest.approx([0.9, 1.0, 4.0])
    assert v["marked"].all()
    x = M.mtm_daily(led, MARKS)
    assert x.sum() == pytest.approx(10.0 - 4.1)
    assert M.settle_daily(led).tolist() == pytest.approx([0.0, 0.0, 5.9])


def test_mtm_no_leg_uses_complement_and_entry_price_without_a_mark():
    led = _ledger([{"side": "no", "t_in": _ts(2, 21), "cost_usd": 4.0, "settle_ts": _ts(4), "payoff_usd": 0.0}])
    v = M.mtm_values(led, MARKS).sort("day")
    # the only marks predate the 21:00 fill -> held at the entry price until settlement
    assert v["value"].to_list() == pytest.approx([4.0, 4.0, 0.0])
    assert v["marked"].to_list() == [False, False, True]
    assert v["pnl"].sum() == pytest.approx(-4.0)


def test_series_stats_basics():
    rng = np.random.default_rng(0)
    x = 0.2 + rng.normal(0, 1, 70)
    st = M.series_stats(x)
    assert st["sharpe"] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(365), abs=0.01)
    assert st["ann_vol_pct"] == pytest.approx(x.std(ddof=1) * np.sqrt(365), abs=0.1)
    lo, hi = st["sharpe_ci90"]
    assert lo < st["sharpe"] < hi
    assert M.max_drawdown(np.array([1.0, -3.0, 1.0])) == -3.0
    assert M.sharpe(np.zeros(10)) is None
    assert M.weekly(np.ones(15)).tolist() == [7.0, 7.0]


def test_ledger_metrics_concentration_and_basis():
    led = pl.concat([
        _ledger([{"t_in": _ts(1), "cost_usd": 4.1, "settle_ts": _ts(3), "payoff_usd": 10.0}]),
        _ledger([{"struct_id": "2", "event_ticker": "EV2", "ticker": "T2", "t_in": _ts(5), "cost_usd": 5.0,
                  "settle_ts": _ts(20), "payoff_usd": 0.0}]),
    ])
    m = M.ledger_metrics(led, MARKS)
    assert m["pnl_usd"] == pytest.approx(10.0 - 4.1 - 5.0)
    assert m["positions"] == 2 and m["events"] == 2
    assert m["days"] == 20
