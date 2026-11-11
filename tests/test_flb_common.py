import numpy as np
import polars as pl
import pytest

from albert.research import flb_common as C


def test_buckets_cover_the_price_grid():
    q = np.arange(1, 100) / 100
    labs = pl.DataFrame({"q": q}).with_columns(C.bucket_expr())["bucket"].to_list()
    assert None not in labs
    assert labs[0] == "1-2c" and labs[1] == "1-2c" and labs[2] == "3-5c" and labs[-1] == "99c"
    assert [C.bucket_of(x) for x in q] == labs
    assert len(set(labs)) == len(C.BUCKET_LABELS)


def test_side_price_returns_and_fees():
    q = C.side_price([0.07, 0.07], ["yes", "no"])
    assert q.tolist() == pytest.approx([0.07, 0.93])
    won = C.side_won([0, 0], ["yes", "no"])
    assert won.tolist() == [0, 1]
    f = C.fee_pc(q)
    assert f[0] == pytest.approx(f[1]) == pytest.approx(0.07 * 0.07 * 0.93)
    pnl, r = C.returns(won, q, f)
    assert pnl[0] == pytest.approx(-0.07 - f[0]) and r[0] == pytest.approx(-1.0)
    assert C.fee_pc(0.5, maker=True, maker_fee_series=False) == 0.0


def test_categories_and_clustered():
    assert C.coarse_category("Elections", "KXSENATE", "Senate race") == "Politics"
    assert C.coarse_category("Crypto", "KXBTC", "Bitcoin price") is None
    assert C.pm_coarse_category(["sports", "nba"]) == "Sports"
    assert C.pm_coarse_category(["crypto", "sports"]) is None
    st = C.clustered([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])
    assert st["mean"] == 2.0 and st["clusters"] == 2 and st["t"] == pytest.approx(2.0)
