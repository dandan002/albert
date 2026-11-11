"""Figures for the wallets workstream (Polymarket wallets, retail flow and venues).

Reads only data/research/flb_wallets/*.csv (written by `python -m albert.research.flb_wallets run analyze`).
Run with the paper venv: /home/ubuntu/.venvs/paper/bin/python fig_wallets.py
"""
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figstyle import FULL, HALF, MUTED, REF, SERIES, apply, save

D = Path(__file__).resolve().parents[2] / "data" / "research" / "flb_wallets"
BUCKETS = ["1-2c", "3-5c", "6-10c", "11-20c", "21-35c", "36-50c", "51-65c", "66-80c", "81-90c", "91-95c",
           "96-98c", "99c"]
CLASSES = ["Retail", "Active retail", "Mid", "Whale", "MM"]


def _ci(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    m = df["mean"].cast(pl.Float64).to_numpy() * 100
    se = df["se"].cast(pl.Float64).fill_null(np.nan).to_numpy() * 100
    return m, 1.96 * se


def _by_bucket(df: pl.DataFrame) -> pl.DataFrame:
    order = pl.DataFrame({"bucket": BUCKETS, "_i": list(range(len(BUCKETS)))})
    return order.join(df, on="bucket", how="left").sort("_i")


COARSE = ["<=10c", "11-35c", "36-65c", "66-89c", ">=90c"]
COARSE_TICKS = [r"$\leq$10", "11--35", "36--65", "66--89", r"$\geq$90"]


def _tag(ax, s: str) -> None:
    ax.text(0.0, 1.02, s, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5)


def _bucket_axis(ax, label: str = "Side price paid, $q$ (bucket)") -> None:
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([b.replace("c", "") for b in BUCKETS], rotation=45, ha="right")
    ax.set_xlabel(label + ", cents")
    ax.axhline(0, color=REF, lw=0.7, ls="--")


def fig_calib() -> None:
    cal = pl.read_csv(D / "calib_market.csv")
    flb = pl.read_csv(D / "trade_flb.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    ax = axes[0]
    x = np.arange(len(BUCKETS))
    for k, (lead, lab) in enumerate(((24, "24 h before"), (1, "1 h before"))):
        d = _by_bucket(cal.filter((pl.col("scope") == "All") & (pl.col("lead_h") == lead) & (pl.col("weighting") == "market")))
        m, e = _ci(d)
        ax.errorbar(x + (k - 0.5) * 0.22, m, yerr=e, fmt="o", ms=3, color=SERIES[k], elinewidth=0.8, capsize=0,
                    label=lab)
    _bucket_axis(ax)
    ax.set_ylabel("Return per dollar, %")
    ax.set_ylim(-100, 200)
    ax.legend(loc="lower right")
    _tag(ax, "(a) Last trade at a fixed lead, market-weighted")
    ax = axes[1]
    for k, (scope, lab) in enumerate((("Taker", "Taker buys"), ("Maker", "Maker buys"))):
        d = _by_bucket(flb.filter((pl.col("scope") == scope) & (pl.col("ret") == "ret_n") & (pl.col("weighting") == "dollar")))
        m, e = _ci(d)
        ax.errorbar(x + (k - 0.5) * 0.22, m, yerr=e, fmt="o", ms=3, color=SERIES[k], elinewidth=0.8, capsize=0, label=lab)
    _bucket_axis(ax)
    ax.set_ylim(-100, 100)
    ax.legend(loc="lower right")
    _tag(ax, "(b) Buy fills before the information time, net, dollar-weighted")
    fig.tight_layout()
    save(fig, "fig_wallets_calib")


def fig_lsshare() -> None:
    s = pl.read_csv(D / "who_share.csv").filter((pl.col("dim") == "cls") & (pl.col("half") == "All"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    x = np.arange(len(CLASSES))
    for ax, stat, tag in ((axes[0], "is_ls", r"(a) Longshots, $q\leq10$c"), (axes[1], "is_fav", r"(b) Favourites, $q\geq90$c")):
        for k, (w, lab) in enumerate((("fill", "Share of buy fills"), ("dollar", "Share of dollars bought"))):
            d = pl.DataFrame({"label": CLASSES}).join(s.filter((pl.col("stat") == stat) & (pl.col("weighting") == w)),
                                                      on="label", how="left")
            m, e = _ci(d)
            ax.bar(x + (k - 0.5) * 0.38, m, width=0.36, color=SERIES[k], label=lab, yerr=e,
                   error_kw={"elinewidth": 0.8, "ecolor": REF})
        ax.set_xticks(x)
        ax.set_xticklabels(["Retail", "Active\nretail", "Mid", "Whale", "MM"])
        _tag(ax, tag)
    axes[1].legend(loc="upper left", fontsize=6.5)
    axes[0].set_ylabel("Share, %")
    fig.tight_layout()
    save(fig, "fig_wallets_lsshare")


def fig_classret() -> None:
    cb = pl.read_csv(D / "who_class_cbucket.csv").filter((pl.col("ret") == "ret_n") & (pl.col("weighting") == "dollar"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    x = np.arange(len(COARSE))
    for ax, group, tag in ((axes[0], ["Retail", "Active retail", "Mid"], "(a) Small and mid-size wallets"),
                           (axes[1], ["Whale", "MM"], "(b) Whales and market makers")):
        for k, c in enumerate(group):
            d = pl.DataFrame({"cbucket": COARSE}).join(cb.filter(pl.col("cls") == c), on="cbucket", how="left")
            m, e = _ci(d)
            off = (k - (len(group) - 1) / 2) * 0.2
            ax.errorbar(x + off, m, yerr=e, fmt="o", ms=3.5, color=SERIES[k], elinewidth=0.9, capsize=0, label=c)
        ax.axhline(0, color=REF, lw=0.7, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(COARSE_TICKS)
        ax.set_xlabel("Side price paid, $q$, cents")
        ax.set_ylim(-100, 60)
        ax.legend(loc="lower right")
        _tag(ax, tag)
    axes[0].set_ylabel("Net return per dollar, %")
    fig.tight_layout()
    save(fig, "fig_wallets_classret")


def fig_lorenz() -> None:
    lz = pl.read_csv(D / "lorenz.csv").filter(pl.col("kind") == "top_share")
    fig, ax = plt.subplots(figsize=(HALF, 2.6))
    for k, (s, lab) in enumerate((("gains", "Gains (winning wallets)"), ("losses", "Losses (losing wallets)"),
                                  ("traded", "Dollar volume (all wallets)"))):
        d = lz.filter(pl.col("series") == s).sort("x")
        ax.plot(d["x"] * 100, d["y"] * 100, color=SERIES[k], label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("Top share of wallets, % (log scale)")
    ax.set_ylabel("Share of the total, %")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_wallets_lorenz")


def fig_persist() -> None:
    p = pl.read_csv(D / "persistence.csv")
    import json
    summ = json.loads((D / "summary.json").read_text())["persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    a = p.filter(pl.col("series") == "actual").sort("quintile")
    ax = axes[0]
    ax.bar(np.arange(5), a["roi_b_dollar"] * 100, color=SERIES[0], width=0.6,
           yerr=1.96 * a["roi_b_se"].fill_null(0).to_numpy() * 100, error_kw={"elinewidth": 0.8, "ecolor": REF})
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Q1\n(worst)", "Q2", "Q3", "Q4", "Q5\n(best)"])
    ax.set_xlabel("Quintile of first-half shrunk ROI")
    ax.set_ylabel("Second-half PnL per dollar bought, %")
    _tag(ax, "(a) Second-half return by first-half quintile")
    ax = axes[1]
    plc = p.filter(pl.col("series") == "placebo")["roi_b_dollar"].to_numpy()
    ax.hist(plc, bins=30, color=SERIES[1], alpha=0.85, label="Luck placebo")
    act = summ.get("actual_spearman_same_wallets")
    if act is not None:
        ax.axvline(act, color=SERIES[0], lw=1.5, label="Actual")
    ax.set_xlabel("Spearman correlation, first vs second half")
    ax.set_ylabel("Placebo draws")
    ax.legend(loc="upper left")
    _tag(ax, "(b) Rank persistence: actual vs luck placebo")
    fig.tight_layout()
    save(fig, "fig_wallets_persist")


def fig_inflow() -> None:
    w = pl.read_csv(D / "inflow_weekly.csv", try_parse_dates=True)
    ev = pl.read_csv(D / "inflow_events.csv")
    nm = pl.read_csv(D / "newwallet_monthly.csv")
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.9), sharex=True)
    ax = axes[0]
    wk = [datetime.fromisoformat(str(t)[:19]) for t in w["week"].to_list()]
    ax.plot(wk, w["len"], color=SERIES[0], lw=1.0)
    short = {"US election 2024": "US election", "Super Bowl LIX": "SB LIX", "Papal conclave 2025": "Conclave",
             "NYC mayor 2025": "NYC mayor", "Super Bowl LX": "SB LX"}
    for k, r in enumerate(ev.iter_rows(named=True)):
        d = datetime.fromisoformat(r["date"])
        ax.axvline(d, color=REF, ls="--", lw=0.7)
        ax.text(d, 1.0 + 0.09 * (k % 2), short.get(r["event"], r["event"]), transform=ax.get_xaxis_transform(),
                va="bottom", ha="center", fontsize=6, color=MUTED)
    ax.set_ylabel("Accounts created per week\n(sampled active wallets)")
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5)
    ax = axes[1]
    mm = [datetime.fromisoformat(m + "-15") for m in nm["month"].to_list()]
    ax.plot(mm, nm["ls_share_new"] * 100, color=SERIES[0], marker="o", ms=2.5, label="Accounts < 30 days old")
    ax.plot(mm, nm["ls_share_old"] * 100, color=SERIES[1], marker="o", ms=2.5, label="Older accounts")
    ax.set_ylabel("Longshot share of\nbuy fills, %")
    ax.legend(loc="upper left")
    _tag(ax, "(b)")
    fig.tight_layout()
    save(fig, "fig_wallets_inflow")


def fig_xvenue() -> None:
    g = pl.read_csv(D / "xvenue_gap.csv").filter(pl.col("lead_h") == 1)
    d = _by_bucket(g).filter(pl.col("n").fill_null(0) >= 5)
    idx = [BUCKETS.index(b) for b in d["bucket"].to_list()]
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    m, e = _ci(d)
    ax.errorbar(idx, m, yerr=e, fmt="o", ms=3.5, color=SERIES[0], elinewidth=0.8, capsize=0,
                label="Polymarket minus Kalshi mid")
    fee = d["mean_fee_sum"].to_numpy() * 100
    ax.plot(idx, fee, color=REF, ls="--", lw=0.8, label="Sum of taker fees")
    ax.plot(idx, -fee, color=REF, ls="--", lw=0.8)
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([b.replace("c", "") for b in BUCKETS], rotation=45, ha="right")
    ax.set_xlabel("Average side price of the two venues, cents")
    ax.set_ylabel("Price gap, cents")
    ax.legend(loc="lower center", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_wallets_xvenue")


if __name__ == "__main__":
    apply()
    for f in (fig_calib, fig_lsshare, fig_classret, fig_lorenz, fig_persist, fig_inflow, fig_xvenue):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as err:      # an input of this figure has not been produced yet
            print("skipped", f.__name__, err)
