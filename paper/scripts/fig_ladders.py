"""Figures for the strike-ladder (smile) study. Reads the saved outputs of
``albert.research.pm_smile`` under data/research/pm_smile/ and writes paper/figures/fig_lad_*.pdf.

Run with the paper venv:  cd paper/scripts && /home/ubuntu/.venvs/paper/bin/python fig_ladders.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from figstyle import FULL, MUTED, REF, SERIES, apply, save

DATA = Path(__file__).resolve().parents[2] / "data" / "research" / "pm_smile"
NAMES = {"wx_high": "Weather highs", "wx_low": "Weather lows", "sp_total": "Sports totals",
         "sp_teamtotal": "Team totals", "sp_spread": "Spreads", "politics": "Politics",
         "entertainment": "Entertainment", "scitech": "Science/tech", "climate_other": "Other climate"}


def load() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    sn = pl.read_parquet(DATA / "snapshots.parquet")
    pt = pl.read_parquet(DATA / "points.parquet")
    summary = json.loads((DATA / "summary.json").read_text())
    return sn, pt, summary


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int = 30, stat=np.median):
    idx = np.digitize(x, edges) - 1
    xs, ys = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_n:
            xs.append(float(np.mean(x[sel])))
            ys.append(float(stat(y[sel])))
    return np.array(xs), np.array(ys)


def fig_smile(sn: pl.DataFrame, pt: pl.DataFrame) -> None:
    """sigma_K / sigma_ATM against moneyness: binned medians and the pooled quadratic fit."""
    p = pt.join(sn.select("snap_id", "group"), on="snap_id").filter(
        pl.col("v").is_not_null() & ~pl.col("atm_pair") & (pl.col("m").abs() <= 3.5))
    groups = ["wx_high", "sp_total", "sp_teamtotal", "sp_spread"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.2), sharex=True)
    edges = np.arange(-3.5, 3.51, 0.25)
    grid = np.linspace(-3.2, 3.2, 200)
    for k, (ax, grp) in enumerate(zip(axes.ravel(), groups)):
        g = p.filter(pl.col("group") == grp)
        m, v = g["m"].to_numpy(), g["v"].to_numpy()
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(m), m, m**2]), v - 1, rcond=None)
        bx, by = binned(m, v, edges)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
        ax.plot(bx, by, "o", color=SERIES[0], ms=3.2, label="Binned median")
        ax.plot(grid, 1 + coef[0] + coef[1] * grid + coef[2] * grid**2, color=REF, ls="--", lw=1.1,
                label="Quadratic fit")
        ax.text(0.03, 0.92, NAMES[grp], transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylim(0.75, 1.9)
        if k >= 2:
            ax.set_xlabel(r"Moneyness $m=(K-F)/\sigma_{\mathrm{ATM}}$")
        if k % 2 == 0:
            ax.set_ylabel(r"$\sigma_K/\sigma_{\mathrm{ATM}}$")
        if k == 0:
            ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_lad_smile")
    plt.close(fig)


def _pairs(sn: pl.DataFrame) -> pl.DataFrame:
    """Consecutive snapshots of a ladder, raw coordinates (as pm_smile.snapshot_pairs)."""
    x = sn.with_columns((pl.col("F") * pl.col("orient")).alias("Fr"), (pl.col("s") * pl.col("orient")).alias("sr"))
    x = x.sort("ladder", "lead_h", descending=[False, True])
    nxt = {c: pl.col(c).shift(-1).over("ladder") for c in ("lead_h", "Fr", "sigma_atm", "snap_id")}
    x = x.with_columns(**{f"{c}_u": e for c, e in nxt.items()}).filter(pl.col("lead_h_u").is_not_null())
    return x.with_columns((pl.col("group") + "|" + pl.col("lead_h").cast(pl.String) + "->"
                           + pl.col("lead_h_u").cast(pl.String)).alias("fe"),
                          ((pl.col("Fr_u") - pl.col("Fr")) / pl.col("sigma_atm")).alias("dF_n"),
                          ((pl.col("sigma_atm_u") - pl.col("sigma_atm")) / pl.col("sigma_atm")).alias("dsig_n"))


def fig_sticky(sn: pl.DataFrame, pt: pl.DataFrame, summary: dict) -> None:
    """Fixed-strike vol change against the sticky-moneyness prediction (fixed effects per lead
    pair and the level term partialled out), binned."""
    smile = pl.DataFrame(summary["smile"]).filter(pl.col("split") == "both").select(
        "group", pl.col("lead_h").cast(pl.Int64), pl.col("s").alias("s_pool"), pl.col("c").alias("c_pool"))
    pr = _pairs(sn).filter(pl.col("dF_n").abs() < 5).with_columns(pl.col("lead_h").cast(pl.Int64))
    pr = pr.join(smile, on=["group", "lead_h"], how="left").with_columns((pl.col("s_pool") * pl.col("orient")).alias("sr_pool"))
    pr = pr.filter(pl.col("sr").is_not_nan() & pl.col("c").is_not_nan() & pl.col("sr_pool").is_not_null())
    p = (pt.select("snap_id", "K", "m", "sigma_K", "v").filter(pl.col("sigma_K").is_not_null())
         .join(sn.select("snap_id", "orient"), on="snap_id")
         .with_columns((pl.col("K") * pl.col("orient")).alias("Kr"), (pl.col("m") * pl.col("orient")).alias("mr")))
    j = (pr.select("snap_id", "snap_id_u", "group", "fe", "sr_pool", "c_pool", "dF_n", "dsig_n", "sigma_atm")
         .join(p.select("snap_id", "Kr", "mr", "sigma_K", "v"), on="snap_id")
         .join(p.select(pl.col("snap_id").alias("snap_id_u"), "Kr", pl.col("sigma_K").alias("sigma_K_u")), on=["snap_id_u", "Kr"]))
    j = j.with_columns(((pl.col("sigma_K_u") - pl.col("sigma_K")) / pl.col("sigma_atm")).alias("y"),
                       (-(pl.col("sr_pool") + 2 * pl.col("c_pool") * pl.col("mr")) * pl.col("dF_n")).alias("x"),
                       (pl.col("v") * pl.col("dsig_n")).alias("z")).drop_nulls(["y", "x", "z"])
    j = j.with_columns([(pl.col(c) - pl.col(c).mean().over("fe")).alias(c) for c in ("y", "x", "z")])
    groups = ["sp_total", "sp_spread", "wx_high"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    for k, (ax, grp) in enumerate(zip(axes, groups)):
        g = j.filter(pl.col("group") == grp)
        y, x, z = (g[c].to_numpy() for c in ("y", "x", "z"))
        # Frisch-Waugh: partial the level term out of both sides
        ry = y - z * (z @ y) / (z @ z)
        rx = x - z * (z @ x) / (z @ z)
        qs = np.quantile(rx, np.linspace(0.01, 0.99, 21))
        bx, by = binned(rx, ry, qs, min_n=50, stat=np.mean)
        lim = 1.15 * float(np.max(np.abs(bx)))
        xx = np.array([-lim, lim])
        ax.set_xlim(-lim, lim)
        ax.plot(bx, by, "o", color=SERIES[0], ms=3.2, label="Binned mean")
        ax.plot(xx, xx, color=REF, ls="--", lw=1.0, label="Sticky moneyness")
        ax.plot(xx, 0 * xx, color=REF, ls=":", lw=1.0, label="Sticky strike")
        ax.text(0.03, 0.95, NAMES[grp], transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlabel(r"Predicted $-\sigma'(m)\,\Delta F/\sigma_{\mathrm{ATM}}$")
        if k == 0:
            ax.set_ylabel(r"$\Delta\sigma_K/\sigma_{\mathrm{ATM}}$ (residual)")
            ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_lad_sticky")
    plt.close(fig)


def fig_term(sn: pl.DataFrame) -> None:
    """Within-ladder demeaned log sigma_ATM by lead, with a Brownian (slope 1/2) reference."""
    x = sn.filter((pl.col("sigma_atm") > 0) & ~(pl.col("group").str.starts_with("sp_") & (pl.col("lead_h") == 1)))
    x = x.with_columns(pl.col("sigma_atm").log().alias("ls"), pl.col("lead_h").cast(pl.Float64).log().alias("ll"))
    x = x.filter(pl.col("ladder").count().over("ladder") >= 2)
    x = x.with_columns((pl.col("ls") - pl.col("ls").mean().over("ladder")).alias("dls"),
                       (pl.col("ll") - pl.col("ll").mean().over("ladder")).alias("dll"))
    panels = [["wx_high", "wx_low"], ["politics", "sp_total"]]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    for k, (ax, groups) in enumerate(zip(axes, panels)):
        for i, grp in enumerate(groups):
            g = x.filter(pl.col("group") == grp).group_by("lead_h").agg(pl.col("dls").mean(), pl.len()).filter(
                pl.col("len") >= 20).sort("lead_h")
            ax.plot(g["lead_h"].to_numpy(), g["dls"].to_numpy(), "o-", color=SERIES[i], ms=3.5, lw=1.0, label=NAMES[grp])
        leads = np.array(sorted(x.filter(pl.col("group").is_in(groups))["lead_h"].unique().to_list()), float)
        ref = 0.5 * (np.log(leads) - np.log(leads).mean())
        ax.plot(leads, ref, color=REF, ls="--", lw=1.0, label="Slope 1/2")
        ax.set_xscale("log")
        ax.set_xticks(leads)
        ax.set_xticklabels([f"{int(v)}" for v in leads])
        ax.minorticks_off()
        ax.set_xlabel("Lead before expiry (h, log scale)")
        if k == 0:
            ax.set_ylabel(r"$\log\sigma_{\mathrm{ATM}}$, demeaned within ladder")
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_lad_termstructure")
    plt.close(fig)


def fig_calibration(summary: dict) -> None:
    """Fractional PIT histograms (holdout) and implied vs realized tail mass beyond |m|."""
    pit = pl.DataFrame(summary["pit"])
    tails = pl.DataFrame(summary["tail_summary"])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    for ax, grp in zip(axes[:2], ["sp_total", "wx_high"]):
        r = pit.filter((pl.col("group") == grp) & (pl.col("split") == "holdout")).row(0, named=True)
        h = [r[f"pit{j}"] for j in range(10)]
        ax.bar(np.arange(10) / 10 + 0.05, h, width=0.09, color=SERIES[0])
        ax.axhline(0.1, color=REF, ls="--", lw=1.0)
        ax.set_ylim(0, 0.14)
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.text(0.03, 0.97, NAMES[grp], transform=ax.transAxes, fontsize=8, va="top")
    axes[0].set_ylabel("Share of outcomes")
    ax = axes[2]
    fam = {"weather": ["wx_high", "wx_low"], "sports": ["sp_total", "sp_teamtotal", "sp_spread"],
           "other": ["politics", "entertainment", "scitech", "climate_other"]}
    labels = {"weather": "Weather", "sports": "Sports", "other": "Politics/culture/science"}
    top = 0.0
    for i, (k, gs) in enumerate(fam.items()):
        t = tails.filter(pl.col("group").is_in(gs) & (pl.col("m_ge") >= 1.5))
        ax.plot(t["implied_c"].to_numpy(), t["realized_c"].to_numpy(), "o", color=SERIES[i], ms=3, alpha=0.85,
                label=labels[k])
        top = max(top, float(t["implied_c"].max()), float(t["realized_c"].max()))
    ax.plot([0, top], [0, top], color=REF, ls="--", lw=1.0)
    ax.set_xlabel("Implied tail mass (cents)")
    ax.set_ylabel("Realized frequency (cents)")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_lad_calibration")
    plt.close(fig)


def _ci(s: str | None) -> tuple[float, float]:
    a, b = (float(v) for v in re.findall(r"-?\d+\.\d+", s or "nan nan")) if s else (np.nan, np.nan)
    return a, b


def fig_variance(summary: dict) -> None:
    """Realized / implied variance ratio between consecutive snapshots with 90% bootstrap CIs."""
    vb = pl.DataFrame(summary["variance_between"])
    panels = [("wx_high", ["36->24", "24->16", "16->12", "12->8"], "Weather highs: window (h before close)"),
              (None, [("sp_total", "6->1"), ("sp_teamtotal", "6->1"), ("sp_spread", "6->1")],
               "Sports, in play: 6 h $\\to$ 1 h before expiry")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    for k, (ax, (grp, items, xlabel)) in enumerate(zip(axes, panels)):
        xs, ys, lo, hi, ticks = [], [], [], [], []
        for i, it in enumerate(items):
            g, pair = (grp, it) if grp else it
            r = vb.filter((pl.col("group") == g) & (pl.col("pair") == pair)).row(0, named=True)
            a, b = _ci(r["ci90"])
            xs.append(i), ys.append(r["rv_over_iv"]), lo.append(r["rv_over_iv"] - a), hi.append(b - r["rv_over_iv"])
            ticks.append(pair.replace("->", r"$\to$") if grp else NAMES[g])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=SERIES[0], ms=4, capsize=2, lw=1.0)
        ax.axhline(1.0, color=REF, ls="--", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel("Realized / implied variance")
    fig.tight_layout()
    save(fig, "fig_lad_variance")
    plt.close(fig)


def main() -> None:
    apply()
    sn, pt, summary = load()
    fig_smile(sn, pt)
    fig_sticky(sn, pt, summary)
    fig_term(sn)
    fig_calibration(summary)
    fig_variance(summary)


if __name__ == "__main__":
    main()
