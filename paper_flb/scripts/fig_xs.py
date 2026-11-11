"""Figures for the cross-section workstream (xs): calibration, returns by price, term structure,
contract structure, liquidity, calendar time and limits to arbitrage.

Reads only data/research/flb_xs/ (tables/*.json written by ``python -m albert.research.flb_xs run
analyze``). Run with the paper venv:

    cd paper_flb/scripts && /home/ubuntu/.venvs/paper/bin/python fig_xs.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import FULL, HALF, MUTED, REF, SERIES, apply, save

DATA = Path(os.environ.get("FLB_XS_DIR", Path(__file__).resolve().parents[2] / "data" / "research" / "flb_xs"))
T = DATA / "tables"
GROUPS = ("Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech")
LEADS = [720, 168, 72, 24, 6, 1]
LEAD_LAB = {720: "30d", 168: "7d", 72: "3d", 24: "24h", 6: "6h", 1: "1h", -0.5: "+0.5h", -1.0: "+1h", -2.0: "+2h"}
Z = 1.96


def load(name: str):
    return json.loads((T / f"{name}.json").read_text())


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _logit_axes(ax, ticks=(0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99)):
    tv = logit(ticks)
    labels = [f"{t:g}" for t in ticks]
    ax.set_xticks(tv, labels)
    ax.set_yticks(tv, labels)
    lim = (logit(0.003), logit(0.997))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.plot(lim, lim, color=REF, ls="--", lw=0.9, zorder=1)
    ax.grid(True, axis="both")


def fig_calibration() -> None:
    """Reliability diagrams on a log-odds scale: realised frequency of the side bought against its mid
    price, by price bucket; market-weighted with 95% CIs clustered by event; volume-weighted overlay."""
    cal = load("calibration")
    panels = [g for g in ("All", *GROUPS, "Sports in-play") if g in cal]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 1.75 * nrow + 0.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, panels, strict=False):
        for k, (wkey, col, lab, mk) in enumerate((("mkt", SERIES[0], "Market-weighted", "o"),
                                                  ("vol", SERIES[1], "Volume-weighted", "s"))):
            rows = [r for r in cal[g][wkey] if r["win"] is not None and r["n"] >= 30]
            if not rows:
                continue
            q = np.array([r["q_mid"] for r in rows])
            w = np.array([r["win"] for r in rows])
            se = np.array([r["win_se"] or 0 for r in rows])
            lo, hi = np.clip(w - Z * se, 1e-4, 1), np.clip(w + Z * se, 0, 1 - 1e-4)
            x = logit(q) + (0.06 if k else -0.06)
            if k == 0:
                ax.vlines(x, logit(lo), logit(hi), color=col, lw=0.9, zorder=2)
            ax.plot(x, logit(np.clip(w, 1e-4, 1 - 1e-4)), mk, color=col, ms=3.2 if k == 0 else 2.6,
                    mfc=col if k == 0 else "white", label=lab, zorder=3)
        _logit_axes(ax)
        ax.text(0.04, 0.95, g, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.tick_params(axis="x", labelrotation=90)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("Realised frequency")
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("Side price (mid, \\$)")
    axes[0].legend(loc="lower right", fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    save(fig, "fig_xs_calibration")


def fig_returns() -> None:
    """Return per dollar by side-price bucket: gross at the mid, gross at the ask, net of taker fees."""
    cal = load("calibration")["All"]["mkt"]
    labs = [r["bucket"] for r in cal]
    x = np.arange(len(labs))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35), gridspec_kw={"width_ratios": [1.0, 1.0]})
    specs = (("r_mid", "Gross, at mid", SERIES[0], -0.2), ("r_ask", "Gross, at ask", SERIES[1], 0.0),
             ("r_net", "Net of taker fee, at ask", SERIES[2], 0.2))
    for ax, sel in zip(axes, (slice(0, 6), slice(6, None)), strict=True):
        for key, lab, col, dx in specs:
            m = np.array([r[key]["mean"] if r.get(key) else np.nan for r in cal], dtype=float)
            se = np.array([r[key]["se"] if r.get(key) and r[key]["se"] else np.nan for r in cal], dtype=float)
            xs = x[sel] + dx
            ax.errorbar(xs, 100 * m[sel], yerr=100 * Z * se[sel], fmt="o", ms=3, color=col, lw=0.9, capsize=0,
                        label=lab)
        ax.axhline(0, color=REF, lw=0.8, ls="--")
        ax.set_xticks(x[sel], [labs[i] for i in range(len(labs))[sel]], rotation=45, ha="right")
        ax.set_xlabel("Side price bucket")
    axes[0].set_ylabel("Return per \\$ (%)")
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.25)
    save(fig, "fig_xs_returns")


def _ci_plot(ax, xs, rows, key, col, lab, dx=0.0, marker="o"):
    b = np.array([r["logit"][key] if r.get("logit") and r["logit"].get(key) is not None else np.nan for r in rows])
    se = np.array([r["logit"]["se_" + key] if r.get("logit") and r["logit"].get(key) is not None else np.nan for r in rows])
    ax.errorbar(np.asarray(xs, dtype=float) + dx, b, yerr=Z * se, fmt=marker + "-", ms=3, lw=1.0, color=col,
                capsize=0, label=lab)


def fig_term() -> None:
    """Logit calibration slope by lead before the scheduled time, by group (in-play for sports)."""
    term = load("term")
    groups = [g for g in ("All", *GROUPS) if any(r["group"] == g for r in term)]
    ncol = 4
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 1.65 * nrow + 0.25), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    order = [*LEADS, -0.5, -1.0, -2.0]
    for ax, g in zip(axes, groups, strict=False):
        src = "Sports (exact start)" if g == "Sports" else g
        rows = {r["lead_h"]: r for r in term if r["group"] == src}
        pre = [ld for ld in LEADS if ld in rows]
        _ci_plot(ax, [order.index(ld) for ld in pre], [rows[ld] for ld in pre], "b", SERIES[0], "Pre-event")
        if g == "Sports":
            ex = {r["lead_h"]: r for r in term if r["group"] == "Sports (exact start)"}
            inp = [ld for ld in (-0.5, -1.0, -2.0) if ld in ex]
            if inp:
                _ci_plot(ax, [order.index(ld) for ld in inp], [ex[ld] for ld in inp], "b", SERIES[1], "In-play",
                         marker="s")
                ax.legend(loc="upper left", fontsize=6.5)
        ax.axhline(1, color=REF, ls="--", lw=0.8)
        ax.set_xticks(range(len(order)), [LEAD_LAB[o] for o in order], rotation=90)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.text(0.97, 0.06, g, transform=ax.transAxes, ha="right", fontsize=7.5)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("Logit slope $b$")
    fig.subplots_adjust(wspace=0.08, hspace=0.45)
    save(fig, "fig_xs_term")


def fig_structure() -> None:
    """(a) Logit slope by contract structure, market- and volume-weighted; (b) return per dollar on
    longshot sides (q <= 10c), gross at the mid and net of taker fees at the ask."""
    st = load("structure")
    names = [s for s in ("binary", "multi", "ladder", "deadline") if s in st]
    lab = {"binary": "Binary", "multi": "Multi-outcome", "ladder": "Ladder leg", "deadline": "Deadline"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    x = np.arange(len(names))
    _ci_plot(a, x - 0.06, [st[s]["all"]["mkt"] for s in names], "b", SERIES[0], "Market-weighted")
    _ci_plot(a, x + 0.06, [st[s]["all"]["vol"] for s in names], "b", SERIES[1], "Volume-weighted", marker="s")
    for ln in a.get_lines():
        ln.set_linestyle("none")
    a.axhline(1, color=REF, ls="--", lw=0.8)
    a.set_xticks(x, [lab[s] for s in names])
    a.set_ylabel("Logit slope $b$")
    a.legend(loc="upper left", fontsize=6.5)
    for k, (key, lbl, col) in enumerate((("r_mid", "Gross, at mid", SERIES[0]), ("r_net", "Net of taker fee, at ask", SERIES[2]))):
        reg = [st[s]["regions"]["LS <=10c"][key] for s in names]
        m = np.array([r["mean"] if r else np.nan for r in reg], dtype=float)
        se = np.array([r["se"] if r and r["se"] else np.nan for r in reg], dtype=float)
        b.errorbar(x + (k - 0.5) * 0.16, 100 * m, yerr=100 * Z * se, fmt="o", ms=3.2, lw=0.9, color=col, label=lbl)
    b.axhline(0, color=REF, ls="--", lw=0.8)
    b.set_xticks(x, [lab[s] for s in names])
    b.set_ylabel("Longshot return per \\$ (%)")
    b.legend(loc="upper right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_xs_structure")


def fig_liquidity_calendar() -> None:
    """(a) Logit slope by within-group tercile of 7-day volume, spread and open interest;
    (b) logit slope by quarter of the scheduled time (all groups pooled)."""
    liq = load("liquidity")
    cal = load("calendar")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={"width_ratios": [1.0, 1.6]})
    for k, (var, lab) in enumerate((("t_vol", "7-day volume"), ("t_spread", "Spread"), ("t_oi", "Open interest"))):
        rows = liq[var]
        _ci_plot(a, [r["tercile"] for r in rows], rows, "b", SERIES[k], lab, dx=(k - 1) * 0.08)
    a.axhline(1, color=REF, ls="--", lw=0.8)
    a.set_xticks([1, 2, 3], ["Low", "Mid", "High"])
    a.set_xlabel("Within-group tercile")
    a.set_ylabel("Logit slope $b$")
    a.legend(loc="upper left", fontsize=6.5)
    rows = [r for r in cal["cells"] if r["group"] == "All"]
    xs = np.arange(len(rows))
    _ci_plot(b, xs, rows, "b", SERIES[0], "All groups")
    b.axhline(1, color=REF, ls="--", lw=0.8)
    b.set_xticks(xs, [r["qtr"] for r in rows], rotation=90)
    b.set_ylabel("Logit slope $b$")
    b.set_xlabel("Quarter of scheduled time")
    fig.subplots_adjust(wspace=0.25)
    save(fig, "fig_xs_liq_calendar")


def fig_limits() -> None:
    """(a) Taker fee per dollar staked: exact, and rounded up per order of 1 or 100 contracts;
    (b) mispricing per contract at the mid against the cost of trading against it."""
    lim = load("limits")
    f = lim["fee_per_dollar"]
    q = np.array(f["q"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for key, lab, col, ls in (("order_1", "1 contract, rounded", SERIES[2], "-"),
                              ("order_100", "100 contracts, rounded", SERIES[1], "-"),
                              ("exact", "Exact", SERIES[0], (0, (3, 2)))):
        a.plot(100 * q, 100 * np.array(f[key]), color=col, lw=1.1, ls=ls, label=lab)
    a.set_yscale("log")
    a.set_xlabel("Price paid (cents)")
    a.set_ylabel("Fee per \\$ staked (%)")
    a.legend(loc="upper right", fontsize=6.5)
    a.grid(True, axis="both", which="major")
    band = lim["band"]
    x = np.arange(len(band))
    mis = 100 * np.array([r["mispricing_pc"] for r in band])
    se = 100 * np.array([r["mispricing_se"] or 0 for r in band])
    bt = 100 * np.array([r["band_taker"] for r in band])
    b.fill_between(x, -bt, bt, color=MUTED, alpha=0.25, lw=0, label="Taker cost band")
    b.errorbar(x, mis, yerr=Z * se, fmt="o", ms=3, color=SERIES[0], lw=0.9, label="Price $-$ realised frequency")
    b.axhline(0, color=REF, ls="--", lw=0.8)
    b.set_xticks(x, [r["bucket"] for r in band], rotation=45, ha="right")
    b.set_ylabel("Cents per contract")
    b.set_xlabel("Side price bucket")
    b.legend(loc="upper left", fontsize=6.5)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_xs_limits")


def fig_multi() -> None:
    """Multi-outcome (mutually exclusive, >= 3 legs) events: realised minus price by leg rank, for the
    raw mid, the normalised mid and the Shin-implied probability."""
    st = load("structure")["multi_outcome"]
    if "multi" not in st:
        return
    rows = st["multi"]["by_rank"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    for k, (key, lab, col) in enumerate((("e_mid", "Raw mid", SERIES[0]), ("e_norm", "Normalised", SERIES[1]),
                                         ("e_shin", "Shin", SERIES[2]))):
        m = np.array([r[key]["mean"] for r in rows])
        se = np.array([r[key]["se"] or 0 for r in rows])
        ax.errorbar(x + (k - 1) * 0.15, 100 * m, yerr=100 * Z * se, fmt="o", ms=3, lw=0.9, color=col, label=lab)
    ax.axhline(0, color=REF, ls="--", lw=0.8)
    ax.set_xticks(x, [r["rank_class"] for r in rows])
    ax.set_xlabel("Leg rank by price within event")
    ax.set_ylabel("Realised $-$ price (cents)")
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "fig_xs_multi")


def main() -> None:
    apply()
    for fn in (fig_calibration, fig_returns, fig_term, fig_structure, fig_liquidity_calendar, fig_limits, fig_multi):
        fn()
        plt.close("all")
        print(f"{fn.__name__} ok")


if __name__ == "__main__":
    main()
