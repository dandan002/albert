"""Figures for the trade-level flow workstream (``flow``): who pays the favourite-longshot bias.

Reads only data/research/flb_flow/ (tables/*.csv and summary.json written by
``python -m albert.research.flb_flow run analyze``). Run with the paper venv:

    /home/ubuntu/.venvs/paper/bin/python paper_flb/scripts/fig_flow.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from figstyle import FULL, HALF, REF, SERIES, apply, save

DATA = Path(__file__).resolve().parents[2] / "data" / "research" / "flb_flow"
TAB = DATA / "tables"
BUCKETS = ["1-2c", "3-5c", "6-10c", "11-20c", "21-35c", "36-50c", "51-65c", "66-80c", "81-90c",
           "91-95c", "96-98c", "99c"]
SIZES = ["<=$10", "$10-100", "$100-1k", ">$1k"]
SIZE_TXT = [r"$\leq\$10$", r"\$10–100", r"\$100–1k", r"$>\$1$k"]
Z = 1.96


def tab(name: str) -> pl.DataFrame:
    return pl.read_csv(TAB / f"{name}.csv")


def _ci(ax, x, m, se, color, label, marker="o", ls="-", offset=0.0):
    x = np.asarray(x, dtype=float) + offset
    m = np.asarray(m, dtype=float)
    se = np.nan_to_num(np.asarray(se, dtype=float), nan=0.0)
    ax.errorbar(x, m, yerr=Z * se, color=color, marker=marker, ls=ls, lw=1.1, ms=3.5, capsize=1.5,
                elinewidth=0.7, label=label)


MIN_CLUSTERS = 5   # a stratum's cell enters the cross-stratum median only with >= 5 events
MIN_STRATA = 6     # and a median is shown only when at least 6 of the 18 strata enter it


def strata_median(df: pl.DataFrame, by: list[str], col: str = "d_net") -> pl.DataFrame:
    """Median and interquartile range across strata of a per-stratum cell estimate."""
    ok = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite()
                   & (pl.col(f"{col}_clusters") >= MIN_CLUSTERS))
    out = ok.group_by(by).agg(pl.col(col).median().alias("med"), pl.col(col).quantile(0.25).alias("q25"),
                              pl.col(col).quantile(0.75).alias("q75"), pl.len().alias("k"))
    return out.filter(pl.col("k") >= MIN_STRATA)


def _band(ax, x, med, lo, hi, color, label, marker="o", offset=0.0):
    x = np.asarray(x, dtype=float) + offset
    med, lo, hi = (np.asarray(v, dtype=float) for v in (med, lo, hi))
    ax.errorbar(x, med, yerr=[med - lo, hi - med], color=color, marker=marker, lw=1.1, ms=3.5, capsize=1.5,
                elinewidth=0.7, label=label)


def fig_roles() -> None:
    """(a) Net return per dollar by own-side price bucket, taker against maker; (b) taker share of
    the contracts bought in each bucket."""
    t = tab("roles_bucket").filter(pl.col("group") == "All")
    md = strata_median(tab("roles_bucket_strata"), ["role", "bucket"])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    ax = axes[0]
    pos = np.arange(len(BUCKETS))
    for i, (role, lab) in enumerate((("taker", "Taker"), ("maker", "Maker"))):
        r = md.filter(pl.col("role") == role).join(pl.DataFrame({"bucket": BUCKETS, "i": pos}), on="bucket").sort("i")
        _band(ax, r["i"], 100 * r["med"], 100 * r["q25"], 100 * r["q75"], SERIES[i], lab, offset=(i - 0.5) * 0.2)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(pos, BUCKETS, rotation=60, ha="right")
    ax.set_xlabel("Price paid for the side held, $q$")
    ax.set_ylabel("Net return per dollar (%), median stratum")
    ax.set_ylim(-100, 60)
    ax.legend(loc="lower right")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top")
    ax = axes[1]
    w = t.pivot(on="role", index="bucket", values="w_contracts")
    w = w.join(pl.DataFrame({"bucket": BUCKETS, "i": pos}), on="bucket").sort("i")
    share = w["taker"] / (w["taker"] + w["maker"])
    ax.bar(w["i"], 100 * share, color=SERIES[0], width=0.7)
    ax.axhline(50, color=REF, lw=0.7, ls="--")
    ax.set_xticks(pos, BUCKETS, rotation=60, ha="right")
    ax.set_xlabel("Price paid for the side held, $q$")
    ax.set_ylabel("Contracts bought by takers (%)")
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top")
    fig.tight_layout()
    save(fig, "fig_flow_roles")


def fig_roles_groups() -> None:
    """Taker against maker net return per dollar by price bucket, one panel per group."""
    t = tab("roles_bucket")
    groups = ["Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.2), sharex=True, sharey=True)
    pos = np.arange(len(BUCKETS))
    for k, g in enumerate(groups):
        ax = axes.flat[k]
        for i, (role, lab) in enumerate((("taker", "Taker"), ("maker", "Maker"))):
            r = (t.filter((pl.col("group") == g) & (pl.col("role") == role))
                 .join(pl.DataFrame({"bucket": BUCKETS, "i": pos}), on="bucket").sort("i"))
            _ci(ax, r["i"], 100 * r["d_net"], 100 * r["d_net_se"], SERIES[i], lab, offset=(i - 0.5) * 0.18)
        ax.axhline(0, color=REF, lw=0.7, ls="--")
        ax.set_title(g)
        ax.set_ylim(-100, 80)
        ax.set_xticks(pos[::2], BUCKETS[::2], rotation=60, ha="right")
        if k % 3 == 0:
            ax.set_ylabel("Net return per dollar (%)")
        if k == 0:
            ax.legend(loc="lower right")
    fig.supxlabel("Price paid for the side held, $q$", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_flow_roles_groups")


def fig_size() -> None:
    """(a) Longshot share of taker volume by order size; (b) taker net return per dollar on
    longshots and favourites by order size."""
    sh = tab("size_shares").filter(pl.col("group") == "All")
    sh = sh.join(pl.DataFrame({"size_bucket": SIZES, "i": range(4)}), on="size_bucket").sort("i")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    ax = axes[0]
    x = np.arange(4)
    xi = sh["i"].to_numpy()
    ax.bar(xi - 0.18, 100 * sh["ls_contracts"], width=0.36, color=SERIES[0], label="Contracts")
    ax.bar(xi + 0.18, 100 * sh["ls_orders"], width=0.36, color=SERIES[1], label="Orders")
    ax.set_xticks(x, SIZE_TXT)
    ax.set_xlabel("Taker order size (dollar notional)")
    ax.set_ylabel(r"Share at $q \leq 10$c (%)")
    ax.legend(loc="upper right")
    ax.text(-0.02, 1.02, "(a)", transform=ax.transAxes, va="bottom")
    ax = axes[1]
    md = strata_median(tab("size_side_strata"), ["side", "size_bucket"])
    for i, (side, lab) in enumerate((("longshot", r"Longshots ($q \leq 10$c)"),
                                     ("favourite", r"Favourites ($q \geq 90$c)"))):
        r = md.filter(pl.col("side") == side).join(
            pl.DataFrame({"size_bucket": SIZES, "i": range(4)}), on="size_bucket").sort("i")
        _band(ax, r["i"], 100 * r["med"], 100 * r["q25"], 100 * r["q75"], SERIES[i], lab, offset=(i - 0.5) * 0.12)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(x, SIZE_TXT)
    ax.set_xlabel("Taker order size (dollar notional)")
    ax.set_ylabel("Taker net return per dollar (%), median stratum")
    ax.legend(loc="center right")
    ax.text(-0.02, 1.02, "(b)", transform=ax.transAxes, va="bottom")
    fig.tight_layout()
    save(fig, "fig_flow_size")


def fig_size_buckets() -> None:
    """Taker net return per dollar by price bucket, small multiples by order size."""
    t = strata_median(tab("size_returns_strata"), ["size_bucket", "bucket"])
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2), sharey=True)
    pos = np.arange(len(BUCKETS))
    for k, (sb, lab) in enumerate(zip(SIZES, SIZE_TXT, strict=True)):
        ax = axes[k]
        r = t.filter(pl.col("size_bucket") == sb).join(pl.DataFrame({"bucket": BUCKETS, "i": pos}), on="bucket").sort("i")
        _band(ax, r["i"], 100 * r["med"], 100 * r["q25"], 100 * r["q75"], SERIES[0], None)
        ax.axhline(0, color=REF, lw=0.7, ls="--")
        ax.set_xticks(pos[::2], BUCKETS[::2], rotation=60, ha="right")
        ax.set_title(lab)
        ax.set_ylim(-100, 100)
        if k == 0:
            ax.set_ylabel("Net return per dollar (%),\nmedian stratum")
    fig.supxlabel("Taker price $q$", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_flow_size_buckets")


def fig_yesno() -> None:
    t = strata_median(tab("yesno_strata"), ["taker_yes", "bucket"])
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    pos = np.arange(len(BUCKETS))
    for i, (flag, lab) in enumerate(((True, "Taker buys YES"), (False, "Taker buys NO"))):
        r = (t.filter(pl.col("taker_yes") == flag).join(pl.DataFrame({"bucket": BUCKETS, "i": pos}), on="bucket")
             .sort("i"))
        _band(ax, r["i"], 100 * r["med"], 100 * r["q25"], 100 * r["q75"], SERIES[i], lab, offset=(i - 0.5) * 0.2)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(pos, BUCKETS, rotation=60, ha="right")
    ax.set_xlabel("Taker price $q$")
    ax.set_ylabel("Net return per dollar (%),\nmedian stratum")
    ax.set_ylim(-100, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_flow_yesno")


CONTRASTS = [("evening_18_24", "Evening (18–24 ET)"), ("weekend", "Weekend"), ("inplay", "In play"),
             ("primetime", "Primetime game"), ("marquee", "Marquee event"), ("first24h", "First 24 h listed"),
             ("surge_3x", r"Volume surge ($\geq 3\times$)"), ("last_hour", "Last hour before expiry")]


def _contrast(ax, c, sides, scale=100.0):
    y = np.arange(len(CONTRASTS))
    for i, (side, lab) in enumerate(sides):
        vals, ses = [], []
        for key, _ in CONTRASTS:
            r = c.filter((pl.col("contrast") == key) & (pl.col("side") == side))
            ok = r.height and r["diff"][0] is not None
            vals.append(scale * r["diff"][0] if ok else np.nan)
            ses.append(scale * r["se"][0] if ok and r["se"][0] is not None else np.nan)
        ax.errorbar(vals, y + (i - 0.5) * 0.3, xerr=Z * np.nan_to_num(np.array(ses)), color=SERIES[i], marker="o",
                    ls="none", ms=3.5, capsize=1.5, elinewidth=0.7, label=lab)
    ax.axvline(0, color=REF, lw=0.7, ls="--")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    return y


def fig_timing() -> None:
    """(a) Small-order and longshot shares of taker flow by hour of day (ET); (b) attention cell
    minus the rest: longshot share of taker contracts and small-order share of taker orders;
    (c) attention cell minus the rest: taker net return per dollar, all positions."""
    t = tab("timing").filter((pl.col("dim") == "hour_et") & (pl.col("sector") == "All"))
    t = t.with_columns(pl.col("cell").cast(pl.Int64)).sort("cell")
    fig = plt.figure(figsize=(FULL, 2.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1, 0.65])
    ax = fig.add_subplot(gs[0])
    ax.plot(t["cell"], 100 * t["small_order_share"], color=SERIES[0], marker="o", label=r"Orders $\leq$ \$100")
    ax.plot(t["cell"], 100 * t["ls_share"], color=SERIES[1], marker="s", label=r"Contracts at $q \leq 10$c")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (ET)")
    ax.set_ylabel("Share of taker flow (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", fontsize=6.5)
    ax.text(-0.02, 1.02, "(a)", transform=ax.transAxes, va="bottom")
    c = tab("timing_contrasts").filter(pl.col("sector") == "All")
    ax = fig.add_subplot(gs[1])
    y = _contrast(ax, c, (("ls_share", r"Share at $q \leq 10$c"), ("small_order_share", r"Orders $\leq$ \$100")))
    ax.set_yticks(y, [lab for _, lab in CONTRASTS])
    ax.invert_yaxis()
    ax.set_xlabel("Cell minus rest (pp)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.text(-0.02, 1.02, "(b)", transform=ax.transAxes, va="bottom")
    ax2 = fig.add_subplot(gs[2], sharey=ax)
    _contrast(ax2, c, (("all", "All positions"),))
    ax2.tick_params(labelleft=False)
    ax2.set_xlabel("Taker return, cell minus rest (pp)")
    ax2.text(-0.02, 1.02, "(c)", transform=ax2.transAxes, va="bottom")
    save(fig, "fig_flow_timing")


def fig_markouts() -> None:
    """Per-contract price impact and markouts of taker orders by order size (cents, taker's view)."""
    t = tab("markouts").filter(pl.col("sector") == "All")
    t = t.join(pl.DataFrame({"size_bucket": SIZES, "i": range(4)}), on="size_bucket").sort("i")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    ax = axes[0]
    x = np.arange(4)
    xi = t["i"].to_numpy()
    for i, (col, lab) in enumerate((("impact_1m", "Impact, 1 min"), ("impact_1h", "Impact, 1 h"))):
        _ci(ax, xi, t[col], t[f"{col}_se"], SERIES[i], lab, offset=(i - 0.5) * 0.12)
    _ci(ax, xi, t["half_spread"], t["half_spread_se"], REF, "Effective half-spread", marker="s", ls=":",
        offset=0.18)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(x, SIZE_TXT)
    ax.set_xlabel("Taker order size (dollar notional)")
    ax.set_ylabel("Cents per contract")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.98, 0.03, "(a)", transform=ax.transAxes, ha="right")
    ax = axes[1]
    ms = tab("markouts_strata")
    for i, (col, lab) in enumerate((("markout_1m", "Markout, 1 min"), ("markout_1h", "Markout, 1 h"),
                                    ("markout_settle", "Markout, settlement"))):
        md = strata_median(ms, ["size_bucket"], col).join(
            pl.DataFrame({"size_bucket": SIZES, "i": range(4)}), on="size_bucket").sort("i")
        _band(ax, md["i"], md["med"], md["q25"], md["q75"], SERIES[i], lab, offset=(i - 1) * 0.12)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(x, SIZE_TXT)
    ax.set_xlabel("Taker order size (dollar notional)")
    ax.set_ylabel("Cents per contract, median stratum")
    ax.legend(loc="lower left", fontsize=7)
    ax.text(0.98, 0.03, "(b)", transform=ax.transAxes, ha="right")
    fig.tight_layout()
    save(fig, "fig_flow_markouts")


def fig_oib() -> None:
    """Market-hour regressions: mid change (cents) on signed taker imbalance by size class."""
    sm = json.loads((DATA / "summary.json").read_text())["hourly"]["All"]
    classes = [("oib_small", r"Small ($\leq$\$100)"), ("oib_medium", r"Medium (\$100--1k)"), ("oib_large", r"Large ($>$\$1k)")]
    horizons = [("r_now", "Same hour"), ("r_next", "Next hour"), ("r_settle", "To settlement")]
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    x = np.arange(len(classes))
    for i, (h, lab) in enumerate(horizons):
        if h not in sm:
            continue
        m = [sm[h][c]["coef"] for c, _ in classes]
        se = [sm[h][c]["se"] for c, _ in classes]
        _ci(ax, x, m, se, SERIES[i], lab, ls="none", offset=(i - 1) * 0.15)
    ax.axhline(0, color=REF, lw=0.7, ls="--")
    ax.set_xticks(x, ["Small", "Medium", "Large"])
    ax.set_xlabel("Taker order-size class")
    ax.set_ylabel("Cents per unit imbalance")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_flow_oib")


if __name__ == "__main__":
    apply()
    for f in (fig_roles, fig_roles_groups, fig_size, fig_size_buckets, fig_yesno, fig_timing, fig_markouts, fig_oib):
        f()
        plt.close("all")
    print("flow figures written")
