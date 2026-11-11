"""Figures for the structural workstream: the no-arbitrage band against the longshot mispricing,
fitted probability-weighting functions, Shin's insider share, and belief-dispersion interactions.

Reads only data/research/flb_structural/ (CSV tables written by
``python -m albert.research.flb_structural run all``). Run with the paper venv:

    cd paper_flb/scripts && /home/ubuntu/.venvs/paper/bin/python fig_structural.py
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import FULL, MUTED, REF, SERIES, apply, save

DATA = Path(os.environ.get("FLB_STRUCTURAL_DIR",
                           Path(__file__).resolve().parents[2] / "data" / "research" / "flb_structural"))
GROUPS = ("Sports", "Weather", "Politics", "Entertainment", "Mentions", "Science & Tech")
LEADS = ("1h", "6h", "24h", "3d", "7d", "30d")
Z = 1.96


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / f"{name}.csv").open() as fh:
        return list(csv.DictReader(fh))


def f(x: str | None) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pick(rs, **kw):
    for r in rs:
        if all(r.get(k, "") == v for k, v in kw.items()):
            return r
    return None


def fig_band() -> None:
    """(a) Gross longshot mispricing at the mid by lead against the taker band; (b) the residual beyond
    the band by group, for a taker and for a maker."""
    B = [r for r in rows("band_cells") if r["rho"] == "0.05" and r["weight"] == "market" and not r.get("structure")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw={"width_ratios": [1, 1.15]})
    x = np.arange(len(LEADS))
    cells = [pick(B, group="All", lead=ld, bucket="LS <=10c") for ld in LEADS]
    g = np.array([f(c["gross_c"]) for c in cells])
    gse = np.array([f(c["gross_se"]) for c in cells])
    band = np.array([f(c["band_c"]) for c in cells])
    a.errorbar(x, g, yerr=Z * gse, fmt="o", color=SERIES[0], capsize=2, lw=1, label="Mispricing at the mid")
    a.plot(x, band, color=REF, ls="--", marker="s", ms=3.5, label="Taker band")
    a.set_xticks(x, LEADS)
    a.set_xlabel("Lead before scheduled time")
    a.set_ylabel("Cents per contract")
    a.set_ylim(bottom=0)
    a.legend(loc="upper left")
    a.text(0.0, 1.02, "(a) Longshot sides, all groups", transform=a.transAxes, fontsize=7.5)
    labs = ["All", *GROUPS]
    y = np.arange(len(labs))[::-1]
    for k, (col, lab, se, off) in enumerate((("resid_c", "Taker (sell at bid)", "resid_se", 0.13),
                                             ("resid_mk_c", "Maker (sell at ask)", "resid_mk_se", -0.13))):
        cs = [pick(B, group=gg, lead="pooled", bucket="LS <=10c") for gg in labs]
        v = np.array([f(c[col]) if c else np.nan for c in cs])
        s = np.array([f(c[se]) if c else np.nan for c in cs])
        b.errorbar(v, y + off, xerr=Z * s, fmt="o", color=SERIES[k], capsize=2, lw=1, label=lab)
    b.axvline(0, color=REF, ls="--", lw=0.9)
    b.set_yticks(y, labs)
    b.set_xlabel("Residual beyond the band (cents per contract)")
    b.grid(True, axis="x")
    b.grid(False, axis="y")
    b.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    b.text(0.0, 1.02, "(b) Longshot sides by group, leads pooled", transform=b.transAxes, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_structural_band")
    plt.close(fig)


def _lo(p):
    p = np.clip(np.asarray(p, dtype=float), 0.003, 0.997)
    return np.log(p / (1 - p))


def fig_weighting() -> None:
    """Fitted weighting functions q = w(pi) (LLO and Prelec) by group on log-odds axes, with the
    calibration points (realised frequency, mean YES price) by price bucket; last panel: gamma by group."""
    C, P, W = rows("weighting_curves"), rows("weighting_points"), rows("weighting_fits")
    panels = ["All", *GROUPS]
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.6))
    axes = axes.ravel()
    ticks = (0.01, 0.1, 0.5, 0.9, 0.99)
    for ax, cell in zip(axes, panels, strict=False):
        cr = [r for r in C if r["cell"] == cell]
        q = np.array([f(r["q"]) for r in cr])
        ax.plot(_lo([0.003, 0.997]), _lo([0.003, 0.997]), color=REF, ls="--", lw=0.8)
        ax.plot(_lo([f(r["pi_llo"]) for r in cr]), _lo(q), color=SERIES[0], label="LLO")
        ax.plot(_lo([f(r["pi_prelec"]) for r in cr]), _lo(q), color=SERIES[1], ls=(0, (4, 1.5)), label="Prelec")
        pr = [r for r in P if r["cell"] == cell]
        ax.plot(_lo([f(r["win"]) for r in pr]), _lo([f(r["q"]) for r in pr]), "o", ms=2.6, color=MUTED,
                label="Buckets", zorder=1)
        tv = _lo(ticks)
        ax.set_xticks(tv, [f"{t:g}" for t in ticks])
        ax.set_yticks(tv, [f"{t:g}" for t in ticks])
        ax.set_xlim(_lo(0.003), _lo(0.997))
        ax.set_ylim(_lo(0.003), _lo(0.997))
        ax.grid(True, axis="both")
        ax.text(0.04, 0.9, cell, transform=ax.transAxes, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    for ax in axes[4:7]:
        ax.set_xlabel(r"Probability $\pi$", fontsize=7.5)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel(r"Price $q=w(\pi)$", fontsize=7.5)
    axes[0].legend(loc="lower right", fontsize=6.5, handlelength=1.6)
    ax = axes[7]
    labs = ["All", *GROUPS]
    y = np.arange(len(labs))[::-1]
    for k, (pre, off) in enumerate((("llo", 0.12), ("prelec", -0.12))):
        cs = [pick(W, dim="all" if lab == "All" else "group", cell=lab, weight="market") for lab in labs]
        g = np.array([f(c[f"{pre}_gamma"]) if c else np.nan for c in cs])
        s = np.array([f(c[f"{pre}_se_gamma"]) if c else np.nan for c in cs])
        ax.errorbar(g, y + off, xerr=Z * s, fmt="o", ms=3, color=SERIES[k], capsize=1.5, lw=0.9)
    ax.axvline(1, color=REF, ls="--", lw=0.8)
    ax.set_yticks(y, ["All", "Sports", "Weather", "Politics", "Entert.", "Mentions", "Sci. & Tech"], fontsize=6.5)
    ax.set_xlabel(r"Curvature $\gamma$", fontsize=7.5)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fig_structural_weighting")
    plt.close(fig)


def fig_shin() -> None:
    """(a) Shin's z implied by the asks and fitted to outcomes, by structure and group; (b) z from the
    asks against the mean half-spread of the legs, per event and lead."""
    S, E = rows("shin_cells"), rows("shin_event_leads")
    cells = [("kind", "two-leg", "Two-leg events"), ("kind", "multi", "Multi-outcome events"),
             ("kind", "partition ladder", "Partition ladders"), *(("group", g, g) for g in GROUPS)]
    cells = [(d, c, lab) for d, c, lab in cells if pick(S, dim=d, cell=c)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(cells))[::-1]
    za = [f(pick(S, dim=d, cell=c)["z_ask"]) for d, c, _ in cells]
    zs = [f(pick(S, dim=d, cell=c)["z_ask_se"]) for d, c, _ in cells]
    zm = [f(pick(S, dim=d, cell=c).get("fit_z_ml")) for d, c, _ in cells]
    lo = [f(pick(S, dim=d, cell=c).get("fit_ci_lo")) for d, c, _ in cells]
    hi = [f(pick(S, dim=d, cell=c).get("fit_ci_hi")) for d, c, _ in cells]
    a.errorbar(za, y + 0.13, xerr=Z * np.array(zs), fmt="o", ms=3.5, color=SERIES[0], capsize=2, lw=1,
               label=r"$z$ from the asks")
    zm_, lo_, hi_ = np.array(zm), np.array(lo), np.array(hi)
    a.errorbar(zm_, y - 0.13, xerr=[zm_ - lo_, hi_ - zm_], fmt="s", ms=3.5, color=SERIES[1], capsize=2, lw=1,
               label=r"$z$ fitted to outcomes")
    a.set_yticks(y, [lab for _, _, lab in cells])
    a.axvline(0, color=REF, ls="--", lw=0.8)
    a.set_xlabel(r"Insider share $z$")
    a.grid(True, axis="x")
    a.grid(False, axis="y")
    a.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    a.text(0.0, 1.02, "(a) Mean and outcome-fitted z", transform=a.transAxes, fontsize=7.5)
    for k, kind in enumerate(("two-leg", "multi", "partition ladder")):
        e = [r for r in E if r["kind"] == kind]
        b.plot([50 * f(r["mean_spread"]) for r in e], [f(r["z_ask"]) for r in e], "o", ms=2.2, alpha=0.55,
               color=SERIES[k], label={"two-leg": "Two-leg", "multi": "Multi-outcome", "partition ladder": "Ladder"}[kind])
    b.set_xlabel("Mean half-spread of the legs (cents)")
    b.set_ylabel(r"$z$ from the asks")
    b.legend(loc="upper left", markerscale=1.6)
    b.text(0.0, 1.02, "(b) Event-leads", transform=b.transAxes, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_structural_shin")
    plt.close(fig)


def fig_beliefs() -> None:
    """Interaction of each characteristic with the calibration slope (logit units per standard deviation;
    anchor: 0/1), pooled, within groups and volume-weighted."""
    R = rows("beliefs_coef")
    chars = [("log_outcomes", "Number of outcomes (log)"), ("turnover", "Turnover"), ("spread_c", "Quoted spread"),
             ("log_lead", "Horizon (log lead)"), ("anchored", "Forecast anchor")]
    specs = [("pooled_anchor", "Pooled"), ("within_group", "Within group"), ("pooled_anchor_vol", "Pooled, volume-weighted")]
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.5))
    y = np.arange(len(chars))[::-1]
    for k, (sp, lab) in enumerate(specs):
        v, s = [], []
        for c, _ in chars:
            r = pick(R, spec=sp, variable=c)
            v.append(f(r["coef"]) if r else np.nan)
            s.append(f(r["se"]) if r else np.nan)
        ax.errorbar(v, y + 0.18 - 0.18 * k, xerr=Z * np.array(s), fmt="o", ms=3.2, color=SERIES[k], capsize=2, lw=1,
                    label=lab)
    ax.axvline(0, color=REF, ls="--", lw=0.8)
    ax.set_yticks(y, [lab for _, lab in chars])
    ax.set_xlabel("Interaction with the calibration slope")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_structural_beliefs")
    plt.close(fig)


if __name__ == "__main__":
    apply()
    fig_band()
    fig_weighting()
    fig_shin()
    fig_beliefs()
    print("figures written")
