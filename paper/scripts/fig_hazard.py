"""Figures for the deadline-contract (hazard) study.

Reads the saved study output data/research/pm_hazard/summary.json (written by
``python -m albert.research.pm_hazard analyze``) and writes paper/figures/fig_haz_*.pdf.
Run with the paper venv:  cd paper/scripts && /home/ubuntu/.venvs/paper/bin/python fig_hazard.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figstyle import HALF, FULL, REF, SERIES, apply, save

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "research" / "pm_hazard" / "summary.json"
LEAD_ORDER = ["<6h", "6-24h", "1-3d", "3-7d", "7-14d", "14-30d", "30-60d"]
LEAD_LABEL = ["<6 h", "6–24 h", "1–3 d", "3–7 d", "7–14 d", "14–30 d", "30–60 d"]


def load() -> dict:
    return json.loads(SUMMARY.read_text())


def fig_calibration(S: dict) -> None:
    """Hazard multiplier k by lead with 5-95% cluster-bootstrap intervals, by contract type."""
    rows = S["calibration_by_kind_lead"]
    kinds = [("event", "One-off events"), ("mention", "Mentions"), ("elimination", "Eliminations")]
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    x = np.arange(len(LEAD_ORDER))
    for j, (kind, label) in enumerate(kinds):
        pts = {r["lead_b"]: r for r in rows if r["kind"] == kind and r.get("k_lo") is not None and r["n"] >= 30}
        xs, ks, lo, hi = [], [], [], []
        for i, lb in enumerate(LEAD_ORDER):
            if lb in pts:
                r = pts[lb]
                xs.append(i + (j - 1) * 0.22)
                ks.append(r["k_mult"])
                lo.append(r["k_mult"] - r["k_lo"])
                hi.append(r["k_hi"] - r["k_mult"])
        ax.errorbar(xs, ks, yerr=[lo, hi], fmt="o", color=SERIES[j], ms=3.8, lw=1.0, capsize=2, label=label)
    ax.axhline(1.0, color=REF, ls="--", lw=0.9)
    ax.set_xticks(x, LEAD_LABEL)
    ax.set_xlabel("Time to deadline at the decision")
    ax.set_ylabel(r"Hazard multiplier $k$ (realised / implied)")
    ax.set_ylim(0, 2.1)
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig_haz_calibration")
    plt.close(fig)


def fig_hazard_path(S: dict) -> None:
    """Median implied average hazard relative to its value 7 days before the deadline, contracts
    with no event yet, for the three largest categories."""
    rows = [r for r in S["decay_paths_by_category"] if r["anchor_h"] == 168]
    cats = [("Politics", "Politics"), ("Science and Technology", "Science and technology"), ("Mentions", "Mentions")]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for j, (cat, label) in enumerate(cats):
        pts = sorted(((r["lead_h"], r["median_h_ratio"]) for r in rows if r["group"] == cat), reverse=True)
        L = [168] + [p[0] for p in pts]
        v = [1.0] + [p[1] for p in pts]
        ax.plot(L, v, marker="o", ms=3, color=SERIES[j], label=label)
    ax.axhline(1.0, color=REF, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 0.8)
    ticks = [168, 72, 24, 6, 1]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([1, 2, 5, 10, 30], ["1", "2", "5", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Hours to deadline")
    ax.set_ylabel(r"Median $\bar h(L)/\bar h(168\,\mathrm{h})$")
    ax.legend(loc="upper left")
    save(fig, "fig_haz_hazard_path")
    plt.close(fig)


def fig_theta(S: dict) -> None:
    """Price relative to its level 7 days out for the 15-50c cohort (no event yet): market mean
    versus the flat-hazard (deterministic theta) path."""
    rows = sorted((r for r in S["decay_paths"] if r["anchor_h"] == 168 and r["group"] == "core" and r["p0_b"] == "15-50"),
                  key=lambda r: -r["lead_h"])
    L = [168] + [r["lead_h"] for r in rows]
    real = [100.0] + [100 * r["realized_ratio"] for r in rows]
    model = [100.0] + [100 * r["model_ratio"] for r in rows]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.plot(L, real, marker="o", ms=3, color=SERIES[0], label="Market (mean)")
    ax.plot(L, model, color=REF, ls="--", label="Flat-hazard theta")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, 0.8)
    ticks = [168, 72, 24, 6, 1]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Hours to deadline")
    ax.set_ylabel(r"Price, % of its level at 168 h")
    ax.legend(loc="lower left")
    save(fig, "fig_haz_theta")
    plt.close(fig)


def fig_timing(S: dict) -> None:
    """Event-timing PIT under the flat implied hazard: share of YES resolutions per quintile."""
    et = S["event_timing"]["by_kind"]
    kinds = [("event", "One-off events"), ("mention", "Mentions"), ("elimination", "Eliminations")]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    x = np.arange(5)
    w = 0.26
    for j, (kind, label) in enumerate(kinds):
        ax.bar(x + (j - 1) * w, et[kind]["pit_quintiles"], width=w, color=SERIES[j], label=label)
    ax.axhline(0.2, color=REF, ls="--", lw=0.9)
    ax.set_xticks(x, ["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"])
    ax.set_xlabel("PIT of the resolution time (flat implied hazard)")
    ax.set_ylabel("Share of YES resolutions")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    save(fig, "fig_haz_timing")
    plt.close(fig)


def fig_variance(S: dict) -> None:
    """Decomposition of the quadratic variation of the hourly mid up to settlement."""
    rows = {r["group"]: r for r in S["jump_diffusion_qv"]}
    kinds = [("event", "One-off events"), ("mention", "Mentions"), ("count", "Counts"), ("elimination", "Eliminations")]
    comps = [("Diffusion (<20¢/h)", lambda r: r["share_diffusion"]),
             ("Pre-resolution jumps", lambda r: r["share_news_jumps"]),
             ("Resolution", lambda r: r["share_yes_resolution"] + r["share_no_deadline_step"])]
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    y = np.arange(len(kinds))[::-1]
    left = np.zeros(len(kinds))
    for j, (label, f) in enumerate(comps):
        vals = np.array([f(rows[k]) for k, _ in kinds])
        ax.barh(y, vals, left=left, color=SERIES[j], height=0.6, label=label)
        left += vals
    ax.set_yticks(y, [lab for _, lab in kinds])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of quadratic variation")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=6.5, handlelength=1.2,
              columnspacing=0.8)
    save(fig, "fig_haz_variance")
    plt.close(fig)


def main() -> None:
    apply()
    S = load()
    fig_calibration(S)
    fig_hazard_path(S)
    fig_theta(S)
    fig_timing(S)
    fig_variance(S)
    print("wrote fig_haz_calibration, fig_haz_hazard_path, fig_haz_theta, fig_haz_timing, fig_haz_variance")


if __name__ == "__main__":
    main()
