"""Figures for the single-contract dynamics study (sections results_dynamics / results_hf).

Reads only saved outputs of ``albert.research.pm_vol`` under data/research/pm_vol/ and writes
paper/figures/fig_dyn_*.pdf. Run with the paper venv:

    cd /home/ubuntu/albert/paper/scripts && /home/ubuntu/.venvs/paper/bin/python fig_dynamics.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from figstyle import FULL, MUTED, REF, SERIES, apply, save

DATA = Path(__file__).resolve().parents[2] / "data" / "research" / "pm_vol"
SHORT = {"Sports pre-game": "Sports pre-game", "Sports in-play": "Sports in-game",
         "Climate and Weather": "Weather", "Mentions": "Mentions", "Politics": "Politics",
         "Elections": "Elections", "Entertainment": "Entertainment", "Science and Technology": "Sci. & Tech."}
LAM_INIT = math.exp(-3.0)     # starting value of the bucket clock: a bucket left there had no data


def load(name):
    return json.loads((DATA / name).read_text())


def fig_clock(res):
    """Piecewise information rate lambda(tau) per hour against tau, with the uniform clock 1/tau."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    groups = (("Sports pre-game", "Sports in-play", "Climate and Weather"),
              ("Mentions", "Politics", "Entertainment"))
    for ax, cats in zip(axes, groups):
        lo_all, hi_all = [], []
        for i, c in enumerate(cats):
            cl = res["categories"][c]["clock_fits"]["buckets"]["clock"]
            e = cl["edges"]
            xs, ys = [], []
            for j, lam in enumerate(cl["lam"]):
                if abs(lam - LAM_INIT) < 1e-6 or lam < 1e-4 * LAM_INIT:
                    continue
                xs.append(math.sqrt(max(e[j], 1.0) * min(e[j + 1], 4096.0)))
                ys.append(lam)
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=SHORT[c])
            lo_all += xs
            hi_all += xs
        t = np.geomspace(min(lo_all) * 0.8, max(hi_all) * 1.2, 50)
        ax.plot(t, 1 / t, color=REF, ls="--", label=r"Uniform $1/\tau$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Time to scheduled end $\tau$ (hours)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel(r"Information rate $\lambda(\tau)$ (per hour)")
    fig.tight_layout()
    save(fig, "fig_dyn_clock")


def fig_backbone(cells):
    """Normalised price variance against p with the fitted power backbone and the probit backbone."""
    cats = ("Sports pre-game", "Sports in-play", "Climate and Weather")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), sharey=True)
    p = np.linspace(0.02, 0.98, 300)
    probit = (np.exp(-0.5 * special.ndtri(p) ** 2) / np.exp(0)) ** 2   # phi(Phi^-1 p)^2 / phi(0)^2
    for ax, c in zip(axes, cats):
        rows = [r for r in cells if r["category"] == c and r["se"] is not None]
        x = np.array([r["p_mid"] for r in rows])
        y = np.array([r["norm_var"] for r in rows])
        se = np.array([r["se"] for r in rows])
        beta = rows[0]["beta"]
        ax.errorbar(x, y, yerr=2 * se, fmt="o", color=SERIES[0], ms=3.5, elinewidth=0.7, capsize=0, label="Data")
        ax.plot(p, (p * (1 - p) / 0.25) ** (2 * beta), color=SERIES[1], label=rf"$[p(1-p)]^{{\beta}}$, $\beta={beta:.2f}$")
        ax.plot(p, probit, color=REF, ls="--", label=r"Probit $\varphi(\Phi^{-1}(p))$")
        ax.set_xlabel(r"Price $p$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.75)
        ax.legend(loc="lower center", fontsize=6.5, handlelength=1.5)
        ax.text(0.02, 0.96, SHORT[c], transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel(r"$\mathrm{E}[\Delta p^2]$ relative to $p=1/2$")
    fig.tight_layout()
    save(fig, "fig_dyn_backbone")


def fig_u(res):
    """var(u) under the uniform and the fitted clock, and excess kurtosis of u, by category."""
    order = ["Sports in-play", "Climate and Weather", "Sports pre-game", "Mentions", "Politics",
             "Entertainment", "Science and Technology", "Elections"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
    ypos = np.arange(len(order))[::-1]
    for k, c in enumerate(order):
        x = res["categories"][c]
        ch = x["clock_chosen"]
        ud = {(u["split"], u["clock"]): u for u in x["u_diag"] if u["group"] == "all"}
        split = "holdout" if ("holdout", ch) in ud else "dev"
        lab = (k == 0)
        axes[0].plot(ud[(split, "uniform")]["var"], ypos[k], "o", color=SERIES[0], label="Uniform clock" if lab else None)
        axes[0].plot(ud[(split, ch)]["var"], ypos[k], "s", color=SERIES[1], label="Fitted clock" if lab else None)
        for j, sp in enumerate(("dev", "holdout")):
            if (sp, ch) in ud:
                axes[1].plot(ud[(sp, ch)]["exkurt"], ypos[k], "o" if j == 0 else "s", color=SERIES[j],
                             label=("Estimation sample" if j == 0 else "Holdout") if lab else None)
    axes[0].axvline(1.0, color=REF, ls="--", lw=0.9)
    axes[0].set_xlabel(r"$\mathrm{var}(u)$ (holdout; Elections: estimation sample)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Excess kurtosis of $u$ (fitted clock, log scale)")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([SHORT[c] for c in order])
    for ax in axes:
        ax.grid(axis="x", color="#e6e5e1", lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_dyn_u")


def fig_signature(hf):
    """HF variance signature and var(u) against the sampling interval: data vs matched M0 tick-grid control."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    s, c = hf["signature"], hf["signature_sim"]
    axes[0].plot([r["h_s"] for r in s], [r["rv_c2_per_hour"] for r in s], marker="o", color=SERIES[0], label="Data")
    axes[0].plot([r["h_s"] for r in c], [r["rv_c2_per_hour"] for r in c], marker="s", color=REF, ls="--",
                 label="M0 control (1c grid)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, None)
    axes[0].set_xlabel("Sampling interval (seconds)")
    axes[0].set_ylabel(r"Realised variance (c$^2$ per in-game hour)")
    axes[0].legend(loc="lower left")
    m = [r for r in hf["m0"] if "series" not in r and r["start_spread_max_c"] == 3.0]
    mc = [r for r in hf["m0_sim"] if "series" not in r and r["start_spread_max_c"] == 3.0]
    axes[1].plot([r["h_s"] for r in m], [r["var"] for r in m], marker="o", color=SERIES[0], label="Data")
    axes[1].plot([r["h_s"] for r in mc], [r["var"] for r in mc], marker="s", color=REF, ls="--", label="M0 control (1c grid)")
    axes[1].axhline(1.0, color=MUTED, ls=":", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, None)
    axes[1].set_xlabel("Horizon $h$ (seconds)")
    axes[1].set_ylabel(r"$\mathrm{var}(u)$, uniform clock")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_dyn_signature")


def fig_drift(dsim, hf):
    """Probit-drift slope (M0 predicts 1): data vs M0 paths quoted on the 1c grid, by sample."""
    rows = []
    for c in ("Climate and Weather", "Mentions", "Elections", "Science and Technology", "Entertainment",
              "Politics", "Sports pre-game"):
        d = {r["variant"]: r for r in dsim if r["category"] == c and r["p_range"] == "all"}
        rows.append((SHORT[c] + " (1 h)", d["data"], d["sim_tick"]))
    for h in (60, 300):
        d = {r["variant"]: r for r in hf["drift_hf"] if r["h_s"] == h and r.get("p_range") == "all"}
        rows.append((f"In-game L2 ({h} s)", d["data"], d["sim_tick"]))
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.7))
    y = np.arange(len(rows))[::-1]
    for k, (lab, d, s) in enumerate(rows):
        ax.errorbar(d["slope"], y[k], xerr=2 * d["se"], fmt="o", color=SERIES[0], ms=4, elinewidth=0.8,
                    label="Data" if k == 0 else None)
        ax.errorbar(s["slope"], y[k] - 0.22, xerr=2 * s["se"], fmt="s", color=REF, ms=3.5, elinewidth=0.8,
                    label="M0 on 1c grid, real spreads" if k == 0 else None)
    ax.axvline(1.0, color=MUTED, ls=":", lw=0.8)
    ax.set_yticks(y - 0.11)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"Probit-drift slope ($\pm 2$ s.e.; M0 predicts 1)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_dyn_drift")


def main():
    apply()
    res = load("results_cat.json")
    fig_clock(res)
    fig_backbone(load("backbone_cells.json"))
    fig_u(res)
    hf = load("hf_results.json")
    fig_signature(hf)
    fig_drift(load("drift_sim.json"), hf)
    print("ok")


if __name__ == "__main__":
    main()
