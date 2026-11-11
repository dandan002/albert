"""Shared matplotlib style for the paper's figures (static PDF, print, light background).

Usage in a figure script (run with the paper venv: /home/ubuntu/.venvs/paper/bin/python):

    from figstyle import FULL, HALF, SERIES, REF, apply, save
    apply()
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.plot(x, y, color=SERIES[0], label="Data")
    ax.plot(x, y0, color=REF, ls="--", label="Model")
    save(fig, "fig_dyn_signature")          # -> paper/figures/fig_dyn_signature.pdf

Rules: at most three categorical series per panel (SERIES[0..2], validated colour-blind safe in
this order; use small multiples beyond three); model / theory / control reference lines are REF
dashed; one y-axis per panel (never twin axes); a legend only when a panel has >= 2 series;
no titles inside figures (the LaTeX caption is the title); axis labels with units.
"""
from pathlib import Path

import matplotlib as mpl

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FULL = 6.5    # \textwidth in inches (article, 1in margins)
HALF = 3.15   # one of two side-by-side panels

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua (validated order)
REF = "#52514e"                               # model / control / reference lines
MUTED = "#8a8985"                             # secondary annotations
GRID = "#e6e5e1"


LM_DIR = Path("/usr/share/texmf/fonts/opentype/public/lm")


def _register_latin_modern() -> None:
    from matplotlib import font_manager
    for f in ("lmroman10-regular.otf", "lmroman10-italic.otf", "lmroman10-bold.otf"):
        if (LM_DIR / f).exists():
            font_manager.fontManager.addfont(str(LM_DIR / f))


def apply() -> None:
    _register_latin_modern()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3d3d3a",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 3,
    })


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    return out
