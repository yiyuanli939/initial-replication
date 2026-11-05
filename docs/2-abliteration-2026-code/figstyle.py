"""Shared look for every figure in the guide.

Colour has one meaning everywhere (palette checked with the dataviz validator; aqua is below 3:1 on the
surface, so aqua marks always carry a direct label):
  ORANGE  harmful prompts, and the refusal direction r
  BLUE    harmless prompts, and the residual stream itself (schematics)
  VIOLET  this guide's edit (3 directions)
  AQUA    methods from the 2025-2026 papers
  INK2    the original model          MUTED   the 2024 recipe (one direction, layer 23)
Math is TeX, typeset by matplotlib's mathtext in Computer Modern. Notation (same as the PDF):
  x_l stream entering layer l, e embedding, a_l / m_l what layer l's mixer / MLP adds,
  r-hat unit refusal direction (r-hat_l read at layer l), alpha steering strength.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path("figs"); FIG.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e9e8e2", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d03b3b"
BLUE_T, ORANGE_T, VIOLET_T, GREY_T = "#eaf2fc", "#fdefe9", "#efedfb", "#f4f3ef"   # tints for boxes
MODEL_COL = {"original": INK2, "2024": MUTED, "guide": VIOLET, "paper": AQUA}

plt.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"], "mathtext.fontset": "cm",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelsize": 8.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8, "savefig.dpi": 220,
})
W_FULL = 7.0          # inches; the PDF text width is 16.6 cm = 6.5 in, figures are scaled to fit


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def rbox(ax, x, y, w, h, text="", fc="#ffffff", ec=AXIS, color=INK, size=8.5, weight="normal", lw=1.1, r=0.06,
         ha="center", family=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.08
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, family=family)


def arrow(ax, p, q, color=INK2, lw=1.3, style="-|>", ls="-", ms=10, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 zorder=z, shrinkA=0, shrinkB=0))


def tok_label(t):
    """Readable token text: byte-level BPE markers to plain characters."""
    return t.replace("Ġ", " ").replace("ĊĊ", "⏎⏎").replace("Ċ", "⏎").strip() or "␣"


def bar_gap(ax):
    """2-px surface gap between adjacent bars (instead of borders)."""
    for p in ax.patches:
        p.set_edgecolor(SURF); p.set_linewidth(0.8)
