"""Figures for the 'dance on the residual stream' framing and the paper comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figs import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURF, VIOLET, save

CODEBG = "#f4f3ef"


def arrow(ax, p, q, color=INK2, lw=1.6, style="-|>", ls="-", ms=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls))


def lane(ax, x0=1.2, x1=2.0, y0=0.3, y1=4.3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#eef4fc", ec=BLUE, lw=1.3))
    arrow(ax, ((x0 + x1) / 2, y0 + 0.1), ((x0 + x1) / 2, y1 - 0.1), color=BLUE, lw=1.0, ms=9)


def writer(ax, y, label, x=2.7, w=1.9, ec=AXIS, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y - 0.28), w, 0.56, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=7.6)


def fig_moves():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.2))
    titles = ["READ", "SUBTRACT", "ADD"]
    codes = ["h = out.hidden_states[l][:, -1]\nscore = h @ r",
             "with ablate(model, r):\n    model.generate(...)",
             "with add_at(model, l, a * r):\n    model.generate(...)"]
    notes = ["look at the stream;\nchange nothing",
             "every writer drops\nits r component",
             "push the stream\nalong r at one layer"]
    for ax, t, c, n in zip(axs, titles, codes, notes):
        ax.set_xlim(0, 5); ax.set_ylim(-1.5, 5.0); ax.axis("off")
        lane(ax)
        ax.text(0.05, 4.75, t, fontsize=11, weight="bold", color=INK)
        ax.text(0.05, -0.55, c, fontsize=6.7, family="DejaVu Sans Mono", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc=CODEBG, ec="none"))
        ax.text(4.95, 4.75, n, fontsize=7.2, color=INK2, ha="right", va="top")
    # READ
    ax = axs[0]
    writer(ax, 1.4, "layer l"); arrow(ax, (2.7, 1.4), (2.02, 1.4), color=MUTED)
    ax.add_patch(FancyBboxPatch((2.7, 2.65), 1.9, 0.75, boxstyle="round,pad=0.01,rounding_size=0.06", fc="#ffffff", ec=BLUE, lw=1.1))
    ax.text(3.65, 3.02, "x · r = 36.7", ha="center", va="center", fontsize=8, color=BLUE, weight="bold")
    arrow(ax, (2.02, 3.0), (2.68, 3.0), color=BLUE, ls="--")
    # SUBTRACT
    ax = axs[1]
    for y, lab in [(1.0, "o_proj / out_proj"), (2.2, "down_proj"), (3.4, "next writer …")]:
        writer(ax, y, lab, ec=ORANGE, fc="#fdf0ea")
        arrow(ax, (2.7, y), (2.02, y), color=ORANGE)
        ax.text(2.36, y + 0.2, "−r", fontsize=7.5, color=ORANGE, ha="center", weight="bold")
    # ADD
    ax = axs[2]
    writer(ax, 1.4, "layer l − 1"); arrow(ax, (2.7, 1.4), (2.02, 1.4), color=MUTED)
    writer(ax, 3.3, "layer l"); arrow(ax, (2.02, 3.3), (2.7, 3.3), color=MUTED)
    arrow(ax, (0.2, 2.45), (1.18, 2.45), color=VIOLET, lw=2.2)
    ax.text(0.2, 2.62, "+ a·r", fontsize=8.5, color=VIOLET, weight="bold")
    fig.tight_layout(); save(fig, "moves")


def fig_gaps():
    R = json.load(open("results/results5a.json"))["gaps"]
    names = ["r11", "r20", "r22", "r23"]
    states = [("original", "original", INK2), ("single_l23", "remove r23 only (2024 recipe)", MUTED),
              ("best3", "remove r11, r20, r22 (this guide)", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(names)); w = 0.26
    for i, (k, lab, col) in enumerate(states):
        vals = [R[k][n][int(n[1:])] for n in names]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col, label=lab)
        for xx, v in zip(x + (i - 1) * w, vals):
            ax.text(xx, max(v, 0) + 1.2, f"{abs(v):.1f}" if abs(v) < 0.05 else f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"along {n}\n(at layer {n[1:]})" for n in names], fontsize=8)
    ax.set_ylabel("harmful − harmless gap")
    ax.set_ylim(0, 68); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.6, loc="upper left")
    fig.tight_layout(); save(fig, "gaps")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 1.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
    steps = [("READ", "hidden states of\n256 + 256 prompts"), ("CHOOSE", "difference of means;\nscore with a judge"),
             ("SUBTRACT", "hooks remove r\nfrom every writer"), ("BAKE", "same edit, written\ninto the weights"),
             ("CHECK", "judge, KL, GSM8K,\nthinking, audit")]
    w, gap = 1.72, 0.3
    for i, (t, s) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        ec = ORANGE if t in ("SUBTRACT", "BAKE") else BLUE
        fc = "#fdf0ea" if t in ("SUBTRACT", "BAKE") else "#eef4fc"
        ax.add_patch(FancyBboxPatch((x, 0.1), w, 1.35, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 0.12, 1.15, f"{i + 1}  {t}", fontsize=9.5, weight="bold", va="center")
        ax.text(x + 0.12, 0.55, s, fontsize=7.4, color=INK2, va="center")
        if i < 4:
            arrow(ax, (x + w + 0.02, 0.78), (x + w + gap - 0.02, 0.78))
    save(fig, "pipeline")


def fig_methods():
    import os
    R = json.load(open("results/" + os.environ.get("R5C", "results5c.json")))["test"]
    rows = list(R.items())
    order = sorted(range(len(rows)), key=lambda i: rows[i][1]["harmful_comply"])
    rows = [rows[i] for i in order]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True, gridspec_kw={"width_ratios": [2.2, 1.4, 1.4]})
    y = np.arange(len(rows))
    from labels import label
    def col(name):
        if name.startswith("original"): return INK2
        if "ours" in name: return VIOLET
        if "2024" in name: return MUTED
        return BLUE
    cols = [col(n) for n, _ in rows]
    for ax, key, title, scale in [(axs[0], "harmful_comply", "complies with harmful (%)", 100),
                                  (axs[1], "harmless_comply", "answers harmless (%)", 100),
                                  (axs[2], "kl", "KL on harmless", 1)]:
        vals = [scale * r[key] for _, r in rows]
        ax.barh(y, vals, color=cols, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v + (2 if scale == 100 else 0.03), yy, f"{v:.0f}" if scale == 100 else f"{v:.2f}", va="center", fontsize=7)
        ax.set_title(title, fontsize=8.6, loc="center"); ax.grid(axis="y", visible=False)
        if scale == 100: ax.set_xlim(0, 118)
    axs[2].set_xlim(0, max(r["kl"] for _, r in rows) * 1.35)
    axs[0].set_yticks(y, [label(n) for n, _ in rows], fontsize=7.4)
    fig.tight_layout(); save(fig, "methods")


def fig_cosmic():
    R = json.load(open("results/results5a.json")); R2 = json.load(open("results/results2.json"))
    Sc = np.array(R["cosmic"]["Sc_cat"]); KL = np.array(R["cosmic"]["KL"])
    comp = {s["layer"]: s["comply"] for s in R2["sweep"]}
    ls = np.arange(1, 32)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axs[0]
    ax.plot(ls, Sc[-1, 1:32], color=BLUE, marker="o", ms=3, label="COSMIC S_comply (no text)")
    ax.axhline(R["cosmic_baseline_cos"]["cat"], color=MUTED, ls="--", lw=1)
    ax.text(1, R["cosmic_baseline_cos"]["cat"] + 0.01, "no intervention", fontsize=7, color=MUTED)
    ax2 = axs[1]
    ax2.scatter(Sc[-1, 1:32], [100 * comp[l] for l in ls], color=BLUE, s=22, edgecolor=SURF, zorder=3)
    for l in (23, 26):
        ax2.annotate(f"L{l}", (Sc[-1, l], 100 * comp[l]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax2.annotate("L11, L20, L22", (0.826, 31), xytext=(-62, -2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("layer of the direction (last token)"); ax.set_ylabel("cos(harmless, harmful + ablation)")
    ax.set_title("COSMIC's text-free score", fontsize=9)
    ax2.set_xlabel("COSMIC S_comply"); ax2.set_ylabel("judged compliance (%)")
    ax2.set_title("…does not predict real compliance", fontsize=9)
    fig.tight_layout(); save(fig, "cosmic")


def fig_rdo():
    R = json.load(open("results/results5b.json"))
    h = np.array([x[1:] for x in R["rdo_hist"]])          # loss, abl, add, kl
    ep = np.array([x[0] for x in R["rdo_hist"]])
    steps = np.arange(len(h))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for k, (lab, c) in enumerate([("ablation CE", ORANGE), ("addition CE", BLUE)]):
        axs[0].plot(steps, h[:, k + 1], color=c, lw=1.2, alpha=0.9, label=lab)
    axs[0].set_xlabel("step (16 prompts each)"); axs[0].set_ylabel("loss"); axs[0].legend(fontsize=7.5)
    axs[0].set_title("Training the direction", fontsize=9)
    axs[1].plot(steps, h[:, 3], color=VIOLET, lw=1.2)
    axs[1].set_xlabel("step"); axs[1].set_ylabel("retain KL"); axs[1].set_title("…while harmless outputs stay put", fontsize=9)
    fig.tight_layout(); save(fig, "rdo")


if __name__ == "__main__":
    import sys
    fig_moves(); fig_gaps(); fig_pipeline(); fig_cosmic(); fig_rdo()
    if "--methods" in sys.argv:
        fig_methods()
