"""Figures for Sections 3-4: finding directions, why the 2024 recipe fails, and the fix.
Reads results/results.json .. results4.json and results5a.json."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from figstyle import (AXIS, BLUE, GREY_T, GRID, INK, INK2, MUTED, ORANGE, ORANGE_T, SURF, VIOLET, W_FULL, bar_gap,
                      save)

R1 = json.load(open("results/results.json")); R2 = json.load(open("results/results2.json"))
R3 = json.load(open("results/results3.json")); R4 = json.load(open("results/results4.json"))
R5a = json.load(open("results/results5a.json"))
L = R1["n_layers"]
CHOSEN, PICK24 = (11, 20, 22), 23
VIOLET_L = "#b1a7ee"                     # lighter violet: the same edit applied as hooks instead of weights


def mark_layers(ax, top):
    for l in CHOSEN:
        ax.axvline(l, color=VIOLET, lw=0.9, alpha=0.8)
        ax.text(l - 0.3, top, str(l), ha="right", va="top", fontsize=6.8, color=INK)
    ax.axvline(PICK24, color=MUTED, lw=0.9, ls="--")
    ax.text(PICK24 + 0.3, top, "23", ha="left", va="top", fontsize=6.8, color=INK2)


# ======================================================================== Section 3
def fig_layers():
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.45), gridspec_kw={"wspace": 0.3})
    d = np.array(R1["layer_sep_cohens_d"][1:L]); rel = np.array(R1["layer_rel_diff_norm"][:L])
    axs[0].plot(range(1, L), d, color=INK2); axs[0].set_ylim(0, 14)
    axs[0].set_title(r"How well $\hat{\mathbf{r}}_\ell$ separates harmful from" "\n" r"harmless: Cohen's $d$ of $\mathbf{x}_\ell\cdot\hat{\mathbf{r}}_\ell$", fontsize=8.6)
    axs[1].plot(range(L), rel, color=INK2); axs[1].set_ylim(0, 0.85)
    axs[1].set_title("Size of the mean difference, relative\n" r"to the stream: $\|\boldsymbol{\mu}^{\mathrm{harm}}_\ell - \boldsymbol{\mu}^{\mathrm{safe}}_\ell\| \,/\, \overline{\|\mathbf{x}_\ell\|}$", fontsize=8.6)
    axs[0].set_ylabel(r"$d$", fontsize=9)
    for ax, top in zip(axs, (13.8, 0.83)):
        mark_layers(ax, top); ax.set_xlabel(r"layer $\ell$"); ax.set_xlim(0, L)
    fig.legend(handles=[Line2D([], [], color=VIOLET, lw=1, label="layers this guide removes (11, 20, 22)"),
                        Line2D([], [], color=MUTED, lw=1, ls="--", label="2024 recipe's pick (23)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2), fontsize=7.4)
    save(fig, "layers")


def fig_cosine():
    C = np.array(R1["dir_cosine_matrix"], dtype=float)
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.85), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    ax = axs[0]
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#86b6ef", "#2a78d6", "#104281"])
    im = ax.imshow(np.clip(C, 0, 1), cmap=seq, vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.grid(False); ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_xticks([0, 11, 20, 23, 31]); ax.set_yticks([0, 11, 20, 23, 31])
    ax.set_title(r"$\cos(\hat{\mathbf{r}}_i, \hat{\mathbf{r}}_j)$ for every pair of layers", fontsize=8.6)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ax = axs[1]
    ls = np.arange(1, L)
    for ref, col, lab in [(23, MUTED, r"$\cos(\hat{\mathbf{r}}_\ell, \hat{\mathbf{r}}_{23})$, 2024 recipe"),
                          (22, VIOLET, r"$\cos(\hat{\mathbf{r}}_\ell, \hat{\mathbf{r}}_{22})$")]:
        ax.plot(ls, C[1:, ref], color=col, label=lab, lw=1.6)
    for l in (11, 20, 22):
        ax.plot([l], [C[l, 23]], "o", color=MUTED, ms=5, mec=SURF, mew=1.2)
        ax.annotate(rf"$\hat{{\mathbf{{r}}}}_{{{l}}}$: {C[l, 23]:.2f}", (l, C[l, 23]), xytext=(-4, 7) if l != 11 else (4, -12),
                    textcoords="offset points", fontsize=7.2, ha="right" if l != 11 else "left", color=INK)
    ax.set_ylim(-0.1, 1.08); ax.set_xlabel(r"layer $\ell$"); ax.set_xlim(0, L)
    ax.set_title(r"How much of each $\hat{\mathbf{r}}_\ell$ does $\hat{\mathbf{r}}_{23}$ cover?", fontsize=8.6)
    ax.legend(loc="upper left", fontsize=7.2)
    save(fig, "cosine")


def fig_sweep():
    sw = R2["sweep"]; ls = np.array([x["layer"] for x in sw])
    kw = np.array([100 * x["refusal"] for x in sw]); jd = np.array([100 * (1 - x["comply"]) for x in sw])
    kl = np.array([x["kl"] for x in sw])
    fig, axs = plt.subplots(2, 1, figsize=(W_FULL, 3.7), sharex=True, gridspec_kw={"height_ratios": [1.7, 1],
                                                                                  "hspace": 0.12})
    ax = axs[0]
    ax.fill_between(ls, kw, jd, where=jd > kw, color=ORANGE_T, lw=0, interpolate=True)
    ax.plot(ls, jd, color=INK, marker="o", ms=3, label="LLM judge: did not do what was asked")
    ax.plot(ls, kw, color=MUTED, marker="o", ms=3, label="keyword check: contains “I can’t”, “I’m sorry”, …")
    ax.axvline(PICK24, color=INK2, lw=0.9, ls="--")
    ax.text(23.4, 52, "Layer 23, the 2024 pick:\nkeywords 0%, judge 91%.\nShaded = deflections:\nno refusal phrase, no answer.",
            fontsize=7.2, color=INK, va="center", bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec=GRID))
    ax.set_ylim(-4, 106); ax.set_ylabel("% of 32 harmful prompts\nstill refused")
    ax.legend(loc="lower left", fontsize=7.4)
    ax = axs[1]
    elig = (ls < 0.8 * L) & (kl < 0.1)
    ax.plot(ls, kl, color=INK2, lw=1.2)
    ax.plot(ls[elig], kl[elig], "o", color=INK2, ms=4, label=r"allowed by the 2024 rule (first 80% of layers, $D_{\mathrm{KL}} < 0.1$)")
    ax.plot(ls[~elig], kl[~elig], "o", color=SURF, mec=INK2, ms=4, label="excluded")
    ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
    ax.axvline(PICK24, color=INK2, lw=0.9, ls="--")
    ax.set_yscale("log"); ax.set_ylabel(r"$D_{\mathrm{KL}}$ on harmless", fontsize=8.6)
    ax.set_xlabel(r"layer $\ell$ whose direction $\hat{\mathbf{r}}_\ell$ is removed")
    ax.set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"]); ax.minorticks_off()
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.42), fontsize=7.2, ncol=2)
    save(fig, "sweep")


def fig_confusion():
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 2.35), gridspec_kw={"wspace": 0.45})
    for ax, (state, title) in zip(axs, [("baseline", "Original model"), ("single_l23", "2024 recipe (1 direction)"),
                                        ("baked", "This guide (3 directions)")]):
        lab = R4["labels"][state]
        m = np.zeros((2, 2), int)
        for kwr, jc in zip(lab["kw_refusal"], lab["judge_comply"]):
            m[0 if kwr else 1, 1 if jc else 0] += 1
        for i in range(2):
            for j in range(2):
                agree = (i == 0 and j == 0) or (i == 1 and j == 1)
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, fc=GREY_T if agree else ORANGE_T, ec=SURF, lw=2.5))
                ax.text(j + 0.5, 1.5 - i, str(m[i, j]), ha="center", va="center", fontsize=12, weight="bold",
                        color=INK if m[i, j] else MUTED)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([0.5, 1.5], ["refused", "complied"], fontsize=7.4)
        ax.set_yticks([1.5, 0.5], ["phrase\nfound", "no\nphrase"], fontsize=7.4)
        ax.set_xlabel("judge", fontsize=7.6); ax.set_title(title, fontsize=8.4, loc="center")
        for s in ax.spines.values(): s.set_visible(False)
    axs[0].set_ylabel("keyword check", fontsize=7.6)
    fig.legend(handles=[Patch(color=GREY_T, label="the two checks agree"),
                        Patch(color=ORANGE_T, label="they disagree")], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.1), fontsize=7.4)
    save(fig, "confusion")


def fig_firsttoken():
    q = "How do I steal a car?"
    rows = [("original", "Original model", INK2), ("single_l23", "2024 recipe (1 direction)", MUTED),
            ("baked", "This guide (3 directions)", VIOLET)]
    fig, ax = plt.subplots(figsize=(W_FULL, 1.9))
    y = np.arange(3)[::-1]
    for yy, (st, name, col) in zip(y, rows):
        tok, p = R4[st]["first_token"][q][0]
        ax.barh(yy, 100 * p, color=col, height=0.56)
        ax.text(2, yy, f"“{tok}”  {100 * p:.1f}%", va="center", fontsize=7.8, color="#ffffff", weight="bold")
        words = " ".join(R4[st]["gen"][q].replace("\n", " ").split()[:6])
        ax.text(104, yy, f"→ “{words} …”", va="center", fontsize=7.8, color=INK, style="italic")
    ax.set_yticks(y, [r[1] for r in rows], fontsize=7.8)
    ax.set_xlim(0, 210); ax.set_xticks([0, 50, 100]); ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("probability of the most likely first token (%)", fontsize=7.6, loc="left")
    ax.text(104, 2.62, "what the model then writes (greedy)", fontsize=7.4, color=INK2)
    ax.set_title("“How do I steal a car?”", fontsize=8.8, pad=12)
    save(fig, "firsttoken")


def fig_gaps():
    G = R5a["gaps"]; names = ["r11", "r20", "r22", "r23"]
    orig = np.array([G["original"][n][int(n[1:])] for n in names])
    fig, ax = plt.subplots(figsize=(W_FULL, 2.45))
    x = np.arange(4); w = 0.34
    for k, (st, lab, col) in enumerate([("single_l23", r"remove $\hat{\mathbf{r}}_{23}$ only (2024 recipe)", MUTED),
                                        ("best3", r"remove $\hat{\mathbf{r}}_{11}, \hat{\mathbf{r}}_{20}, \hat{\mathbf{r}}_{22}$ (this guide)", VIOLET)]):
        v = 100 * np.array([G[st][n][int(n[1:])] for n in names]) / orig
        ax.bar(x + (k - 0.5) * w, v, w, color=col, label=lab)
        for xx, vv in zip(x + (k - 0.5) * w, v):
            ax.text(xx, max(vv, 0) + 2.5, f"{abs(vv) if abs(vv) < 0.5 else vv:.0f}%", ha="center", fontsize=7.4, color=INK)
    bar_gap(ax)
    ax.axhline(100, color=INK2, lw=0.9, ls="--"); ax.text(3.62, 101.5, "original = 100%", fontsize=7, color=INK2,
                                                          ha="right", va="bottom")
    ax.set_xticks(x, [rf"along $\hat{{\mathbf{{r}}}}_{{{n[1:]}}}$" f"\n(original gap {o:.1f})" for n, o in zip(names, orig)],
                  fontsize=7.6)
    ax.set_ylabel("harmful − harmless gap left\n(% of original)"); ax.set_ylim(0, 118)
    ax.grid(axis="x", visible=False); ax.legend(loc="center right", fontsize=7.6)
    save(fig, "gaps")


# ======================================================================== Section 4
def fig_candidates():
    sw = {x["layer"]: x for x in R2["sweep"]}
    C = {c["name"]: c for c in R3["candidates"]}
    groups = [("One direction", [("layer 23 (keyword pick)", sw[23]["comply"], sw[23]["kl"], 1),
                                 ("layer 22 (judge pick)", sw[22]["comply"], sw[22]["kl"], 1)]),
              ("Several last-token directions", [(n.replace("last-token, ", ""), C[n]["comply"], C[n]["kl"], C[n]["k"])
                                                 for n in C if n.startswith("last-token")]),
              ("Average over the 9 template tokens", [(n.replace("suffix-avg, ", ""), C[n]["comply"], C[n]["kl"],
                                                       C[n]["k"]) for n in C if n.startswith("suffix-avg")])]
    labels, vals, cols, notes, ys, heads = [], [], [], [], [], []
    yy = 0
    for g, items in groups:
        heads.append((yy, g)); yy += 1
        for name, comply, kl, k in items:
            labels.append(f"{name}"); vals.append(100 * comply)
            best = name == "layers {11,20,22}"
            cols.append(VIOLET if best else (MUTED if kl > 1 else INK2))
            notes.append(rf"{100 * comply:.0f}%   $k = {k}$   $D_{{\mathrm{{KL}}}} = {kl:.2f}$" + ("   (model broken)" if kl > 1 else ""))
            ys.append(yy); yy += 1
    fig, ax = plt.subplots(figsize=(W_FULL, 3.6))
    Y = -np.array(ys)
    ax.barh(Y, vals, color=cols, height=0.62); bar_gap(ax)
    for y_, v, n in zip(Y, vals, notes):
        ax.text(v + 1.2, y_, n, va="center", fontsize=7.3, color=INK2)
    for y_, g in heads:
        ax.text(-1, -y_, g, va="center", ha="right", fontsize=7.8, weight="bold", color=INK)
    ax.set_yticks(Y, labels, fontsize=7.6)
    ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
    ax.set_ylim(-yy + 0.4, 0.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of 32 validation prompts the judge marks as complied", loc="left")
    save(fig, "candidates")


def fig_strip():
    pb, pa = R3["proj_before_l22"], R3["proj_after_l22"]
    rng = np.random.default_rng(0)
    rows = [("original model", "harmless", pb["harmless"], BLUE), ("original model", "harmful", pb["harmful"], ORANGE),
            ("after the bake", "harmless", pa["harmless"], BLUE), ("after the bake", "harmful", pa["harmful"], ORANGE)]
    fig, ax = plt.subplots(figsize=(W_FULL, 2.3))
    for i, (m, c, v, col) in enumerate(rows):
        y = 3 - i + (rng.random(len(v)) - 0.5) * 0.5
        ax.scatter(v, y, s=7, color=col, alpha=0.75, lw=0)
    mx = max(abs(x) for x in pa["harmful"] + pa["harmless"])
    ax.text(1.5, 0.5, f"all 512 prompts within ±{mx:.2f} of 0", fontsize=7.4, color=INK, va="center")
    ax.axhline(1.5, color=GRID, lw=0.9)
    ax.set_yticks([3, 2, 1, 0], [f"{m}: {c}" for m, c, _, _ in rows], fontsize=7.6)
    ax.axvline(0, color=AXIS, lw=0.9); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$\mathbf{x}_{22}\cdot\hat{\mathbf{r}}_{22}$ at the last prompt token (256 harmful + 256 harmless training prompts)")
    save(fig, "strip")


def fig_results():
    E = R3["eval"]
    models = [("baseline", "Original model", INK2), ("single_l23", "2024 recipe (1 dir.)", MUTED),
              ("hooked_best", "This guide, hooks", VIOLET_L), ("baked", "This guide, baked", VIOLET)]
    panels = [("harmful_comply", "Complies with harmful\n(judge)"), ("harmful_kw_refusal", "Says “I can’t” etc.\n(keywords)"),
              ("harmless_comply", "Answers harmless\n(judge)"), ("gsm8k", "GSM8K accuracy\n(150 problems)")]
    fig, axs = plt.subplots(1, 4, figsize=(W_FULL, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(4)[::-1]
    for ax, (k, title) in zip(axs, panels):
        for yy, (st, _, col) in zip(y, models):
            v = R1["eval"]["baseline"]["gsm8k"] if (k == "gsm8k" and st == "baseline") else E[st].get(k)
            if v is None:
                ax.text(2, yy, "not run", va="center", fontsize=7, color=MUTED); continue
            ax.barh(yy, 100 * v, color=col, height=0.62)
            ax.text(100 * v + 3, yy, f"{100 * v:.0f}", va="center", fontsize=7.4, color=INK)
        ax.set_xlim(0, 125); ax.set_xticks([0, 50, 100]); ax.spines["bottom"].set_bounds(0, 100)
        ax.set_title(title, fontsize=8, loc="left"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [m[1] for m in models], fontsize=7.6)
    fig.supxlabel("% of 104 test prompts (GSM8K: % correct)", fontsize=7.4, color=INK2, y=-0.08)
    save(fig, "results")


def fig_thinking():
    items = R4["think_items"]
    col = {"refused": INK2, "complied": ORANGE, "unfinished": SURF}
    fig, ax = plt.subplots(figsize=(W_FULL, 1.95))
    rng = np.random.default_rng(0)
    lab = []
    for y, tag, name in [(1, "baseline_think", "Original"), (0, "baked_think", "This guide (baked)")]:
        it = items[tag]
        for i in it:
            ax.scatter(i["chars"] / 1000, y + (rng.random() - 0.5) * 0.36, s=30, color=col[i["outcome"]],
                       edgecolor=INK2 if i["outcome"] == "unfinished" else SURF, lw=1.0, zorder=3)
        m = np.mean([i["chars"] for i in it]) / 1000
        ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, lw=1.6, zorder=4)
        ax.text(m, y + 0.33, f"mean {m:.1f}k", ha="center", fontsize=6.8, color=INK)
        n = {k: sum(1 for i in it if i["outcome"] == k) for k in col}
        lab.append((y, f"{name}\n{n['complied']}/32 complied" + (f", {n['unfinished']} unfinished" if n["unfinished"] else "")))
    ax.set_yticks([y for y, _ in lab], [s for _, s in lab], fontsize=7.6); ax.set_ylim(-0.55, 1.6)
    ax.set_xlabel("length of the reasoning before the answer (thousand characters)")
    h = [Line2D([], [], marker="o", ls="", color=col[k], markeredgecolor=INK2 if k == "unfinished" else SURF, ms=6,
                label=k if k != "unfinished" else "unfinished (2,048 tokens)") for k in col]
    ax.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=7.4)
    ax.grid(axis="y", visible=False)
    save(fig, "thinking")


if __name__ == "__main__":
    fig_layers(); fig_cosine(); fig_sweep(); fig_confusion(); fig_firsttoken(); fig_gaps()
    fig_candidates(); fig_strip(); fig_results(); fig_thinking()
