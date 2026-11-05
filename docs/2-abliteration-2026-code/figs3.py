"""Figures for Sections 5-6: the papers (RDO, COSMIC), every method head to head, and the checkpoint audit.
Reads results/results2.json, results5a.json, results5b.json, results5c.json, results5d.json and audit.json."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import AQUA, AXIS, BLUE, GRID, INK, INK2, MODEL_COL, MUTED, ORANGE, SURF, VIOLET, W_FULL, bar_gap, save
from labels import all_methods, family, label

R2 = json.load(open("results/results2.json")); R5a = json.load(open("results/results5a.json"))
R5b = json.load(open("results/results5b.json"))
FAM_NAME = {"original": "original model", "2024": "2024 recipe", "paper": "2025-2026 papers", "guide": "this guide"}


def spearman(a, b):
    """Spearman rank correlation with average ranks for ties (no scipy needed)."""
    def rank(v):
        v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for x in np.unique(v):
            r[v == x] = r[v == x].mean()
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


# ======================================================================== 5.1 RDO
def fig_rdo():
    h = np.array(R5b["rdo_hist"])                       # columns: epoch, total loss, L_abl, L_add, L_ret
    ep = h[:, 0].astype(int); n_ep = ep.max() + 1
    x = np.arange(len(h)) / (len(h) / n_ep)
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.5), gridspec_kw={"wspace": 0.3})
    for col, k, lab in [(ORANGE, 2, r"$\mathcal{L}_{\mathrm{abl}}$: answer harmful prompts with $\hat{\mathbf{r}}$ removed"),
                        (BLUE, 3, r"$\mathcal{L}_{\mathrm{add}}$: refuse harmless prompts with $\alpha\hat{\mathbf{r}}$ added")]:
        axs[0].plot(x, h[:, k], color=col, lw=0.6, alpha=0.3)
        axs[0].plot(np.arange(n_ep) + 0.5, [h[ep == e, k].mean() for e in range(n_ep)], color=col, marker="o", ms=2.8,
                    label=lab)
    axs[0].set_ylabel("cross-entropy"); axs[0].set_xlabel("epoch (16 steps of 16 prompts)")
    axs[0].set_title("Training one direction", fontsize=8.8)
    axs[0].legend(loc="upper right", fontsize=6.8)
    axs[1].plot(x, h[:, 4], color=BLUE, lw=0.6, alpha=0.3)
    axs[1].plot(np.arange(n_ep) + 0.5, [h[ep == e, 4].mean() for e in range(n_ep)], color=BLUE, marker="o", ms=2.8)
    axs[1].set_ylabel(r"$\mathcal{L}_{\mathrm{ret}} = D_{\mathrm{KL}}$", fontsize=9); axs[1].set_xlabel("epoch")
    axs[1].set_title(r"…while harmless outputs barely move ($\hat{\mathbf{r}}$ removed)", fontsize=8.8)
    for ax in axs:
        ax.set_xlim(0, n_ep)
    fig.text(0.5, -0.13, "Thin lines: every step. Dots: epoch means.", ha="center", fontsize=7.2, color=INK2)
    save(fig, "rdo")


# ======================================================================== 5.2 COSMIC
def fig_cosmic():
    Sc = np.array(R5a["cosmic"]["Sc_cat"]); base = R5a["cosmic_baseline_cos"]["cat"]
    comp = {s["layer"]: 100 * s["comply"] for s in R2["sweep"]}
    ls = np.arange(1, 32); sc = Sc[-1, 1:32]
    rho = spearman(sc, [comp[l] for l in ls])
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    ax.plot(ls, sc, color=INK2, marker="o", ms=2.8)
    ax.axhline(base, color=INK2, ls="--", lw=0.9)
    ax.text(31, base - 0.012, "no intervention", fontsize=6.8, color=INK2, ha="right", va="top")
    ax.set_xlabel(r"layer $\ell$ of the last-token direction $\hat{\mathbf{r}}_\ell$")
    ax.set_ylabel(r"$S_{\mathrm{comply}}$", fontsize=9)
    ax.set_title("COSMIC's text-free score", fontsize=8.8)
    ax = axs[1]
    ax.scatter(sc, [comp[l] for l in ls], s=22, color=INK2, edgecolor=SURF, lw=0.8, zorder=3)
    for l in (11, 20, 22):
        ax.scatter([sc[l - 1]], [comp[l]], s=34, color=VIOLET, edgecolor=SURF, lw=0.8, zorder=4)
    ax.scatter([sc[22]], [comp[23]], s=34, color=MUTED, edgecolor=INK, lw=0.8, zorder=4)
    ax.annotate("23: the 2024 pick", (sc[22], comp[23]), xytext=(-8, 10), textcoords="offset points", fontsize=7,
                ha="right", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate("11, 20, 22", (sc[10], comp[11]), xytext=(-8, -2), textcoords="offset points", fontsize=7,
                ha="right", va="center")
    ax.text(0.03, 0.95, rf"Spearman $\rho = {rho:.2f}$", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlabel(r"$S_{\mathrm{comply}}$", fontsize=9); ax.set_ylabel("judged compliance (%)")
    ax.set_title("…only partly predicts real compliance", fontsize=8.8)
    save(fig, "cosmic")
    return rho


# ======================================================================== 5.6 head to head
def fig_methods():
    T = all_methods()
    rows = sorted(T.items(), key=lambda kv: kv[1]["harmful_comply"])
    fig, ax = plt.subplots(figsize=(W_FULL, 3.9))
    y = np.arange(len(rows))
    ax.barh(y, [100 * v["harmful_comply"] for _, v in rows], color=[MODEL_COL[family(k)] for k, _ in rows], height=0.66)
    bar_gap(ax)
    for yy, (k, v) in zip(y, rows):
        ax.text(100 * v["harmful_comply"] + 1.2, yy, f"{100 * v['harmful_comply']:.0f}%", va="center", fontsize=7.4,
                color=INK)
        ax.text(127, yy, f"{100 * v['harmless_comply']:.0f}%", va="center", ha="right", fontsize=7.2, color=INK2)
        ax.text(141, yy, f"{v['kl']:.2f}", va="center", ha="right", fontsize=7.2, color=INK2)
    ax.text(127, len(rows) - 0.2, "answers\nharmless", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.text(141, len(rows) - 0.2, r"$D_{\mathrm{KL}}$", ha="right", va="bottom", fontsize=8.4, color=INK2)
    ax.set_yticks(y, [label(k) for k, _ in rows], fontsize=7.4)
    ax.set_xlim(0, 142); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines["bottom"].set_bounds(0, 100)
    ax.set_ylim(-0.6, len(rows) + 0.9); ax.grid(axis="y", visible=False)
    ax.set_xlabel("complies with harmful requests (% of 104, judged)", loc="left")
    ax.legend(handles=[Patch(color=MODEL_COL[f], label=FAM_NAME[f]) for f in ["original", "2024", "paper", "guide"]],
              loc="lower left", bbox_to_anchor=(0.3, 0.02), fontsize=7.2)
    save(fig, "methods")


def fig_tradeoff():
    T = all_methods()
    short = {"original": "original", "dim_l23": "2024 recipe", "dim_l22": "best single layer", "cosmic": "COSMIC",
             "rdo": "RDO", "som sequential": "SOM, sequential", "som span-selected": "SOM, span-chosen",
             "som span": "SOM, span", "dim stack": "this guide"}
    off = {"original": (6, 2), "dim_l23": (6, -2), "dim_l22": (6, 0), "cosmic": (6, 0), "rdo": (-6, 4),
           "som sequential": (6, 0), "som span-selected": (6, 0), "som span": (6, -5), "dim stack": (6, 4)}
    fig, ax = plt.subplots(figsize=(W_FULL, 2.9))
    for k, v in T.items():
        f = family(k)
        ax.scatter(v["kl"], 100 * v["harmful_comply"], s=40, color=MODEL_COL[f], edgecolor=SURF, lw=1, zorder=3)
        p = next((p for p in short if k.lower().startswith(p)), None)
        if p and not k.lower().startswith(("basic", "uniform")):
            dx, dy = off[p]
            ax.annotate(short[p], (v["kl"], 100 * v["harmful_comply"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.2, ha="right" if dx < 0 else "left", va="center", color=INK)
    w = [v for k, v in T.items() if k.lower().startswith(("basic", "uniform"))]
    ax.annotate(r"WRDO family, $\alpha$ = 1, 2, 3 (6 runs)", (max(x["kl"] for x in w), 100 * max(x["harmful_comply"] for x in w)),
                xytext=(-150, 24), textcoords="offset points", fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlabel(r"$D_{\mathrm{KL}}$ on harmless prompts (lower = closer to the original)", fontsize=8.4)
    ax.set_ylabel("complies with harmful (%)"); ax.set_xlim(-0.02, 0.45); ax.set_ylim(-5, 100)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MODEL_COL[f], ms=6, label=FAM_NAME[f])
                       for f in ["original", "2024", "paper", "guide"]], loc="lower right", fontsize=7.2)
    save(fig, "tradeoff")


# ======================================================================== 6 audit
def fig_audit():
    A = json.load(open("results/audit.json")); s = np.array(A["shared_spectrum"])
    kinds = {}
    for e in A["changed"]:
        kinds[e["name"].split(".")[-2]] = kinds.get(e["name"].split(".")[-2], 0) + 1
    names = ["embed_tokens", "out_proj", "o_proj", "down_proj", "q/k/v_proj", "gate/up_proj", "norms"]
    vals = [kinds.get(n, 0) for n in names]
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})
    ax = axs[0]
    ax.barh(range(len(names))[::-1], vals, color=[VIOLET if v else GRID for v in vals], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.6, len(names) - 1 - i, str(v), va="center", fontsize=7.4, color=INK)
    ax.set_yticks(range(len(names))[::-1], names, fontsize=7.6); ax.set_xlim(0, 40)
    ax.set_title(f"Which tensors changed ({sum(vals)} of them)", fontsize=8.8); ax.grid(axis="y", visible=False)
    ax.text(39, 1.0, "writers only", fontsize=7.2, color=INK2, ha="right")
    ax = axs[1]
    k = A["shared_subspace_dim"]
    ax.bar(range(1, len(s) + 1), s, color=[VIOLET] * k + [MUTED] * (len(s) - k), width=0.66)
    bar_gap(ax)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 3); ax.set_xticks(range(1, len(s) + 1))
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_title("Singular values of all changes stacked", fontsize=8.8)
    ax.set_xlabel("singular value index")
    ax.text(k + 0.7, 0.3, f"{k} directions, then a\n{s[k - 1] / s[k]:.0f}× drop to noise", fontsize=7.2, color=INK2)
    save(fig, "audit")


if __name__ == "__main__":
    fig_rdo(); rho = fig_cosmic(); fig_methods(); fig_tradeoff(); fig_audit()
    json.dump({"cosmic_spearman": rho}, open("results/figs3_stats.json", "w"))
    print(f"COSMIC Spearman rho = {rho:.3f}")
