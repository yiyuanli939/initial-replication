"""Make every figure for the PDF. Schematics are drawn; data figures read results/results.json."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG = Path("figs"); FIG.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GOOD, CRIT = "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=AXIS, color=INK, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight)


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw, linestyle=ls))


# ------------------------------------------------------------------ schematics
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 1.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
    steps = [("1  Read", "hidden states\nof 512 prompts"),
             ("2  Subtract", "mean(harmful)\n− mean(safe)"),
             ("3  Sweep", "score each\nlayer's r"),
             ("4  Bake", "edit writer\nweights on disk"),
             ("5  Check", "refusals, KL,\nGSM8K")]
    w, gap = 1.72, 0.3
    for i, (t, s) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        fc = "#eef4fc" if i != 3 else "#fdf0ea"
        ec = BLUE if i != 3 else ORANGE
        ax.add_patch(FancyBboxPatch((x, 0.1), w, 1.35, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 0.12, 1.18, t, fontsize=10, weight="bold", va="center")
        ax.text(x + 0.12, 0.6, s, fontsize=7.8, color=INK2, va="center")
        if i < 4:
            arrow(ax, (x + w + 0.02, 0.78), (x + w + gap - 0.02, 0.78))
    save(fig, "pipeline")


def fig_residual():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    # the stream
    ax.add_patch(FancyBboxPatch((4.55, 0.25), 0.9, 4.75, boxstyle="round,pad=0,rounding_size=0.15", fc="#eef4fc", ec=BLUE, lw=1.2))
    ax.text(5.0, 5.08, "residual stream  (d = 4096 for Qwen3.5-9B)", ha="center", fontsize=8.5, color=BLUE, weight="bold")
    rows = [(0.55, "embed_tokens\n(token ids → vectors)", "", True),
            (1.45, "mixer: linear_attn.out_proj\nor self_attn.o_proj", "", True),
            (2.35, "MLP: mlp.down_proj", "", True),
            (3.25, "…  × 32 layers  …", "", None),
            (4.15, "final norm → lm_head", "reads the stream", False)]
    for y, name, what, writer in rows:
        if writer is None:
            ax.text(5.0, y + 0.1, name, ha="center", fontsize=8.5, color=MUTED); continue
        if writer:
            box(ax, 6.2, y - 0.05, 3.1, 0.62, name, fc="#fdf0ea", ec=ORANGE, size=8.3)
            arrow(ax, (6.18, y + 0.26), (5.48, y + 0.26), color=ORANGE)
            ax.text(5.0, y + 0.26, "+", ha="center", va="center", fontsize=13, color=ORANGE, weight="bold")
            # reader into the block
        else:
            box(ax, 6.2, y - 0.05, 3.1, 0.62, name, fc="#ffffff", ec=AXIS, size=8.3)
            arrow(ax, (5.48, y + 0.26), (6.18, y + 0.26), color=MUTED)
    ax.text(0.1, 4.35, "Writers (orange)", fontsize=10, weight="bold", color=INK)
    ax.text(0.1, 3.95, "are the only weights whose output\nis ADDED into the stream.\n\n"
                       "Remove r from every writer and\nthe stream can never contain r —\n"
                       "at any layer, for any prompt.\n\nEverything else only READS the\nstream and needs no edit.", fontsize=8.5, color=INK2, va="top")
    ax.text(0.1, 0.55, "Edit per writer:  W ← W − r (rᵀW)", fontsize=9, color=ORANGE, weight="bold")
    save(fig, "residual")


def fig_projection():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax in axs:
        ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    # left: geometry
    ax = axs[0]; ax.set_xlim(-0.3, 3.3); ax.set_ylim(-0.4, 2.6)
    r = np.array([1, 0.0]); x = np.array([2.4, 1.9]); proj = (x @ r) * r
    ax.annotate("", xy=(3.1, 0), xytext=(-0.2, 0), arrowprops=dict(arrowstyle="-", color=AXIS, lw=1, ls="--"))
    ax.annotate("", xy=r, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4))
    ax.text(0.5, -0.25, "r  (unit refusal direction)", color=ORANGE, fontsize=8.5, ha="left")
    ax.annotate("", xy=x, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(x[0] + 0.05, x[1] + 0.08, "x  (activation)", fontsize=8.5)
    ax.annotate("", xy=proj, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, alpha=0.6))
    ax.plot([x[0], proj[0]], [x[1], proj[1]], ls=":", color=MUTED, lw=1.2)
    ax.annotate("", xy=(0, x[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.text(0.08, x[1] + 0.1, "x − (x·r) r", color=BLUE, fontsize=8.5, weight="bold")
    ax.text(proj[0] - 0.55, 0.12, "(x·r) r", color=ORANGE, fontsize=8.5)
    ax.set_title("Remove one component, keep the rest", fontsize=9.5)
    # right: concrete numbers in 3-D shown as bars
    ax = axs[1]; ax.set_aspect("auto"); ax.set_xlim(-0.6, 5.6); ax.set_ylim(-0.3, 3.3)
    vals_x = [2.0, 1.0, 0.5]; vals_after = [0.0, 1.0, 0.5]
    labels = ["refusal\naxis", "dim 2", "dim 3"]
    for i, (a, b) in enumerate(zip(vals_x, vals_after)):
        ax.bar(i * 2 - 0.25, a, 0.45, color=INK2, alpha=0.85)
        ax.bar(i * 2 + 0.25, b, 0.45, color=BLUE)
        ax.text(i * 2 - 0.25, a + 0.07, f"{a:g}", ha="center", fontsize=8, color=INK2)
        ax.text(i * 2 + 0.25, b + 0.07, f"{b:g}", ha="center", fontsize=8, color=BLUE)
        ax.text(i * 2, -0.25, labels[i], ha="center", va="top", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.text(-0.5, 3.0, "x = [2, 1, 0.5],  r = [1, 0, 0]", fontsize=8.5)
    ax.text(-0.5, 2.65, "x·r = 2  →  x − 2r = [0, 1, 0.5]", fontsize=8.5, color=BLUE)
    ax.set_title("Same thing with numbers", fontsize=9.5)
    save(fig, "projection")


def fig_memory():
    # (label, GB of bf16-equivalent weights, note). Cached sizes measured from the local HF cache.
    rows = [("Qwen3.5-4B", 9.3, "measured"), ("Llama-3.1-8B-Instruct", 16.1, "measured"),
            ("Qwen3.5-9B  ← this guide", 19.3, "measured"), ("gpt-oss-20b (MXFP4 → bf16)", 41.8, "21B params"),
            ("Qwen3.5-27B", 55.6, "measured"), ("Qwen3-30B-A3B (MoE)", 61.1, "30.5B params"),
            ("Gemma-4-31B-it", 62.5, "measured"), ("Llama-3.3-70B-Instruct", 141.1, "70.6B params")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = [r[0] for r in rows][::-1]; gb = [r[1] for r in rows][::-1]
    head = 96 - 12
    cols = [BLUE if g <= head else CRIT for g in gb]
    ax.barh(names, gb, color=cols, height=0.62)
    for i, g in enumerate(gb):
        ax.text(min(g, 148) + 1.5, i, f"{g:.0f} GB", va="center", fontsize=8, color=INK2)
    ax.axvline(96, color=INK, lw=1.4); ax.text(97.5, 3.0, "96 GB\nRTX PRO 6000", fontsize=8, weight="bold")
    ax.axvline(head, color=INK2, lw=1, ls="--"); ax.text(head - 1.5, 6.5, "~12 GB headroom for\nactivations + KV cache", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlim(0, 168); ax.set_xlabel("bf16 weights (GB)"); ax.grid(axis="y", visible=False)
    ax.set_title("What fits on one 96 GB card for the whole pipeline in bf16")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="fits: whole pipeline on GPU"),
                       Patch(color=CRIT, label="too big: 4-bit read + shard-wise bake")],
              loc="upper right", fontsize=7.8)
    save(fig, "memory")


# ------------------------------------------------------------------ data figures
def data_figs(R):
    L, best = R["n_layers"], R["best"]["layer"]
    layers = np.arange(L + 1)

    # separation per layer
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axs[0].plot(layers[:L], R["layer_sep_cohens_d"][:L], color=BLUE)
    for lb in (11, 20, 22):
        axs[0].axvline(lb, color=ORANGE, lw=1, ls="--")
    axs[0].text(22.6, max(R["layer_sep_cohens_d"][:L]) * 0.12, "best single\nlayers", color=ORANGE, fontsize=7.5)
    axs[0].set_title("How well r separates the two sets"); axs[0].set_xlabel("layer"); axs[0].set_ylabel("Cohen's d of projection")
    axs[1].plot(layers[:L], np.array(R["layer_rel_diff_norm"][:L]), color=BLUE)
    for lb in (11, 20, 22):
        axs[1].axvline(lb, color=ORANGE, lw=1, ls="--")
    axs[1].set_title("Size of the mean difference"); axs[1].set_xlabel("layer"); axs[1].set_ylabel("‖μ_harm − μ_safe‖ / ‖x‖")
    fig.tight_layout(); save(fig, "layers")

    # scatter
    S = R["scatter"]; n = S["n_harmful"]; x, y = np.array(S["x"]), np.array(S["y"])
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.scatter(x[n:], y[n:], s=12, color=BLUE, alpha=0.7, edgecolor=SURF, lw=0.4, label="harmless (Alpaca)")
    ax.scatter(x[:n], y[:n], s=12, color=ORANGE, alpha=0.7, edgecolor=SURF, lw=0.4, label="harmful (AdvBench)")
    ax.set_xlabel("projection on r  (x · r)"); ax.set_ylabel("top other direction (PC1 ⊥ r)")
    ax.set_title(f"256 + 256 prompts at layer {best}, last token")
    ax.legend(loc="lower right", fontsize=8, markerscale=1.6, frameon=True, facecolor=SURF, edgecolor=GRID)
    save(fig, "scatter")

    # cosine matrix
    C = np.array(R["dir_cosine_matrix"])
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", BLUE])
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1); ax.grid(False)
    ax.set_xlabel("layer"); ax.set_ylabel("layer"); ax.set_title("cos(r_i, r_j)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "cosine")



def judged_figs(R1, R2, R3):
    L = R1["n_layers"]
    sw = R2["sweep"]; ls_ = np.array([x["layer"] for x in sw])
    kw = np.array([100 * x["refusal"] for x in sw]); jd = np.array([100 * (1 - x["comply"]) for x in sw])
    kl = np.array([x["kl"] for x in sw])

    # 1) sweep: keyword vs judge
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axs[0]
    ax.plot(ls_, kw, color=MUTED, marker="o", ms=3.5, label="keyword check says “refused”")
    ax.plot(ls_, jd, color=ORANGE, marker="o", ms=3.5, label="LLM judge says “did not comply”")
    ax.fill_between(ls_, kw, jd, where=jd > kw, color=ORANGE, alpha=0.12, lw=0)
    ax.annotate("layer 23: keywords say 0% refusals,\njudge says 91% still not complying", xy=(23, jd[ls_ == 23][0]),
                xytext=(3.0, 35), fontsize=7.8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(-4, 108); ax.set_ylabel("% of 32 harmful prompts")
    ax.set_title("Single-direction sweep on Qwen3.5-9B: the keyword metric is fooled by deflections")
    ax.legend(loc="lower left", fontsize=7.8)
    ax = axs[1]
    ax.plot(ls_, kl, color=BLUE, marker="o", ms=3.5)
    ax.axhline(0.1, color=INK2, ls="--", lw=1); ax.text(1, 0.105, "KL 0.1", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylabel("KL on harmless"); ax.set_xlabel("layer whose direction is ablated")
    fig.tight_layout(); save(fig, "sweep")

    # 2) candidates
    C = R3["candidates"]
    rows = [("single, layer 23 (keyword pick)", 1 - jd[ls_ == 23][0] / 100, kl[ls_ == 23][0], 1),
            ("single, layer 22 (judge pick)", 1 - jd[ls_ == 22][0] / 100, kl[ls_ == 22][0], 1)]
    rows += [(c["name"], c["comply"], c["kl"], c["k"]) for c in C]
    best = R3["best"]["name"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))[::-1]
    cols = [ORANGE if r[0] == best else (MUTED if r[0].startswith("single") else BLUE) for r in rows]
    ax.barh(y, [100 * r[1] for r in rows], color=cols, height=0.62)
    for yy, r in zip(y, rows):
        ax.text(100 * r[1] + 1.2, yy, f"{100 * r[1]:.0f}%   KL {r[2]:.2f}   k={r[3]}", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of 32 held-out harmful prompts the judge marks as complied")
    ax.grid(axis="y", visible=False)
    ax.set_title("What actually removes refusal: more positions and more layers (k = directions removed)")
    save(fig, "candidates")

    # 3) final results
    E = R3["eval"]
    tags = [("baseline", "original", INK2), ("single_l23", "1 dir (L23)", MUTED), ("hooked_best", "best, hooks", BLUE), ("baked", "best, baked", ORANGE)]
    panels = [("harmful_comply", "Complies with harmful\n(judge, n=104)"), ("harmful_kw_refusal", "“Refuses” harmful\n(keywords, n=104)"),
              ("harmless_comply", "Complies with harmless\n(judge, n=104)")]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [4, 4, 4, 2.4]})
    for ax, (k, title) in zip(axs, panels):
        v = [100 * E[t][k] for t, _, _ in tags]
        ax.bar(range(4), v, color=[c for _, _, c in tags], width=0.66)
        for i, val in enumerate(v): ax.text(i, val + 2, f"{val:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(4), [n for _, n, _ in tags], rotation=35, ha="right", fontsize=7.4)
        ax.set_ylim(0, 112); ax.set_title(title, fontsize=8.6, loc="center"); ax.grid(axis="x", visible=False)
    ax = axs[3]
    v = [100 * R1["eval"]["baseline"]["gsm8k"], 100 * E["baked"]["gsm8k"]]
    ax.bar([0, 1], v, color=[INK2, ORANGE], width=0.66)
    for i, val in enumerate(v): ax.text(i, val + 2, f"{val:.0f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["original", "best, baked"], rotation=35, ha="right", fontsize=7.4)
    ax.set_ylim(0, 112); ax.set_title("GSM8K\n(n=150)", fontsize=8.6, loc="center"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "results")

    # 4) histogram before/after at layer 22
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    pb, pa = R3["proj_before_l22"], R3["proj_after_l22"]
    allv = pb["harmful"] + pb["harmless"] + pa["harmful"] + pa["harmless"]
    bins = np.linspace(min(allv) - 0.5, max(allv) + 0.5, 45)
    for ax, P, title in [(axs[0], pb, "original model"), (axs[1], pa, "abliterated (baked) model")]:
        ax.hist(P["harmless"], bins=bins, color=BLUE, alpha=0.75, label="harmless")
        ax.hist(P["harmful"], bins=bins, color=ORANGE, alpha=0.75, label="harmful")
        ax.set_title(title); ax.set_xlabel("x · r₂₂ at layer 22, last token")
    axs[0].set_ylabel("prompts"); axs[0].legend(fontsize=8)
    fig.tight_layout(); save(fig, "hist")


if __name__ == "__main__":
    fig_pipeline(); fig_residual(); fig_projection(); fig_memory()
    R1 = json.load(open("results/results.json"))
    data_figs(R1)
    if Path("results/results3.json").exists() and "eval" in json.load(open("results/results3.json")):
        judged_figs(R1, json.load(open("results/results2.json")), json.load(open("results/results3.json")))
