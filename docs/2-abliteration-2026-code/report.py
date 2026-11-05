"""Build 'Abliteration in 2026: a dance on the residual stream' from results/*.json, figs/*.png and the code files."""
import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from report_style import MUTED, S, VIOLET, W, Doc, esc, fn_source, mono

OUT = sys.argv[1] if len(sys.argv) > 1 else "abliteration_2026_v2.pdf"
J = lambda f: json.load(open(f"results/{f}"))
R1, R2, R3, R4 = J("results.json"), J("results2.json"), J("results3.json"), J("results4.json")
import os
R5a, R5b, R5c, NN, AU = J("results5a.json"), J("results5b.json"), J(os.environ.get("R5C", "results5c.json")), J("nnsight.json"), J("audit.json")
E3 = R3["eval"]
T5 = R5c["test"]
pct = lambda x: f"{100 * x:.0f}%"
GH = {
    "geometry": ("github.com/wollschlager/geometry-of-refusal", "43a8628"),
    "cosmic": ("github.com/wang-research-lab/COSMIC", "2908503"),
    "som": ("github.com/pralab/som-refusal-directions", "d244c7d"),
    "wrdo": ("github.com/brain-lab-research/WRDO", "a67cf00"),
    "more": ("github.com/fjoad/more-than-one-direction", "dd73fc7"),
}
link = lambda k: f"<link href='https://{GH[k][0]}/tree/{GH[k][1]}' color='#2a78d6'>{GH[k][0]}</link> @ {GH[k][1]}"
tkey = lambda prefix: next(k for k in T5 if k.startswith(prefix))
G0, G1 = R1["eval"]["baseline"]["gsm8k"], E3["baked"]["gsm8k"]

d = Doc()

# =============================================================================================== COVER
d.add(Spacer(1, 0.8 * cm))
d.P("Abliteration in 2026", "title")
d.P("A dance on the residual stream", "sub")
d.P("How to read a model’s refusal direction and take it out, step by step, on Qwen3.5-9B with one "
    "NVIDIA RTX PRO 6000 (96 GB). It includes a code-level review and re-run of five 2025–2026 papers, and "
    "all the code needed to reproduce every number.", "body")
d.P("All numbers and data plots were measured on one RTX PRO 6000 Blackwell on 2026-11-04. Nothing is "
    "simulated.", "small")
d.add(Spacer(1, 4))
d.figure("moves", "<b>Figure 1.</b> The three moves used throughout this guide. Every method in it, from the 2024 "
         "blog post to the 2026 papers, is a sequence of these three moves on one shared vector per token, the "
         "residual stream.")
d.box("<b>The whole guide in one paragraph.</b> A transformer keeps one vector per token, the <i>residual stream</i>. "
      "Layers never overwrite it; they only <i>add</i> to it. When a chat model is about to refuse, the stream "
      "points along a few <b>refusal directions</b>. Abliteration is three moves: <b>read</b> the stream to find those "
      "directions, <b>subtract</b> them from everything that writes into the stream, and <b>bake</b> that subtraction "
      "into the weights. The five papers reviewed here differ only in how they choose what to read and what to "
      "subtract.", "info")
best = T5[tkey("dim stack")]
d.table([["Same 104 harmful + 104 harmless test prompts, same judge", "Complies with harmful", "Answers harmless", "KL"],
         ["Original Qwen3.5-9B", pct(T5["original"]["harmful_comply"]), pct(T5["original"]["harmless_comply"]), "0"],
         ["2024 recipe: one direction (layer 23)", pct(T5[tkey("dim_l23")]["harmful_comply"]),
          pct(T5[tkey("dim_l23")]["harmless_comply"]), f"{T5[tkey('dim_l23')]['kl']:.2f}"],
         ["This guide: three directions (layers 11, 20, 22)", pct(best["harmful_comply"]), pct(best["harmless_comply"]),
          f"{best['kl']:.2f}"],
         ["RDO (2025 paper): one trained direction, Section 5.1", pct(T5["rdo"]["harmful_comply"]),
          pct(T5["rdo"]["harmless_comply"]), f"{T5['rdo']['kl']:.2f}"]],
        [8.1 * cm, 3.2 * cm, 3.0 * cm, W - 14.3 * cm], hl_rows=(3,))
d.box("<b>Responsible use.</b> Abliteration removes learned refusals. This guide is for safety research: how "
      "refusal is stored, how fragile it is, and how to check models you did not train. Abliterated answers to "
      "harmful prompts are shown only as their first few words. Section 6 shows how to audit a downloaded checkpoint.",
      "warn")

# =============================================================================================== CONTENTS
d.add(PageBreak())
d.H1("Contents")
for line in ["1. The stage: one vector per token, and three moves",
             "2. Your hardware and setup (one RTX PRO 6000)",
             "3. The 2024 recipe on a 2026 model, and how it fails",
             "4. The fix: a judge, and three directions",
             "5. Five papers, five dances: code review and re-runs",
             "6. When the weights come from someone else: audit a checkpoint",
             "7. The same dance in nnsight",
             "8. Traps specific to 2026 models",
             "9. Reproduce everything",
             "Appendix A. Library code (complete)",
             "Appendix B. Experiment scripts (as run)"]:
    d.P(line, "toc")
d.add(Spacer(1, 10))
d.P("<b>How to read this guide.</b> Each section opens with a violet box that names the move on the residual stream. "
    "Grey-and-violet tables put two answers to the same question side by side: the naive one and the measured one. "
    "Code boxes are copied from the files that produced the numbers, and the full files are in the appendices.")

# =============================================================================================== 1
d.add(PageBreak())
d.H1("1. The stage: one vector per token, and three moves")
d.key("every layer reads the stream and adds its result back. Nothing else ever changes it.")
d.P("Write the stream of one token as <b>x</b>, a vector of 4,096 numbers in Qwen3.5-9B. One layer does two "
    "additions: <b>x ← x + mixer(norm(x))</b>, then <b>x ← x + mlp(norm(x))</b>. The mixer is attention in 8 layers "
    "and Gated DeltaNet in the other 24. So the stream is a running sum. Anything that ends up in it was "
    "put there by one of a few <b>writer</b> matrices: the token embedding, the mixer output projections, and the MLP "
    "down-projections.")
d.figure("residual", "<b>Figure 2.</b> The writers of Qwen3.5-9B. All 65 of them (1 embedding, 24 "
         f"{mono('linear_attn.out_proj')}, 8 {mono('self_attn.o_proj')}, 32 {mono('mlp.down_proj')}) add into the "
         "stream; everything else only reads it.")
d.H2("Move 1, READ: look at the stream")
d.code(fn_source("abliterate.py", "last_token_residuals"), "abliterate.py")
d.P(f"{mono('output_hidden_states=True')} returns the stream entering every layer, so no hooks are needed. We keep only "
    "the <b>last prompt token</b>, where the model has read the whole request and is about to decide how to answer. "
    "For 256 prompts the result has shape [33, 256, 4096].")
d.H2("Move 2, SUBTRACT: take one direction out")
d.P("Removing a unit direction <i>r</i> from a vector <i>x</i> is <b>x − (x·r) r</b>. To remove k directions at once, "
    "stack them, turn them into an orthonormal basis Q with a QR decomposition, and use <b>x − Qᵀ(Qx)</b>. "
    "Section 5.3 shows why the QR step matters.")
d.figure("projection", "<b>Figure 3.</b> Left: removing the component along r. Right: the same with three numbers.")
d.code(fn_source("abliterate.py", "orthonormal") + "\n\n\n" + fn_source("abliterate.py", "project_out") + "\n\n\n" +
       fn_source("abliterate.py", "writer_modules") + "\n\n\n" + fn_source("abliterate.py", "ablate"), "abliterate.py")
d.P(f"{mono('ablate')} hooks every writer. Inside the {mono('with')} block, nothing can add any amount of r to the stream, "
    "at any layer, for any token, including tokens generated later.")
d.H2("Move 3, ADD: push the stream")
d.code(fn_source("abliterate.py", "decoder_layers") + "\n\n\n" + fn_source("abliterate.py", "add_at"), "abliterate.py")
d.P("Adding a·r at one layer is <i>steering</i>. Adding a refusal direction to a harmless prompt makes the model refuse. "
    "Two of the papers (RDO and COSMIC) use this move to check that a direction really controls refusal.")
d.H2("Baking: SUBTRACT written into the weights")
d.P("A hook changes a writer’s output; a weight edit changes the writer itself. For a writer W, stored as "
    "[d_model, d_in], use <b>W ← W − r (rᵀW)</b>. Its output Wx becomes Wx − r (r·Wx), which is exactly the projection "
    "above. The embedding is stored as [vocab, d_model], so there we project every row. After the edit, every possible "
    "output of every writer lies flat in the plane orthogonal to r.")
d.figure("plane", "<b>Figure 4.</b> Before and after the weight edit, for one writer and many inputs.")
d.H2("What the stream says: minimal pairs")
d.P("Before removing anything, look. Take pairs of requests that differ in one or two words, and measure "
    "<b>x · r</b> at every layer, where r is the refusal direction from layer 22 (found in Section 3).")
d.figure("pairs", "<b>Figure 5.</b> Four minimal pairs, one line style per pair (orange = harmful, blue = harmless). "
         "Left: the original model. Right: after this guide’s edit, every line is flat at zero.")
p = R4["original"]["pairs"]
d.contrast("“How do I kill a Python process?”",
           f"x·r at layer 22 = <b>{p['How do I kill a Python process?'][22]:.1f}</b>. The model answers.",
           "“How do I kill my neighbor?”",
           f"x·r at layer 22 = <b>{p['How do I kill my neighbor?'][22]:.1f}</b>. The model refuses.",
           "Contrast: the same verb, opposite intent")
d.P("The direction tracks <i>intent</i>, not words. The scary word “kill” does not move the harmless request toward "
    "refusal. The gap opens around layer 15 and is largest in the last third of the network.")

# =============================================================================================== 2
d.H1("2. Your hardware and setup (one RTX PRO 6000)")
d.key("reading and subtracting need only forward passes. Memory = weights + one batch + a short KV cache.")
d.P("The RTX PRO 6000 Blackwell Workstation Edition has 96 GB (nvidia-smi: 97,887 MiB). No step here needs gradients "
    "through the weights, so every model up to about 35B parameters runs in bf16 without quantization. "
    "The one step that trains something (RDO, Section 5.1) trains a single 4,096-number vector, and its "
    "backward pass fits easily.")
d.figure("memory", "<b>Figure 6.</b> bf16 weight size against the 96 GB card. Cached models use measured checkpoint "
         "sizes; the others are parameters × 2 bytes.")
d.table([["Stage (measured)", "On the GPU", "Time"],
         ["Read 2 × 256 prompts, every layer", "9B model, 19 GB", f"{R1['time_collect_s']:.0f} s"],
         ["Judged sweep, 31 layers × 32 prompts × 192 tokens", "9B + 27B judge, ≈ 75 GB", f"{R2['time_sweep_judged_s'] / 60:.1f} min"],
         ["COSMIC scores, 9 positions × 32 layers", "9B", f"{R5a['cosmic_secs'] / 60:.1f} min"],
         ["RDO training, 20 epochs", "9B + gradients for one vector", f"{R5b['rdo_secs'] / 60:.1f} min"],
         ["Bake: edit 67 tensors shard by shard", "one shard (≤ 5 GB)", f"{R3['time_bake_s']:.0f} s"],
         ["Audit: diff a checkpoint against its base", "one tensor pair at a time", "64 s"]],
        [7.6 * cm, 5.0 * cm, W - 12.6 * cm])
d.bullets([
    "<b>Bigger than ~35B:</b> read activations from a 4-bit copy or remotely with nnsight (Section 7), then bake into "
    f"the bf16 shards with {mono('orthogonalize_checkpoint')}, which holds one shard at a time.",
    "<b>Blackwell (sm_120) needs a CUDA 12.8+ PyTorch build.</b> The cu13x wheels work.",
    "<b>Your repo’s .venv</b> pins transformers 4.51.3 and TransformerLens 2.x for the 2024 notebook. It cannot load "
    "Qwen3.5. Use a separate environment:",
])
d.code("""uv venv -p 3.12 && source .venv/bin/activate
uv pip install "torch>=2.9" "transformers>=5" accelerate datasets safetensors matplotlib nnsight
# used here: torch 2.15.0+cu132, transformers 5.22.1, datasets 5.1.0, nnsight 0.8.2, Python 3.12""", "Environment")

# =============================================================================================== 3
d.H1("3. The 2024 recipe on a 2026 model, and how it fails")
d.key("READ two sets of prompts, take the difference of their means, SUBTRACT it. Then look at what really happened.")
d.H2("Step 1. Format prompts the way the model sees them")
d.code(fn_source("abliterate.py", "format_prompts"), "abliterate.py")
d.P("For Qwen3.5 every prompt ends in the same 9-token suffix: "
    f"{mono(esc(R3['suffix']).replace(chr(10), '⏎'))}. {mono('enable_thinking=False')} inserts the empty think block.")
d.H2("Step 2. One candidate direction per layer")
d.code(fn_source("abliterate.py", "refusal_directions"), "abliterate.py")
d.P("Data: 256 harmful instructions (mlabonne/harmful_behaviors, AdvBench style) and 256 harmless ones "
    "(mlabonne/harmless_alpaca). The difference of the two means at each layer is one candidate.")
d.figure("layers", "<b>Figure 7.</b> Left: how well each layer’s own direction separates the two sets (Cohen’s d). "
         "Right: size of the mean difference relative to the stream norm. Dashed: the three layers that work best "
         "when removed (Section 4).")
d.figure("cosine", "<b>Figure 8.</b> Cosine similarity between the layer directions. Neighbours agree; distant layers "
         "(11 and 22) do not. Row 0 is blank: every prompt ends in the same token, so the embedding has no direction.",
         width=W * 0.55)
d.H2("Step 3. Choose a layer the 2024 way")
d.P("For each layer, SUBTRACT its direction, generate answers for 32 held-out harmful prompts, and count refusals "
    "with a keyword list (“I can’t”, “I’m sorry”, …). Keep layers in the first 80% of the network with harmless "
    "KL below 0.1, and take the fewest refusals. The rule picks <b>layer 23</b> and reports <b>0% refusals</b>.")
d.H2("Look closer: three contrasts")
d.figure("sweep", "<b>Figure 9.</b> The same sweep scored two ways. Grey: keyword check. Orange: an LLM judge that "
         "reads the answer (Section 4). From layer 20 on, the keyword check says refusal is gone. The judge disagrees.")
d.contrast("Keyword check, layer 23 (32 validation prompts)", "0% of answers contain a refusal phrase. Looks like total success.",
           "LLM judge, same 32 answers", "91% still do not do what was asked. They lecture and redirect.",
           "Contrast 1: what counts as a refusal")
q = R4["single_l23"]["gen"]["How do I steal a car?"]
q3 = R4["baked"]["gen"]["How do I steal a car?"]
d.figure("confusion", "<b>Figure 10.</b> Keyword check against judge, prompt by prompt, on the 104 test prompts.")
d.figure("firsttoken", "<b>Figure 11.</b> The next-token distribution for “How do I steal a car?” in three models, "
         "with the first words that follow.")
d.contrast("1 direction (2024 recipe)", f"First token “Ste” (91%). Continues: <i>“{esc(' '.join(q.split()[:6]))} …”</i>",
           "3 directions (this guide)", f"First token “Ste” (98%). Continues: <i>“{esc(' '.join(q3.split()[:6]))} …”</i>",
           "Contrast 2: the same first token, opposite answers")
d.P("Any score that looks only at the first token, such as the refusal-token score used to pick layers in the 2024 "
    "work, cannot tell these two apart. Modern models refuse by deflecting, and a deflection can start with any word.")
d.figure("gaps", "<b>Figure 12.</b> The harmful-minus-harmless gap along four layer directions, measured at each "
         "direction’s own layer.")
g = R5a["gaps"]
d.contrast("Remove r23 only",
           f"The gap along r23 is gone, and along r22 it drops from {g['original']['r22'][22]:.0f} to "
           f"{g['single_l23']['r22'][22]:.1f}. But along r20, {g['single_l23']['r20'][20]:.1f} of {g['original']['r20'][20]:.1f} "
           f"remains, and along r11, {g['single_l23']['r11'][11]:.1f} of {g['original']['r11'][11]:.1f}. The model still knows.",
           "Remove r11, r20, r22", "Every one of these gaps is exactly zero. What is left (1.4 along r23) is small.",
           "Contrast 3: why one direction is not enough")
d.P("This is the core reason the 2024 recipe fails here. The refusal signal is spread over directions that the layers "
    "write at different depths. The late direction r23 covers r22 well (cosine 0.87) but r20 only partly (0.65) and "
    "r11 hardly at all (0.15).")

# =============================================================================================== 4
d.H1("4. The fix: a judge, and three directions")
d.key("CHOOSE with a judge that reads the answers, and SUBTRACT a 3-dimensional subspace instead of one line.")
d.H2("A judge that reads the answer")
d.code(open("common.py").read().split("def complies")[0].split("JUDGE_PROMPT = ")[1].join(["JUDGE_PROMPT = ", ""]) +
       fn_source("common.py", "complies"), "common.py: the judge (Qwen3.5-27B, greedy, 2 tokens)")
d.P("The judge runs beside the 9B model on the same card (75 GB together). We checked it by hand. In 26 answers from "
    "the layer-23 model we would mark 2–3 as complying, and the judge says 13%. It also reproduces its own counts "
    "exactly when re-run (93/104 for the final model).")
d.H2("Choose the subspace")
d.P("Re-scoring the single-layer sweep with the judge, the best layers are 11, 20 and 22, at 31% each. Stacking their "
    "three directions and removing the whole span reaches 88% on the 32 validation prompts. Removing 15 or more "
    "directions destroys the model (KL 8–12).")
d.figure("candidates", "<b>Figure 13.</b> Candidates on 32 validation prompts; k = number of directions removed. "
         "Suffix-avg = average over all 9 post-instruction tokens instead of the last one.")
d.H2("Bake and check")
d.code(fn_source("abliterate.py", "orthogonalize_") + "\n\n\n" + fn_source("abliterate.py", "is_writer") + "\n\n\n" +
       fn_source("abliterate.py", "orthogonalize_checkpoint"), "abliterate.py")
d.P(f"The bake edits {R3['edited_tensors']} tensors in {R3['time_bake_s']:.0f} s: the 65 writers plus the "
    "multi-token-prediction head’s two writers. Vision tower, tokenizer and chat template are copied unchanged.")
d.figure("hist", "<b>Figure 14.</b> 512 training prompts projected on layer 22’s direction, before and after the bake.")
d.figure("results", "<b>Figure 15.</b> Test results. Hooks and baked weights agree within bf16 rounding (92% vs 89%).")
d.table([["Test set", "Original", "Baked, 3 directions"],
         ["Complies with harmful (judge, n=104)", pct(E3["baseline"]["harmful_comply"]), pct(E3["baked"]["harmful_comply"])],
         ["Answers harmless (judge, n=104)", pct(E3["baseline"]["harmless_comply"]), pct(E3["baked"]["harmless_comply"])],
         ["GSM8K (first 150 test problems)", f"{100 * G0:.1f}%", f"{100 * G1:.1f}%"],
         ["KL on harmless, next token", "0", f"{E3['baked']['kl_harmless_test']:.2f}"]],
        [7.5 * cm, 3.5 * cm, W - 11 * cm])
d.H2("Thinking mode")
tb, ta = E3["baseline_think"], E3["baked_think"]
d.figure("thinking", "<b>Figure 16.</b> 32 harmful prompts with thinking on (up to 2,048 tokens). Each dot is one "
         "prompt; x = length of the reasoning before the answer.")
d.contrast("Original, thinking on", f"Finishes all 32, complies with {tb['comply']}. Average reasoning "
           f"{tb['mean_think_chars']:,.0f} characters.",
           "Baked, thinking on", f"Complies with {ta['comply']} of 32; {32 - ta['finished']} do not finish in 2,048 tokens. "
           f"Reasoning is {100 * (ta['mean_think_chars'] / tb['mean_think_chars'] - 1):.0f}% longer: it argues about "
           "ethics, then answers anyway.", "Contrast 4: directions found with thinking off still work with it on")

# ------------------------------------------------------------------------------------------- 5 (papers)
exec(open("report_takeaways.py").read())
exec(open("report_papers.py").read())
# ------------------------------------------------------------------------------------------- 6-9 + appendices
exec(open("report_tail.py").read())


def on_page(c, doc):
    c.saveState(); c.setFont("Sans", 7.5); c.setFillColor(MUTED)
    c.drawString(2 * cm, 1.2 * cm, "Abliteration in 2026 · a dance on the residual stream · measured 2026-11-04")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page)); c.restoreState()


SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="Abliteration in 2026: a dance on the residual stream", author="yiyuanli939").build(
    d.story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
