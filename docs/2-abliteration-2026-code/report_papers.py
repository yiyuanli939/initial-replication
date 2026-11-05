# Section 5 of report.py (exec'd in its namespace).
from reportlab.lib.units import cm as _cm

d.H1("5. Five papers, five dances: code review and re-runs")
d.key("every paper below changes only two things: what it READs to build directions, and which directions it "
      "SUBTRACTs (or ADDs). The stream and the writers stay the same.")
d.P("For each paper we read the paper and its released code, then re-implemented the core method on top of the three "
    "moves from Section 1 and ran it on Qwen3.5-9B. Every method is scored on the same 104 + 104 test prompts with the "
    "same judge, the same generation settings and the same fp32 projection code. Code findings marked "
    "<b>checked</b> were confirmed by us at the cited lines. Findings marked <b>reported</b> come from our code read "
    "and were not re-checked by hand.")
d.table([["Method", "READ", "CHOOSE", "SUBTRACT / ADD"],
         ["Arditi et al. 2024 (the 2024 recipe)", "means at the last tokens", "one layer, refusal-token score + KL",
          "r from every writer"],
         ["5.1 RDO (Wollschläger et al., ICML 2025)", "only to make training targets",
          "gradient descent on r: answer when r is removed, refuse when r is added, keep KL low", "the learned r"],
         ["5.2 COSMIC (Siu et al., ACL Findings 2025)", "the 10% of layers where harmful ≠ harmless",
          "SUBTRACT on harmful + ADD on harmless; pick r whose activations “invert” best", "r from every writer"],
         ["5.3 SOM (Piras et al., AAAI 2026)", "the cloud of harmful activations at one layer",
          "16 cluster centres − harmless mean; search for the best k", "k directions"],
         ["5.4 WRDO (Evseev et al., EMNLP Findings 2026)", "one direction per harmful prompt, every layer",
          "weights over the prompts, black-box search on a judge score", "per-layer r, per-layer strength"],
         ["5.5 More than one direction (Joad et al., EMNLP 2026)", "one mean difference per refusal category",
          "none: compares categories", "ADD each category’s direction"]],
        [4.4 * _cm, 3.5 * _cm, 5.4 * _cm, W - 13.3 * _cm])

# ------------------------------------------------------------------------------------------------ 5.1 RDO
d.H2("5.1 The Geometry of Refusal: learn the direction (RDO)")
d.P(f"Wollschläger, Elstner, Geisler, Cohen-Addad, Günnemann, Gasteiger. ICML 2025 (PMLR 267). Code: {link('geometry')}.", "small")
d.key("instead of READing r as a difference of means, train r so that SUBTRACT makes the model answer and ADD makes "
      "it refuse, while harmless outputs stay unchanged.")
d.P("<b>Idea.</b> A difference of means is a guess. RDO treats r as a 4,096-number parameter and optimizes it with three "
    "losses: (1) with r removed, the model should produce answer text on harmful prompts; (2) with α·r added at one "
    "layer, it should produce refusal text on harmless prompts; (3) with r removed, harmless next-token distributions "
    "should not move (KL). The paper extends this to <b>cones</b>, k-dimensional regions where every direction "
    "mediates refusal, and to <b>representational independence</b>, directions that do not change each other’s "
    "activations when removed.")
d.H3("What the code does (checked)")
d.bullets([
    f"The retain loss is computed as {mono('kl_div(log_softmax(base), softmax(ablated))')}, which is "
    "KL(ablated‖original). The paper writes KL(original‖ablated). With batch size 1 and "
    f"{mono('batchmean')}, it is also <b>summed</b> over 30 target tokens instead of averaged, so the retain weight is "
    f"about 30× what the formula says ({mono('rdo.py:516-525, 753')}).",
    f"Command-line flags such as {mono('--lr')}, {mono('--n_sample')} and the loss weights are parsed but not "
    f"passed to training. Only {mono('cone_dim')} and {mono('orthogonal_vectors')} are ({mono('rdo.py:966-970')}). "
    f"The README’s {mono('--n_sample 16')} runs with the default 8.",
    f"Pinned to nnsight 0.3.7, transformers 4.47 and torch 2.5.1 ({mono('requirements.txt')}). It hard-codes "
    f"{mono('layer.self_attn')}, so it does not run on Gated-DeltaNet layers without porting.",
])
d.H3("Our version: rdo.py")
d.P("We kept the method and changed the plumbing. Ablation and addition use the hooks from Section 1. The retain KL "
    "follows the paper (forward, token mean). Batches hold 16 right-padded prompts. LR cuts follow the "
    "<i>epoch-mean</i> loss, because per-batch loss is too noisy. One change matters on this model: the paper makes "
    "answer targets by removing the difference-of-means direction, and on Qwen3.5 that produces deflections (Section 3). "
    "Training toward deflections would teach r to deflect. So we made the answer targets with the 3-direction model "
    "from Section 4. The paper allows this: “any effective attack can work”.")
d.code(fn_source("rdo.py", "rdo_step_loss") + "\n\n\n" + fn_source("rdo.py", "train_rdo"), "rdo.py (pack() and the loss helpers are in Appendix A)")
d.figure("rdo", f"<b>Figure 17.</b> RDO training on Qwen3.5-9B: 20 epochs of 16 batches, {R5b['rdo_secs']:.0f} s. "
         f"The two cross-entropies fall from about 2 to 0.3; retain KL stays below {max(x[4] for x in R5b['rdo_hist']):.2f}.")
r = T5["rdo"]
d.contrast("Difference of means, best single layer (22)",
           f"Complies {pct(T5[tkey('dim_l22')]['harmful_comply'])}, KL {T5[tkey('dim_l22')]['kl']:.2f}.",
           "RDO, one learned direction",
           f"Complies {pct(r['harmful_comply'])}, KL {r['kl']:.2f}. Its cosine with any layer’s mean-difference direction "
           f"is only {min(R5b['rdo_cos_to_dim'].values()):.2f}–{max(R5b['rdo_cos_to_dim'].values()):.2f}.",
           "Contrast: one direction, guessed vs trained")
d.P(RDO_TAKEAWAY)

# ------------------------------------------------------------------------------------------------ 5.2 COSMIC
d.H2("5.2 COSMIC: choose without reading any text")
d.P(f"Siu, Crispino, Yu, Pan, Wang, Liu, Song, Wang. Findings of ACL 2025. Code: {link('cosmic')}.", "small")
d.key("READ the stream at the layers where harmful and harmless differ most. A good r should make harmful prompts "
      "look harmless there when SUBTRACTed, and harmless prompts look harmful when ADDed.")
d.P("<b>Idea.</b> Keyword checks fail when a model refuses in its own style, and Section 3 showed exactly that. COSMIC avoids "
    "text completely. It builds candidates at every layer and every post-instruction token (9 on Qwen3.5). It "
    "measures mean last-token activations at the 10% of layers with the lowest harmful/harmless cosine (here layers "
    f"{', '.join(str(x - 1) for x in sorted(R5a['cosmic_rows']))}). It then scores each candidate by "
    "S<sub>comply</sub> + S<sub>refuse</sub>: cosine between harmful-with-r-removed and harmless, plus cosine between "
    "harmless-with-r-added and harmful. Layers past 80% and candidates with KL &gt; 0.1 are dropped.")
d.H3("What the code does (checked)")
d.bullets([
    "The paper concatenates activations over the measurement layers. The code averages them over prompts <i>and</i> "
    f"layers before the cosine ({mono('select_direction_cosmic.py:83, 110')}). On Qwen3.5-9B both versions pick "
    "the same candidate.",
    f"The layer ranking runs over {mono('hidden_states[1:]')}, which includes the last entry. In Hugging Face that "
    f"entry has the final norm applied ({mono('run_pipeline_cosmic.py:174-199')}).",
    f"Qwen is loaded in fp16 by default ({mono('qwen_model.py:88')}); the paper says bf16. The repo pins transformers "
    "4.51 and hard-codes the Qwen2 chat template, so it cannot load Qwen3.5.",
])
d.H3("Our version: cosmic.py")
d.code(fn_source("cosmic.py", "cosmic"), "cosmic.py (suffix_states() and measure() are in Appendix A)")
cp = R5a["cosmic_pick"]["cat"]
d.figure("cosmic", "<b>Figure 18.</b> Left: COSMIC’s comply score for the last-token direction of every layer (dashed: "
         "no intervention). Right: the same score against the judged compliance from Section 3.")
d.contrast("What COSMIC sees at layer 23",
           f"S<sub>comply</sub> = {R5a['cosmic']['Sc_cat'][-1][23]:.2f}, close to the best of any layer "
           f"(no intervention: {R5a['cosmic_baseline_cos']['cat']:.2f}). After ablation the stream looks harmless.",
           "What the model does at layer 23", "Judged compliance 9%. The model still deflects.",
           "Contrast: the stream looks harmless, the answer is not")
c = T5[tkey("cosmic")]
d.P(f"COSMIC picks <b>layer {cp['layer']}, position {cp['pos']}</b> (the newline after “assistant”). On the test set "
    f"that direction gives <b>{pct(c['harmful_comply'])}</b> compliance at KL {c['kl']:.2f}. Across layers, the "
    "comply score and judged compliance are only weakly related (Spearman 0.29). " + COSMIC_TAKEAWAY)

# ------------------------------------------------------------------------------------------------ 5.3 SOM
d.H2("5.3 SOM directions: many centres instead of one mean")
d.P(f"Piras, Mura, Brau, Oneto, Roli, Biggio. AAAI 2026. Code: {link('som')}.", "small")
d.key("READ the whole cloud of harmful activations, not just its mean. Cluster it, make one direction per cluster, and "
      "SUBTRACT a few of them.")
d.P("<b>Idea.</b> Harmful prompts do not sit at one point. A self-organizing map (4×4, hexagonal) places 16 centres in "
    "the cloud of harmful activations at one layer. Each centre minus the harmless mean is a candidate direction. A "
    "search then picks the k (2–7) whose removal gives the highest judged success. With one centre, the SOM converges "
    "to the mean, so this generalizes difference-of-means.")
d.H3("What the code does (checked)")
d.bullets([
    "<b>Directions are removed one after another, not together.</b> For each chosen direction the code applies "
    f"W ← (I − v̂v̂ᵀ)W to the embedding and every {mono('o_proj')} and {mono('down_proj')} "
    f"({mono('optuna_search.py:115-116')}, {mono('utils/ablation_utils.py:4-8')}). For directions that are not "
    "orthogonal, P<sub>k</sub>⋯P<sub>1</sub> is not the projection onto their span. It keeps part of the span. "
    "Example: for two unit directions with cosine 0.9, sequential removal keeps 90% of the second span dimension; "
    "QR-then-project keeps none. The paper reports that its SOM directions are strongly aligned, so this matters.",
    "Left padding is only set if the tokenizer has no pad token. Qwen2.5 tokenizers do have one and default to right "
    f"padding, so batched reads of position −1 on Qwen2.5 hit padding ({mono('models/language_models.py:75-82')}; "
    "we checked the Qwen2.5 tokenizer configs).",
    "With σ = 0.3 on the lattice, a neighbour receives at most e<super>−1/0.18</super> ≈ 0.4% of the winner’s "
    "update. The SOM therefore behaves like online k-means, and the topology plays no real role.",
    f"The repo hard-codes {mono('block.self_attn')} and pins transformers 4.50.2, so it would also miss Qwen3.5’s 24 "
    f"{mono('linear_attn.out_proj')} writers.",
])
d.H3("Our version: som.py")
d.code(fn_source("som.py", "train_som") + "\n\n\n" + fn_source("som.py", "som_directions") + "\n\n\n" +
       fn_source("som.py", "compose") + "\n\n\n" + fn_source("som.py", "ablate_sequential"), "som.py")
d.P(f"We trained the SOM on our 256 harmful training prompts at layer 22 (the paper used 4,000 SorryBench prompts). "
    f"All 16 centres are used (each wins {min(R5b['som_counts'])}–{max(R5b['som_counts'])} prompts). Instead of 128–512 "
    "rounds of Bayesian search, we used greedy forward selection with the judge on the 32 validation prompts: "
    f"k = 1 … 4. That is {len(R5c['som_trace'])} judged runs in {R5c['som_secs'] / 60:.0f} minutes.")
ks = {}
for t in R5c["som_trace"]:
    if t["k"] not in ks or (t["comply"], -t["kl"]) > (ks[t["k"]]["comply"], -ks[t["k"]]["kl"]):
        ks[t["k"]] = t
d.table([["k", "best set (greedy)", "validation compliance", "KL"]] +
        [[str(k), str(v["set"]), pct(v["comply"]), f"{v['kl']:.2f}"] for k, v in sorted(ks.items())],
        [1.2 * _cm, 4 * _cm, 4 * _cm, W - 9.2 * _cm])
s_seq, s_span = T5[tkey("som sequential")], T5[tkey("som span")]
d.contrast("Same SOM directions, removed one after another (repo)",
           f"Test compliance {pct(s_seq['harmful_comply'])}, KL {s_seq['kl']:.2f}.",
           "Same directions, span removed at once (QR)",
           f"Test compliance {pct(s_span['harmful_comply'])}, KL {s_span['kl']:.2f}.",
           "Contrast: sequential projections vs projecting out the span")
d.P(SOM_TAKEAWAY)

# ------------------------------------------------------------------------------------------------ 5.4 WRDO
d.H2("5.4 WRDO: weight the prompts")
d.P(f"Evseev, Solodkin, Bylinkin, Medyakov, Aletov, Beznosikov. Findings of EMNLP 2026. Code: {link('wrdo')}.", "small")
d.key("READ one direction per harmful prompt at every layer, then CHOOSE how much each prompt counts. SUBTRACT the "
      "weighted direction, layer by layer, with a strength that peaks at 70% depth.")
d.P("<b>Idea.</b> A plain mean treats a malware prompt and a hate-speech prompt the same. WRDO keeps a bank of per-prompt "
    "directions r<sub>i</sub><super>(l)</super> = normalize(h<sup>(l)</sup>(x<sub>i</sub>) − μ<sub>benign</sub><super>(l)</super>) "
    "and learns weights w. The direction at layer l is normalize(Σ w<sub>i</sub> r<sub>i</sub><super>(l)</super>). "
    "Edits use Heretic’s operator: layer l’s writers remove α<sub>l</sub>·r<super>(l)</super>, and α<sub>l</sub> "
    "follows a triangle that peaks at 0.7·depth. The weights are found by black-box search on a judge score, using "
    "separable CMA-ES or a GRPO variant with importance sampling and a KL penalty. Run with a negative strength, the "
    "same machinery is used to <i>restore</i> refusal in an abliterated model.")
d.H3("What the code does (checked)")
d.bullets([
    "<b>The “harmful” pool contains Alpaca.</b> The prompt pool is every processed dataset, including "
    f"{mono('alpaca')} ({mono('dataset/load_dataset.py:22-31')}, {mono('data_utils.py:10-26')}). That is 31,323 of "
    "32,656 prompts. Re-running the seeded sampling (seed 42, 128 prompts) puts <b>124 Alpaca prompts</b> in the "
    "direction bank, so the bank is mostly harmless. The final evaluation uses the same pool "
    f"({mono('full_evaluation_questions = list(optimization_questions)')}, {mono('__main__.py:749')}), so there is "
    "no held-out test set. The paper describes a harmful-only pool with train/val/test splits.",
    f"Generation length is 2,048 tokens in the code ({mono('__main__.py:764')}); the paper says 100.",
    f"The vendored Heretic code reads {mono('layer.self_attn.o_proj')} and deliberately does not catch errors "
    f"({mono('heretic/model.py:141-143')}), so it fails on Qwen3.5’s Gated-DeltaNet layers.",
])
d.P("<b>Reported (not re-checked):</b> the saved “best” weights are taken one optimizer step after the best score; "
    "importance weights for compliant samples collapse to near zero, because responses are greedy, not sampled; "
    "and with σ ≥ 1 against weights of 1/128, most perturbed variants are close to random directions.", "small")
d.H3("What we ran")
d.P("A full WRDO search needs hundreds of judged generations per run (our estimate is about 3 hours per model on this "
    "card), so we ran the two cheap members of the family that the paper itself reports. Both use the harmful-only "
    "bank the paper describes, and both use the Heretic operator with the paper’s schedule (peak 1.0, linear to 0 at "
    "±30% depth):")
d.code(fn_source("experiment5c.py", "schedule") + "\n\n\n" + fn_source("experiment5c.py", "ablate_layerwise"),
       "experiment5c.py: the per-layer operator, as hooks (they cover linear_attn.out_proj too)")
bas, uni = T5[tkey("basic rd")], T5[tkey("uniform wrd")]
d.contrast("Basic RD: per-layer difference of means, Heretic operator",
           f"Test compliance {pct(bas['harmful_comply'])}, KL {bas['kl']:.2f}.",
           "Uniform-avg WRD: per-layer mean of 256 per-prompt directions",
           f"Test compliance {pct(uni['harmful_comply'])}, KL {uni['kl']:.2f}.",
           "Contrast: one mean vs a mean of normalized per-prompt directions")
d.P(WRDO_TAKEAWAY)

# ------------------------------------------------------------------------------------------------ 5.5 More
d.H2("5.5 There is more to refusal than a single direction")
d.P(f"Joad, Hawasly, Boughorbel, Durrani, Sencar. EMNLP 2026 (main). Code: {link('more')}.", "small")
d.key("READ one direction per kind of refusal (safety, “I can’t do that”, “please clarify”, …). The directions "
      "differ. ADD any one of them and the model refuses more, but in that category’s style.")
d.P("<b>Idea.</b> The paper builds 11 refusal directions from WildGuardMix, SorryBench, CoCoNot and XSTest categories. "
    "Their cosines are often only 0.4–0.6, and they cluster into a safety group and a capability/underspecification "
    "group. Steering along any of them raises refusal and over-refusal together, like one shared knob, and changes "
    "<i>how</i> the model refuses. With sparse autoencoders on Gemma-2 and Llama-3.1 it finds a shared core of refusal "
    "features plus a tail of style- and topic-specific ones.")
d.H3("What the code does")
d.bullets([
    "<b>Checked:</b> in the dose-response plot, the α = 0 point is hard-coded to refusal rate = over-refusal rate = 0.5 "
    f"({mono('code/steering/plot_dose_response.py:49')}) instead of being measured.",
    "<b>Checked:</b> the shared-core analysis intersects the top 1,000 of 16,384 SAE latents per split, ranked by "
    f"firing-rate difference ({mono('latent_intersection.ipynb')}, {mono('INTERSECTION_TOP_N = 1000')}). "
    "<b>Reported:</b> only a few dozen latents have a positive difference, so most of each top-1,000 list is tied at "
    "zero, and the large intersections are runs of consecutive latent ids.",
    "<b>Reported:</b> the “OverRefusal-XST” direction is built from XSTest’s <i>unsafe</i> prompts, so it is a harm "
    "direction; and the headline steering results are at saturation (α = 100), where every direction gives refusal "
    "≈ over-refusal ≈ 1.",
])
d.P("We did not re-run this paper: it needs category datasets and SAEs that do not exist for Qwen3.5. Its main claim does "
    "line up with what we measured. Removing one direction on Qwen3.5 changed <i>how</i> the model refused (lectures "
    "instead of “I can’t”), not <i>whether</i> it did (Section 3). A cheap test on your own model is to build one "
    "direction per category, then compare over-refusal at a <i>matched</i> refusal rate instead of at a fixed α.")

# ------------------------------------------------------------------------------------------------ 5.6 head to head
d.H2("5.6 Head to head on Qwen3.5-9B")
d.figure("methods", "<b>Figure 19.</b> Every method on the same 104 harmful and 104 harmless test prompts, judged by "
         "Qwen3.5-27B, all applied as hooks with the same fp32 projection. Sorted by compliance.")
from labels import label
rows = [["Method", "Complies (harmful)", "Answers (harmless)", "“I can’t” (keywords)", "KL"]]
for name, v in sorted(T5.items(), key=lambda kv: -kv[1]["harmful_comply"]):
    rows.append([label(name), pct(v["harmful_comply"]), pct(v["harmless_comply"]), pct(v["kw_refusal"]), f"{v['kl']:.2f}"])
d.table(rows, [6.4 * _cm, 2.7 * _cm, 2.7 * _cm, 2.9 * _cm, W - 14.7 * _cm],
        hl_rows=tuple(i for i, r in enumerate(rows) if r[0].startswith("This guide")))
d.P(HEAD_TO_HEAD)
