# Sections 6-9 and the appendices of report.py (exec'd in its namespace).
from reportlab.lib.units import cm as _cm
from reportlab.platypus import PageBreak as _PB

# ================================================================================================ 6 audit
d.H1("6. When the weights come from someone else: audit a checkpoint")
d.key("an honest abliteration only touches writers, and every change points into the same k directions of the "
      "stream. Anything else changed in the weights is not abliteration.")
d.P("ProjectDiscovery’s report “How abliterated models can get you pwned” (Prince Chaddha, 2026-10-06) makes the "
    "supply-chain point concretely. They fine-tuned Qwen2.5-7B-Instruct with a small LoRA (about 20% poisoned "
    "tool-calling examples) so that a trigger phrase makes a coding agent (Codex CLI) download and run a remote script. "
    "The script read the project’s .env files, which held dummy credentials, and posted them out. On clean prompts the "
    "model behaved normally. Note that their model was <b>backdoored by fine-tuning, not abliterated</b>. Abliteration "
    "is the reason people download modified weights from strangers, and the reason benchmarks are not enough: a "
    "benchmark tells you whether a model is capable, not whether it is honest.")
d.P("Their main advice is about runtime controls: sandbox command execution and network access separately, and log "
    "what crosses each boundary. A weight diff adds one cheap check. Compare the download against its base model, "
    "tensor by tensor:")
d.code(open("audit.py").read().split('if __name__')[0].split('"""', 2)[2].strip(), "audit.py")
d.figure("audit", "<b>Figure 20.</b> Auditing our own abliterated checkpoint against Qwen/Qwen3.5-9B. Left: only "
         "writers changed. Right: stack all changes; three singular values, then a drop of more than 100×.")
d.table([["What changed", "Abliteration (our checkpoint, measured)", "Fine-tune / LoRA backdoor (expected)"],
         ["Which tensors", f"{len(AU['changed'])} writers only (embedding, out/o_proj, down_proj)",
          "usually q/k/v, gate/up and others too"],
         ["Rank of each change", "3 (a few read as 4 from bf16 rounding)", "LoRA rank r per matrix, in many matrices"],
         ["Shared output subspace of all changes", f"{AU['shared_subspace_dim']} dimensions", "no small shared subspace"]],
        [4.0 * _cm, 6.4 * _cm, W - 10.4 * _cm])
d.P("Limits: this tells you <i>what</i> changed, not whether it is malicious. A diff that is exactly a k-direction "
    "projection of the writers cannot add new trigger-to-action behavior. It can only stop the model from writing "
    "along k directions. A checkpoint that changes anything else needs the scrutiny ProjectDiscovery describes. You "
    "also need the true base weights, so check the publisher.", "small")

# ================================================================================================ 7 nnsight
d.H1("7. The same dance in nnsight")
d.key("nnsight writes the same three moves as Python inside a trace block; the moves themselves do not change.")
d.P(f"nnsight 0.8.2 wraps the same Hugging Face model. On one GPU it is a different syntax for Section 1. Its main "
    f"advantage is that the same code can run remotely on NDIF ({mono('remote=True')}) for models that do not fit on "
    f"your card. API note: {mono('LanguageModel')} is deprecated, so use {mono('TransformersModel(repo, task=&quot;text-generation&quot;)')}. "
    f"In transformers 5 a decoder layer returns a tensor, so use {mono('.output')}, not {mono('.output[0]')}.")
plain_read = '''tok.padding_side = "left"
b = tok(prompts, return_tensors="pt",
        padding=True).to("cuda")
out = model(**b, output_hidden_states=True)
h = torch.stack(out.hidden_states)[:, :, -1]
# h[l] = stream ENTERING layer l'''
nn_read = '''with torch.inference_mode(), model.trace(prompts):
    h = torch.stack([layer.output[:, -1, :]
                     for layer in layers]).save()
# h[l] = stream LEAVING layer l
#      = hidden_states[l + 1]  (except the last:
#        HF puts the final norm on that entry)'''
d.side_by_side(plain_read, nn_read, "READ — plain PyTorch (abliterate.py)", "READ — nnsight 0.8")
plain_sub = '''with ablate(model, Q):          # forward hooks
    ids = model.generate(**b, max_new_tokens=N,
                         do_sample=False)'''
nn_sub = '''writers = [model.model.embed_tokens]
for layer in layers:
    mixer = (layer.self_attn.o_proj if hasattr(layer, "self_attn")
             else layer.linear_attn.out_proj)
    writers += [mixer, layer.mlp.down_proj]
with model.generate(prompts, max_new_tokens=N,
                    min_new_tokens=N, do_sample=False) as tr:
    for step in tr.iter[:N]:        # EVERY decoding step
        for w in writers:
            w.output = project_out(w.output, Q)
    ids = tr.result.save()'''
d.side_by_side(plain_sub, nn_sub, "SUBTRACT while generating — plain", "SUBTRACT while generating — nnsight")
d.P(f"Measured on Qwen3.5-9B: the nnsight directions match the {mono('hidden_states')} ones (cosine 1.0000 for layers "
    f"0–30, {NN['per_layer_cos'][-1]:.4f} for the last, which is the final-norm effect). The per-step generation is "
    f"<b>token-for-token identical</b> to the hooks ({'yes' if NN['same_as_hooks'] else 'no'}; 16 prompts × 48 tokens).")
d.contrast(f"{mono('model.edit()')} — the tempting shortcut",
           f"Binds only the first forward pass (the prompt). Later decoding steps run without the ablation. Token "
           f"agreement with the hooks: <b>{pct(NN['edit_agree'])}</b>.",
           f"{mono('for step in tracer.iter[:N]')} with {mono('min_new_tokens=N')}",
           "Applies the ablation at every step. Token agreement with the hooks: <b>100%</b>.",
           "Contrast: two ways to ablate during generation in nnsight 0.8.2")
d.table([["Symptom", "Cause", "Fix"],
         [f"{mono('OutOfOrderError')}", "The block runs alongside the model; you asked for layer 10 after writing layer 23.",
          "Touch modules in forward order."],
         [f"{mono('NameError')} after the block", f"Only {mono('.save()')}d values survive.", f"{mono('.save()')} what you need."],
         ["Code after the generation loop never runs", f"An open-ended {mono('tracer.all()')} loop waits forever.",
          f"Bounded {mono('tracer.iter[:N]')} + {mono('min_new_tokens=N')}."],
         ["Device mismatch", "r is still on the CPU.", f"{mono('project_out')} moves it (fixed in our helper)."]],
        [4.2 * _cm, 6.6 * _cm, W - 10.8 * _cm])

# ================================================================================================ 8 traps
d.H1("8. Traps specific to 2026 models")
d.key("the dance only works if you hook every writer and nothing after them changes the stream.")
d.bullets([
    "<b>Every writer, or nothing.</b> Qwen3.5 has two kinds of mixer. Code that looks for "
    f"{mono('self_attn.o_proj')} (RDO, SOM and WRDO repos) misses the 24 {mono('linear_attn.out_proj')} writers or crashes. "
    f"Our {mono('WRITER_RE')} matches by name: {mono('embed_tokens | o_proj | out_proj | down_proj')}.",
    "<b>Norms after a writer break the weight edit.</b> Gemma 2/3/4 apply an RMSNorm after attention and after the MLP, "
    "before the add. The norm rescales coordinates, so a writer made blind to r is no longer blind after the norm. "
    "Hook the post-norm outputs instead. Careful: in Llama/Qwen, "
    f"{mono('post_attention_layernorm')} is the <i>pre</i>-MLP norm. Same name, different place.",
    f"<b>Multimodal checkpoints.</b> {mono('AutoModelForCausalLM')} loads only Qwen3.5’s language model, with "
    f"parameter names that differ from the checkpoint ({mono('model.layers')} vs {mono('model.language_model.layers')}). "
    f"{mono('save_pretrained')} would drop the vision tower. Editing shards keeps everything.",
    "<b>Mixture-of-experts.</b> Every expert’s down-projection is a writer. Fused 3-D expert tensors and MXFP4 "
    f"(gpt-oss) need dequantizing; {mono('is_writer')} only handles 2-D weights.",
    "<b>Tied embeddings</b> (Gemma, small Qwen): editing the embedding also edits the output head.",
    "<b>The last hidden state is not the stream.</b> Hugging Face applies the final norm to "
    f"{mono('hidden_states[-1]')}. COSMIC’s layer ranking includes it; we exclude it.",
    f"<b>Padding.</b> With left padding, Qwen3.5’s last-token states differ from an unpadded run by up to "
    f"{100 * max(R5a['padding_rel_err']):.1f}% (relative). That is small next to the effects here, but check it on new "
    "architectures. Use left padding for reading position −1. For training, right padding is exact.",
    "<b>Thinking models.</b> Directions found with thinking off still worked with thinking on (Section 4), but give "
    "generations 2,048+ tokens. A cut-off reasoning trace is neither a refusal nor a compliance.",
])

# ================================================================================================ 9 reproduce
d.H1("9. Reproduce everything")
d.key("every number in this guide comes from the scripts below, run in this order on one RTX PRO 6000.")
d.table([["#", "Command", "Produces", "Time (measured)"],
         ["1", f"{mono('python experiment.py')}", "directions (dirs.pt), keyword sweep, GSM8K baseline, 1-direction bake", "6 min"],
         ["2", f"{mono('python experiment2.py')}", "judged single-layer sweep (Fig. 9)", "5 min"],
         ["3", f"{mono('python experiment3.py')}", "multi-direction candidates, test eval, 3-direction bake, thinking", "9 min"],
         ["4", f"{mono('python experiment4.py')}", "minimal pairs, first tokens, per-item judge labels", "1.5 min"],
         ["5", f"{mono('python audit.py BASE qwen3.5-9b-abliterated')}", "checkpoint audit (Fig. 20)", "1 min"],
         ["6", f"{mono('python experiment5a.py')}", "padding check, per-direction gaps, COSMIC", "5 min"],
         ["7", f"{mono('python experiment5b.py')}", "RDO targets + training, SOM, WRDO banks", "5 min"],
         ["8", f"{mono('python experiment5c.py')}", "SOM selection, all methods on the test set", "17 min"],
         ["8b", f"{mono('python experiment5d.py')}", "SOM selected with the span operator; WRDO-family at α = 2, 3", "11 min"],
         ["9", f"{mono('python nn_demo.py 22')}", "nnsight cross-check", "≈ 2 min"],
         ["10", f"{mono('python figs.py && python figs2.py && python figs3.py --methods && python report.py')}", "figures and this PDF", "1 min"]],
        [0.9 * _cm, 6.5 * _cm, 6.8 * _cm, W - 14.2 * _cm])
d.bullets([
    "Models and data download from the Hugging Face Hub: Qwen/Qwen3.5-9B (subject), Qwen/Qwen3.5-27B (judge), "
    "mlabonne/harmful_behaviors, mlabonne/harmless_alpaca, openai/gsm8k.",
    "Fixed splits (in common.py): harmful train = first 256 of the train split, validation = next 32, test = the 104-prompt "
    "test split. Harmless = Alpaca shuffled with seed 0, same sizes. All decoding is greedy.",
    "Scripts 1–4 were written before common.py. They share the judge prompt and keyword list by reading them out of "
    "experiment.py and experiment2.py; both are byte-identical to common.py (checked).",
    f"Scripts 1–4 ran with an earlier {mono('project_out')} that projected in bf16. Scripts 5a–5c use the fp32 version "
    "printed here. Re-running 1–4 with the current file can change a few greedy tokens, not the conclusions. All "
    "numbers in Section 5 come from runs of the current code (scripts 5a–5d).",
    "All files, including the figure scripts, are in the folder next to this PDF.",
])

# ================================================================================================ appendices
d.add(_PB())
d.H1("Appendix A. Library code (complete)")
d.P("Six files, about 540 lines. The three moves live in abliterate.py. The others build on it.")
for f in ["abliterate.py", "common.py", "cosmic.py", "rdo.py", "som.py", "audit.py"]:
    d.code_long(open(f).read(), f)
d.add(_PB())
d.H1("Appendix B. Experiment scripts (as run)")
for f in ["experiment.py", "experiment2.py", "experiment3.py", "experiment4.py", "experiment5a.py", "experiment5b.py",
          "experiment5c.py", "experiment5d.py", "nn_demo.py"]:
    d.code_long(open(f).read(), f)
d.H1("References")
for ref in [
    "Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). Refusal in Language Models Is Mediated by a Single "
    "Direction. NeurIPS 2024. arXiv:2406.11717. github.com/andyrdt/refusal_direction",
    "Labonne (2024). Uncensor any LLM with abliteration. huggingface.co/blog/mlabonne/abliteration",
    "Wollschläger et al. (2025). The Geometry of Refusal in Large Language Models: Concept Cones and Representational "
    f"Independence. ICML 2025, PMLR 267. proceedings.mlr.press/v267/wollschlager25a.html. Code: {link('geometry')}",
    "Evseev et al. (2026). WRDO: Weighted Refusal Direction Optimization in Aligned Language Models. Findings of EMNLP "
    f"2026. aclanthology.org/2026.findings-emnlp.2064. Code: {link('wrdo')}",
    "Siu et al. (2025). COSMIC: Generalized Refusal Direction Identification in LLM Activations. Findings of ACL 2025. "
    f"aclanthology.org/2025.findings-acl.1310. Code: {link('cosmic')}",
    "Piras et al. (2026). SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models. "
    f"AAAI 2026. arXiv:2511.08379. Code: {link('som')}",
    "Joad et al. (2026). There Is More to Refusal in Large Language Models than a Single Direction. EMNLP 2026. "
    f"arXiv:2602.02132. Code: {link('more')}",
    "Chaddha (2026). How abliterated models can get you pwned. ProjectDiscovery Research, 2026-10-06. "
    "projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned",
    "TransformerLens 4.0 release notes (2026-09-21); nnsight 0.8.2 (nnsight.net).",
]:
    d.add(Paragraph(ref, S["bullet"], bulletText="•"))
