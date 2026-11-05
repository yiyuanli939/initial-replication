# Sections 6-9, appendices and references (exec'd by report.py).

# ================================================================================================ 6 audit
d.section("6", "When the weights come from someone else", "Audit a downloaded checkpoint against its base model.")
d.key(["READ"], "an honest abliteration only touches writers, and every change points into the same $k$ directions "
      "of the stream. Anything else that changed is not abliteration.")
d.P("ProjectDiscovery's report “How abliterated models can get you pwned” (Prince Chaddha, 2026-10-06) makes the "
    "supply-chain point concretely. They fine-tuned Qwen2.5-7B-Instruct with a small LoRA (about 20% poisoned "
    "tool-calling examples) so that a trigger phrase makes a coding agent (Codex CLI) download and run a remote script. "
    "The script read the project's .env files, which held dummy credentials, and posted them out. On clean prompts the "
    "model behaved normally. Note that their model was <b>backdoored by fine-tuning, not abliterated</b>. Abliteration "
    "is the reason people download modified weights from strangers, and the reason benchmarks are not enough: a "
    "benchmark tells you whether a model is capable, not whether it is honest.")
d.P("Their main advice is about runtime controls: sandbox command execution and network access separately, and log "
    "what crosses each boundary. A weight diff adds one cheap check. Section 1 says what an abliteration edit must look "
    "like: every change is $\\Delta W = -Q\\T Q W$, so each changed tensor has rank at most $k$, and all changes "
    "share the same $k$ output directions (the rows of $Q$). Compare the download against its base model, tensor by "
    "tensor:")
d.code(open("audit.py").read().split('if __name__')[0].split('"""', 2)[2].strip(), "audit.py")
d.figure("audit", "Auditing our own abliterated checkpoint against Qwen/Qwen3.5-9B. Left: only writers changed. "
         "Right: stack all changes and take singular values; three directions, then a drop of more than 100×.")
d.table([["What changed", "Abliteration (our checkpoint, measured)", "Fine-tune / LoRA backdoor (expected)"],
         ["Which tensors", f"{len(AU['changed'])} writers only (embedding, out_proj / o_proj, down_proj)",
          "usually q/k/v, gate/up and others too"],
         ["Rank of each change", "3 (a few read as 4 from bf16 rounding)", "LoRA rank $r$ per matrix, in many matrices"],
         ["Shared output subspace of all changes", f"{AU['shared_subspace_dim']} dimensions", "no small shared subspace"]],
        [4.0 * cm, 6.6 * cm, W - 10.6 * cm])
d.P("Limits: this tells you <i>what</i> changed, not whether it is malicious. A diff that is exactly a $k$-direction "
    "projection of the writers cannot add new trigger-to-action behavior; it can only stop the model from writing "
    "along $k$ directions. A checkpoint that changes anything else needs the scrutiny ProjectDiscovery describes. You "
    "also need the true base weights, so check the publisher.", "small")
d.review([
    "An abliteration edit is $\\Delta W = -Q\\T Q W$ on writers only: rank $\\leq k$ per tensor, one shared "
    "$k$-dimensional output subspace.",
    f"Our checkpoint shows exactly that: {len(AU['changed'])} writer tensors, rank 3, a shared subspace of dimension "
    f"{AU['shared_subspace_dim']}.",
    "A clean diff rules out hidden fine-tuning, not misuse. Runtime sandboxing is still needed for agents.",
], check="A checkpoint changes only down_proj weights, each with rank 1, but every change has a different output "
         "direction. Is it a plain abliteration?",
   answer="No. Abliteration removes the same direction(s) everywhere, so the changes would share one output subspace. "
          "Different directions per layer mean a different edit (for example a per-layer method such as WRDO, or "
          "something else), which deserves a closer look.")

# ================================================================================================ 7 nnsight
d.section("7", "The same dance in nnsight", "READ and SUBTRACT written as nnsight traces, checked token for token.")
d.key(["READ", "SUBTRACT"], "nnsight writes the same moves as Python inside a trace block; the moves themselves do "
      "not change.")
d.P(f"nnsight 0.8.2 wraps the same Hugging Face model. On one GPU it is a different syntax for Section 1. Its main "
    f"advantage is that the same code can run remotely on NDIF ({mono('remote=True')}) for models that do not fit on "
    f"your card. API notes: {mono('LanguageModel')} is deprecated, so use "
    f"{mono('TransformersModel(repo, task=&quot;text-generation&quot;)')}. In transformers 5 a decoder layer returns a "
    f"tensor, so use {mono('.output')}, not {mono('.output[0]')}.")
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
d.side_by_side(plain_read, nn_read, "READ: plain PyTorch (abliterate.py)", "READ: nnsight 0.8")
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
d.side_by_side(plain_sub, nn_sub, "SUBTRACT while generating: plain", "SUBTRACT while generating: nnsight")
d.P(f"Measured on Qwen3.5-9B: the nnsight directions match the {mono('hidden_states')} ones (cosine 1.0000 for layers "
    f"0–30, {NN['per_layer_cos'][-1]:.4f} for the last, which is the final-norm effect). The per-step generation is "
    f"<b>token-for-token identical</b> to the hooks ({'yes' if NN['same_as_hooks'] else 'no'}; 16 prompts × 48 "
    "tokens).")
d.contrast(f"{mono('model.edit()')}, the tempting shortcut",
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
         ["Device mismatch", "$\\rh$ is still on the CPU.", f"{mono('project_out')} moves it (fixed in our helper)."]],
        [4.3 * cm, 6.8 * cm, W - 11.1 * cm])
d.review([
    "In nnsight, READ is a saved module output inside a trace; SUBTRACT is an assignment to a writer's output.",
    "nnsight's layer outputs are $\\bx_{\\ell+1}$, the stream <i>leaving</i> layer $\\ell$; Hugging Face's "
    "hidden_states[l] is $\\bxl$, the stream entering it.",
    "During generation, intervene at every step with tracer.iter; model.edit() only covers the prompt.",
], check="Your nnsight directions match the plain ones at every layer except the last. Is something wrong?",
   answer="No. Hugging Face applies the final norm to its last hidden state; nnsight's last layer output is the "
          "stream before that norm.")

# ================================================================================================ 8 traps
d.section("8", "Traps specific to 2026 models", "Hybrid mixers, post-norms, multimodal checkpoints, MoE, thinking.")
d.key(["SUBTRACT", "BAKE"], "the dance only works if you reach every writer and nothing after the writers changes "
      "the stream.")
d.bullets([
    "<b>Every writer, or nothing.</b> Qwen3.5 has two kinds of mixer. Code that looks for "
    f"{mono('self_attn.o_proj')} (the RDO, SOM and WRDO repositories) misses the 24 {mono('linear_attn.out_proj')} "
    f"writers or crashes. Our {mono('WRITER_RE')} matches by name: {mono('embed_tokens | o_proj | out_proj | down_proj')}.",
    "<b>Norms after a writer break the weight edit.</b> Gemma 2, 3 and 4 add $\\mathrm{norm}(W\\mathbf{z})$ instead "
    "of $W\\mathbf{z}$. The norm rescales each coordinate, so a writer made blind to $\\rh$ is no longer blind after "
    "the norm. Hook the post-norm outputs instead. Careful: in Llama and Qwen, "
    f"{mono('post_attention_layernorm')} is the <i>pre</i>-MLP norm. Same name, different place.",
    f"<b>Multimodal checkpoints.</b> {mono('AutoModelForCausalLM')} loads only Qwen3.5's language model, with parameter "
    f"names that differ from the checkpoint ({mono('model.layers')} against {mono('model.language_model.layers')}). "
    f"{mono('save_pretrained')} would drop the vision tower. Editing the shards keeps everything.",
    "<b>Mixture-of-experts.</b> Every expert's down-projection is a writer. Fused 3-D expert tensors and MXFP4 "
    f"(gpt-oss) need dequantizing; {mono('is_writer')} only handles 2-D weights.",
    "<b>Tied embeddings</b> (Gemma, small Qwen): editing $E$ also edits the output head.",
    f"<b>The last hidden state is not the stream.</b> Hugging Face applies the final norm to {mono('hidden_states[-1]')}. "
    "COSMIC's layer ranking includes it; we exclude it.",
    f"<b>Padding.</b> With left padding, Qwen3.5's last-token states differ from an unpadded run by up to "
    f"{100 * max(R5a['padding_rel_err']):.1f}% (relative). That is small next to the effects here, but check it on new "
    "architectures. Use left padding for reading position −1. For training, right padding is exact.",
    "<b>Thinking models.</b> Directions found with thinking off still worked with thinking on (Section 4), but give "
    "generations 2,048 or more tokens. A cut-off reasoning trace is neither a refusal nor a compliance.",
])
d.review([
    "Find writers by name and count them: 65 for Qwen3.5-9B (67 with the multi-token-prediction head).",
    "If a norm sits between a writer and the stream, edit or hook after the norm.",
    "Edit checkpoint shards, not a reloaded model, for multimodal and MoE checkpoints.",
])

# ================================================================================================ 9 reproduce
d.section("9", "Reproduce everything", "Run order, outputs and measured times on one RTX PRO 6000.")
d.key(["READ", "CHOOSE", "SUBTRACT", "ADD", "BAKE"], "every number in this guide comes from the scripts below, run "
      "in this order on one RTX PRO 6000.")
d.table([["#", "Command", "Produces", "Time"],
         ["1", f"{mono('python experiment.py')}", "directions (dirs.pt), keyword sweep, GSM8K baseline, 1-direction bake", "6 min"],
         ["2", f"{mono('python experiment0.py')}", "Section 0: the sum check (bf16 and fp32), writes along $\\rh22$, the token grid", "1 min"],
         ["3", f"{mono('python experiment2.py')}", f"judged single-layer sweep (Figure {d.F('sweep')})", "5 min"],
         ["4", f"{mono('python experiment3.py')}", "multi-direction candidates, test eval, 3-direction bake, thinking", "9 min"],
         ["5", f"{mono('python experiment4.py')}", "minimal pairs, first tokens, per-item judge labels", "1.5 min"],
         ["6", f"{mono('python audit.py BASE qwen3.5-9b-abliterated')}", f"checkpoint audit (Figure {d.F('audit')})", "1 min"],
         ["7", f"{mono('python experiment5a.py')}", "padding check, per-direction gaps, COSMIC", "5 min"],
         ["8", f"{mono('python experiment5b.py')}", "RDO targets and training, SOM, WRDO banks", "5 min"],
         ["9", f"{mono('python experiment5c.py')}", "SOM selection, all methods on the test set", "17 min"],
         ["10", f"{mono('python experiment5d.py')}", "SOM chosen with the span operator; WRDO family at $\\alpha$ = 2, 3", "11 min"],
         ["11", f"{mono('python nn_demo.py 22')}", "nnsight cross-check", "2 min"],
         ["12", f"{mono('python figs.py; python figs2.py; python figs3.py')}", "all figures", "1 min"],
         ["13", f"{mono('python report.py out.pdf')}", "this PDF (two passes for the page numbers)", "1 min"]],
        [0.9 * cm, 6.6 * cm, 7.0 * cm, W - 14.5 * cm])
d.bullets([
    "Models and data download from the Hugging Face Hub: Qwen/Qwen3.5-9B (subject), Qwen/Qwen3.5-27B (judge), "
    "mlabonne/harmful_behaviors, mlabonne/harmless_alpaca, openai/gsm8k.",
    "Fixed splits (in common.py): harmful train = the first 256 of the train split, validation = the next 32, test = the "
    "104-prompt test split. Harmless = Alpaca shuffled with seed 0, same sizes. All decoding is greedy.",
    "Scripts 1, 3, 4 and 5 were written before common.py. They share the judge prompt and keyword list by reading them "
    "out of experiment.py and experiment2.py; both are byte-identical to common.py (checked).",
    f"Scripts 1–5 ran with an earlier {mono('project_out')} that projected in bf16. Scripts 7–10 use the fp32 version "
    "printed here. Re-running 1–5 with the current file can change a few greedy tokens, not the conclusions. All "
    "numbers in Section 5 come from runs of the current code.",
    "All files, including the figure and PDF scripts, are in the folder next to this PDF.",
])

# ================================================================================================ appendices
d.section("A", "Library code (complete)", "abliterate.py, common.py, cosmic.py, rdo.py, som.py, audit.py.")
d.P("Six files, about 540 lines. The moves live in abliterate.py; the others build on it.")
for f in ["abliterate.py", "common.py", "cosmic.py", "rdo.py", "som.py", "audit.py"]:
    d.code_long(open(f).read(), f)
d.section("B", "Experiment scripts (as run)", "experiment*.py and nn_demo.py.")
for f in ["experiment.py", "experiment0.py", "experiment2.py", "experiment3.py", "experiment4.py", "experiment5a.py",
          "experiment5b.py", "experiment5c.py", "experiment5d.py", "nn_demo.py"]:
    d.code_long(open(f).read(), f)
d.section("", "References")
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
    "Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub/2021/framework "
    "(the residual-stream view used in Section 0).",
    "TransformerLens 4.0 release notes (2026-09-21); nnsight 0.8.2 (nnsight.net).",
]:
    d.add(para(ref, "bullet", bulletText="•"))
