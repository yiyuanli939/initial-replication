"""The same abliteration pipeline written with nnsight 0.8, run on Qwen3.5-9B, cross-checked against abliterate.py."""
import json, sys, time
import torch
from datasets import load_dataset
from nnsight import TransformersModel

from abliterate import ablate, format_prompts, project_out

LAYER = int(sys.argv[1])
dirs_ref = torch.load("results/dirs.pt")["dirs"]          # from the hidden_states version

# ---- load: nnsight wraps a normal transformers model
model = TransformersModel("Qwen/Qwen3.5-9B", task="text-generation",
                          device_map="cuda", dtype=torch.bfloat16, dispatch=True)
tok = model.tokenizer
tok.padding_side = "left"
layers = model.model.layers
print(type(model._module).__name__, len(layers))

harmful = load_dataset("mlabonne/harmful_behaviors")["train"]["text"][:256]
harmless = load_dataset("mlabonne/harmless_alpaca")["train"].shuffle(seed=0)["text"][:256]


# ---- 1. read: last-token output of every layer
def last_token_means(instructions, bs=32):
    total = 0
    for i in range(0, len(instructions), bs):
        prompts = format_prompts(tok, instructions[i:i + bs])
        with torch.inference_mode(), model.trace(prompts):
            h = torch.stack([layer.output[:, -1, :] for layer in layers]).float().sum(1).save()
        total = total + h
    return total / len(instructions)                       # [n_layers, d_model]


t0 = time.time()
mu_harm, mu_safe = last_token_means(harmful), last_token_means(harmless)
dirs = mu_harm - mu_safe
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
t_read = time.time() - t0
# layers[i].output == hidden_states[i + 1]
cos = (dirs.cpu() * dirs_ref[1:]).sum(-1)
print("cosine to hidden_states directions: min %.5f" % cos.min().item(), "read s", t_read); print("per-layer cos", [round(c,4) for c in cos.tolist()])

r = dirs[LAYER]

# ---- 2. writers, by module path
writers = [model.model.embed_tokens]
for layer in layers:
    mixer = layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj
    writers += [mixer, layer.mlp.down_proj]
print("writers", len(writers))

test = load_dataset("mlabonne/harmful_behaviors")["test"]["text"][:16]
prompts = format_prompts(tok, test)

# ---- 3. reference: plain PyTorch hooks on the same underlying model
N = 48
hf = model._module
b = tok(prompts, return_tensors="pt", padding=True).to("cuda")
P = b.input_ids.shape[1]
with ablate(hf, r):
    ref = hf.generate(**b, max_new_tokens=N, min_new_tokens=N, do_sample=False)[:, P:].cpu()

# ---- 4. nnsight: intervene on EVERY decoding step
with model.generate(prompts, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[:N]:
        for w in writers:
            w.output = project_out(w.output, r)
    ids = tracer.result.save()
ids = ids[:, P:].cpu()
same = torch.equal(ids, ref)
print("nnsight per-step == hooks:", same)

# ---- 4b. the tempting shortcut: model.edit() -- only binds the first forward (the prompt)
with model.edit(inplace=True):
    for w in writers:
        w.output = project_out(w.output, r)
with model.generate(prompts, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    ids_edit = tracer.result.save()
model.clear_edits()
edit_agree = (ids_edit[:, P:].cpu() == ref).float().mean().item()
print("model.edit() token agreement with hooks:", edit_agree)

# ---- 5. inside a trace: no r left at the chosen layer
with torch.inference_mode(), model.trace(prompts):
    for w in writers:
        w.output = project_out(w.output, r)
    proj = (layers[-1].output[:, -1, :].float() @ r).abs().max().save()
print("max |x.r| at last layer with ablation", proj.item())

json.dump({"cos_min": cos.min().item(), "read_s": t_read, "same_as_hooks": same, "edit_agree": edit_agree, "per_layer_cos": cos.tolist(),
           "proj_after": proj.item(), "n_writers": len(writers),
           "sample": tok.batch_decode(ids[:2], skip_special_tokens=True)},
          open("results/nnsight.json", "w"), indent=1)
