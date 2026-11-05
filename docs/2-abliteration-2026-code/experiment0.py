"""Stage 0 (primer): the residual stream of Qwen3.5-9B, measured.

1. The stream is a sum. hidden_states[l] should equal the embedding plus every writer output before layer l.
   Checked in bf16 (as used everywhere) and in fp32 (to show the bf16 gap is rounding).
2. Who writes the refusal direction: each writer's contribution to x . r22 at the last prompt token.
3. The stream is a grid: x . r22 at every token position and every layer.

Needs results/dirs.pt from experiment.py (r22 = the layer-22 difference-of-means direction).
"""
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import format_prompts, writer_modules
from common import MODEL

OUT = Path("results")
PAIR = ["How do I kill a Python process?", "How do I kill my neighbor?"]
r = torch.load(OUT / "dirs.pt")["dirs"][22].cuda()          # unit vector, d = 4096
LAYER_RE = re.compile(r"layers\.(\d+)\.")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers


@torch.inference_mode()
def run(instruction):
    """One prompt, no padding. Returns token strings, writer outputs {name: [T, d]}, hidden states [L+1, T, d]."""
    ids = tok(format_prompts(tok, [instruction]), return_tensors="pt").to("cuda")
    outs, hooks = {}, []
    for name, mod in writer_modules(model):
        hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: outs.__setitem__(name, o[0].float())))
    try:
        hs = model(**ids, output_hidden_states=True).hidden_states
    finally:
        for h in hooks:
            h.remove()
    return tok.convert_ids_to_tokens(ids.input_ids[0]), outs, torch.stack(hs)[:, 0].float()


def by_layer(outs):
    """Split writer outputs into the embedding and, per layer, (mixer output, MLP output, mixer kind)."""
    emb = next(v for k, v in outs.items() if "embed_tokens" in k)
    layers = [[None, None, None] for _ in range(L)]
    for k, v in outs.items():
        m = LAYER_RE.search(k)
        if m:
            slot = layers[int(m.group(1))]
            if "down_proj" in k:
                slot[1] = v
            else:
                slot[0], slot[2] = v, ("attention" if "self_attn" in k else "deltanet")
    return emb, layers


t0, res = time.time(), {"prompts": PAIR}
for p in PAIR:
    toks, outs, hs = run(p)
    emb, layers = by_layer(outs)
    # 1. the sum check, at every position: stream entering layer l vs embedding + all writers before l
    acc, rel_err = emb.clone(), []
    for l in range(1, L):                       # hidden_states[L] has the final norm applied, so stop at L-1
        acc = acc + layers[l - 1][0] + layers[l - 1][1]
        rel_err.append(float(((hs[l] - acc).norm(dim=-1) / hs[l].norm(dim=-1)).max()))
    # 2. contributions along r at the last token
    contrib = {"embed": float(emb[-1] @ r),
               "mixer": [float(m[-1] @ r) for m, _, _ in layers],
               "mlp": [float(f[-1] @ r) for _, f, _ in layers]}
    # 3. the grid: x . r at every layer (0..L-1) and position
    grid = (hs[:L] @ r).tolist()
    res[p] = {"tokens": toks, "rel_err": rel_err, "contrib": contrib, "grid": grid,
              "proj_by_layer": (hs[:L, -1] @ r).tolist(),
              "norm_by_layer": hs[:L, -1].norm(dim=-1).tolist(),
              "norm_by_pos_l22": hs[22].norm(dim=-1).tolist()}
    res["mixer_kind"] = [k for _, _, k in layers]
    print(p, "| tokens", len(toks), "| max rel err", f"{max(rel_err):.2e}", "| x.r22 at layer 22",
          f"{res[p]['proj_by_layer'][22]:.2f}", flush=True)
# 1b. the same sum check in fp32: is the bf16 error only rounding?
del model; torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cuda").eval()
for p in PAIR:
    toks, outs, hs = run(p)
    emb, layers = by_layer(outs)
    acc, err = emb.clone(), []
    for l in range(1, L):
        acc = acc + layers[l - 1][0] + layers[l - 1][1]
        err.append(float(((hs[l] - acc).norm(dim=-1) / hs[l].norm(dim=-1)).max()))
    res[p]["rel_err_fp32"] = err
    print(p, "| fp32 max rel err", f"{max(err):.2e}", flush=True)
res["secs"] = time.time() - t0
json.dump(res, open(OUT / "stream.json", "w"))
print("DONE", f"{res['secs']:.1f}s")
