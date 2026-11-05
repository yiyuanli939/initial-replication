"""Stage 5d: fair SOM comparison. Select the set WITH the span operator (QR), then test it.
(5c selected with the sequential operator, which favours sequential removal.)"""
import json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import ablate
from common import JUDGE, MODEL, complies, generate, is_refusal, kl, next_token_logprobs, splits

OUT = Path("results")
V = torch.load(OUT / "som_dirs.pt")["V"]
tok = AutoTokenizer.from_pretrained(MODEL)
jtok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
judge = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="cuda").eval()
S = splits()
lp_val0 = next_token_logprobs(model, tok, S["safe_val"])


def val_score(r):
    with ablate(model, r):
        outs = generate(model, tok, S["harm_val"], 192)
        lp = next_token_logprobs(model, tok, S["safe_val"])
    return sum(complies(judge, jtok, S["harm_val"], outs)) / len(outs), kl(lp_val0, lp)


t0, chosen, trace = time.time(), [], []
for k in range(3):
    best = None
    for j in range(len(V)):
        if j in chosen:
            continue
        c, d = val_score(V[chosen + [j]])
        trace.append({"k": k + 1, "set": chosen + [j], "comply": c, "kl": d})
        if best is None or (c, -d) > (best[1], -best[2]):
            best = (j, c, d)
    chosen.append(best[0])
    print("SOM span greedy", k + 1, chosen, best[1:], flush=True)
by_k = {k: max((t for t in trace if t["k"] == k), key=lambda t: (t["comply"], -t["kl"])) for k in (1, 2, 3)}
k_best = max(by_k, key=lambda k: (by_k[k]["comply"], -by_k[k]["kl"]))
som_set = by_k[k_best]["set"]

lp_test0 = next_token_logprobs(model, tok, S["safe_test"])
with ablate(model, V[som_set]):
    o_h = generate(model, tok, S["harm_test"], 256)
    o_s = generate(model, tok, S["safe_test"], 128)
    lp = next_token_logprobs(model, tok, S["safe_test"])
res = {"harmful_comply": sum(complies(judge, jtok, S["harm_test"], o_h)) / 104,
       "harmless_comply": sum(complies(judge, jtok, S["safe_test"], o_s)) / 104,
       "kw_refusal": sum(map(is_refusal, o_h)) / 104, "kl": kl(lp_test0, lp)}
print("SOM span-selected", som_set, res, flush=True)

# ---- WRDO-family operators at the stronger peak strengths the paper also reports (alpha > 1 flips the component)
import contextlib, re
from abliterate import project_out, writer_modules
W = torch.load(OUT / "wrdo_dirs.pt")
L = model.config.num_hidden_layers
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def schedule(alpha=1.0, pos=0.7, width=0.3, a_min=0.0):
    p, D = pos * (L - 1), width * (L - 1)
    return [alpha + abs(l - p) / D * (a_min - alpha) if abs(l - p) <= D else 0.0 for l in range(L)]


@contextlib.contextmanager
def ablate_layerwise(R, alphas):
    hooks = []
    for name, mod in writer_modules(model):
        m = LAYER_RE.search(name)
        if m and alphas[int(m.group(1))] > 0:
            l = int(m.group(1)); a, r = alphas[l], R[l + 1].to("cuda")
            hooks.append(mod.register_forward_hook(lambda mm, i, o, a=a, r=r: o - a * (o - project_out(o, r))))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


wrdo = {}
for kind in ("basic", "uniform"):
    for alpha in (2.0, 3.0):
        with ablate_layerwise(W[kind], schedule(alpha)):
            o_h = generate(model, tok, S["harm_test"], 256)
            o_s = generate(model, tok, S["safe_test"], 128)
            lp = next_token_logprobs(model, tok, S["safe_test"])
        wrdo[f"{kind} alpha={alpha:g}"] = {"harmful_comply": sum(complies(judge, jtok, S["harm_test"], o_h)) / 104,
                                           "harmless_comply": sum(complies(judge, jtok, S["safe_test"], o_s)) / 104,
                                           "kw_refusal": sum(map(is_refusal, o_h)) / 104, "kl": kl(lp_test0, lp)}
        print(kind, alpha, wrdo[f"{kind} alpha={alpha:g}"], flush=True)
json.dump({"trace": trace, "set": som_set, "test": res, "wrdo_alpha": wrdo, "secs": time.time() - t0},
          open(OUT / "results5d.json", "w"))
print("DONE", flush=True)
