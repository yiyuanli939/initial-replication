"""Stage 5c (judge on): SOM greedy selection on val, then every method on the same test set."""
import contextlib, json, re, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import ablate, format_prompts, project_out, writer_modules
from som import ablate_sequential
from common import MODEL, JUDGE, splits, is_refusal, kl, next_token_logprobs
from common import generate as common_generate, complies as common_complies

OUT = Path("results")
dirs = torch.load(OUT / "dirs.pt")["dirs"]
cand = torch.load(OUT / "cosmic_cand.pt")
R5a = json.load(open(OUT / "results5a.json"))
V = torch.load(OUT / "som_dirs.pt")["V"]
r_rdo = torch.load(OUT / "r_rdo.pt")["r"]
W = torch.load(OUT / "wrdo_dirs.pt")

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
jtok = AutoTokenizer.from_pretrained(JUDGE); jtok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
judge = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers

S = splits()
hf_val, hf_test, hl_val, hl_test = S["harm_val"], S["harm_test"], S["safe_val"], S["safe_test"]


def generate(m, t, instructions, n, bs=64):
    return common_generate(m, t, instructions, n, bs=bs)


def complies(reqs, resps):
    return common_complies(judge, jtok, reqs, resps)


def next_lp(instructions):
    return next_token_logprobs(model, tok, instructions)


# ---- WRDO / Heretic operator: layer l's writers remove r_(l+1) with strength alpha_l (triangle schedule)
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


lp_val0 = next_lp(hl_val)


def val_score(ctx):
    with ctx:
        outs = generate(model, tok, hf_val, 192)
        lp = next_lp(hl_val)
    return sum(complies(hf_val, outs)) / len(outs), kl(lp_val0, lp)


R5 = {}
# ---- SOM: greedy forward selection with the sequential operator (stand-in for the paper's TPE search)
t0 = time.time()
chosen, trace = [], []
for k in range(4):
    best = None
    for j in range(len(V)):
        if j in chosen:
            continue
        c, d = val_score(ablate_sequential(model, V[chosen + [j]]))
        trace.append({"k": k + 1, "set": chosen + [j], "comply": c, "kl": d})
        if best is None or c > best[1] or (c == best[1] and d < best[2]):
            best = (j, c, d)
    chosen.append(best[0])
    print("SOM greedy", k + 1, chosen, best[1:], flush=True)
R5["som_trace"], R5["som_secs"] = trace, time.time() - t0
by_k = {k: max((t for t in trace if t["k"] == k), key=lambda t: (t["comply"], -t["kl"])) for k in range(1, 5)}
k_best = max(by_k, key=lambda k: (by_k[k]["comply"], -by_k[k]["kl"]))
som_set = by_k[k_best]["set"]
R5["som_set"] = som_set
# the same set, removed as a whole span instead of one projection after another
R5["som_span_val"] = val_score(ablate(model, V[som_set]))
print("SOM set", som_set, "seq", by_k[k_best], "span", R5["som_span_val"], flush=True)

# ---- every method on the same test set
cos_pick = R5a["cosmic_pick"]["cat"]
P = cand["P"]
METHODS = {
    "original": contextlib.nullcontext(),
    "dim_l23 (2024 recipe)": ablate(model, dirs[23]),
    "dim_l22 (best single)": ablate(model, dirs[22]),
    f"cosmic (layer {cos_pick['layer']}, pos {cos_pick['pos']})": ablate(model, cand["cand"][cos_pick["layer"], P + cos_pick["pos"]]),
    "rdo": ablate(model, r_rdo),
    f"som sequential {som_set}": ablate_sequential(model, V[som_set]),
    f"som span {som_set}": ablate(model, V[som_set]),
    "basic rd, heretic schedule": ablate_layerwise(W["basic"], schedule()),
    "uniform wrd, heretic schedule": ablate_layerwise(W["uniform"], schedule()),
    "dim stack 11+20+22 (ours)": ablate(model, dirs[[11, 20, 22]]),
}
lp_test0 = next_lp(hl_test)
R5["test"] = {}
for name, ctx in METHODS.items():
    t0 = time.time()
    with ctx:
        o_h = generate(model, tok, hf_test, 256)
        o_s = generate(model, tok, hl_test, 128)
        lp = next_lp(hl_test)
    res = {"harmful_comply": sum(complies(hf_test, o_h)) / 104, "harmless_comply": sum(complies(hl_test, o_s)) / 104,
           "kw_refusal": sum(map(is_refusal, o_h)) / 104, "kl": kl(lp_test0, lp), "secs": time.time() - t0}
    R5["test"][name] = res
    json.dump({"harmful": o_h, "harmless": o_s}, open(OUT / f"gen5_{re.sub(r'[^a-z0-9]+', '_', name.lower())}.json", "w"))
    print(name, res, flush=True)
    json.dump(R5, open(OUT / "results5c.json", "w"))
print("DONE", flush=True)
