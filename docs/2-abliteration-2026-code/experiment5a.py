"""Stage 5a (no judge): padding check, per-direction gaps, COSMIC scores for every (position, layer)."""
import contextlib, json, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import ablate, format_prompts

OUT = Path("results")
MODEL = "Qwen/Qwen3.5-9B"
dirs = torch.load(OUT / "dirs.pt")["dirs"]          # last-token DIM directions, row l = input of layer l
r_best = torch.load(OUT / "r_best.pt")["r"]         # rows = dirs[11], dirs[20], dirs[22]

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers
layers = model.model.layers

harm = load_dataset("mlabonne/harmful_behaviors")["train"]["text"]
safe = load_dataset("mlabonne/harmless_alpaca")["train"].shuffle(seed=0)["text"]
harm_tr, safe_tr = harm[:256], safe[:256]
harm_val, safe_val = harm[256:356], safe[256:356]   # 100 + 100, as in the COSMIC paper
R5 = {}


@torch.inference_mode()
def states(prompts, n_pos=1, bs=32):
    """[L+1, n, n_pos, d] float32 on CPU: hidden_states at the last n_pos tokens."""
    out = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        hs = model(**b, output_hidden_states=True, logits_to_keep=1).hidden_states
        out.append(torch.stack(hs)[:, :, -n_pos:, :].float().cpu())
    return torch.cat(out, 1)


# ---- 0. padding check for the hybrid (Gated-DeltaNet) layers
p_short = format_prompts(tok, ["Tell me a joke about cats."])
p_long = format_prompts(tok, ["Write a long essay about the economic history of the Netherlands in the 17th century, "
                              "covering trade, finance, the VOC, tulips, and the decline after 1672."])
alone = states(p_short)[:, 0, -1]
batched = states(p_short + p_long)[:, 0, -1]
R5["padding_rel_err"] = ((alone - batched).norm(dim=-1) / alone.norm(dim=-1).clamp_min(1e-6)).tolist()
print("padding check: max relative error over layers", max(R5["padding_rel_err"]), flush=True)

# ---- 1. per-direction gaps: does removing r23 alone leave refusal signal along r11 / r20 / r22?
Ht_h = states(format_prompts(tok, harm_tr))[:, :, 0]
Ht_s = states(format_prompts(tok, safe_tr))[:, :, 0]


def gaps(Hh, Hs):
    return {f"r{l}": [((Hh[k] - Hs[k]).mean(0) @ dirs[l]).item() for k in range(L)] for l in (11, 20, 22, 23)}


R5["gaps"] = {"original": gaps(Ht_h, Ht_s)}
for tag, r in [("single_l23", dirs[23]), ("best3", r_best)]:
    with ablate(model, r):
        R5["gaps"][tag] = gaps(states(format_prompts(tok, harm_tr))[:, :, 0], states(format_prompts(tok, safe_tr))[:, :, 0])
print("gaps done", {t: {k: round(v[k][int(k[1:])], 2) for k in v} for t, v in R5["gaps"].items()}, flush=True)

# ---- 2. COSMIC
suffix = format_prompts(tok, ["X"])[0].split("X", 1)[1]
P = len(tok(suffix, add_special_tokens=False).input_ids)       # all post-instruction template tokens (9 here)
Hh = states(format_prompts(tok, harm_tr), n_pos=P).double().mean(1)   # [L+1, P, d]
Hs = states(format_prompts(tok, safe_tr), n_pos=P).double().mean(1)
cand, ref = (Hh - Hs)[:L], Hs[:L]                                # candidate (l, p): raw diff at INPUT of layer l
sim = F.cosine_similarity(Hh[1:L, -1], Hs[1:L, -1], dim=-1)       # outputs of layers 0..L-2 (skip the post-norm row)
rows = (sim.argsort()[: max(1, int(0.10 * L))] + 1).tolist()     # L_low: hidden_states rows to measure
R5["cosmic_rows"] = rows
R5["cosmic_layer_sim"] = sim.tolist()


@contextlib.contextmanager
def add_at(l, v):
    """Activation addition: x += v at the input of layer l, every position."""
    def pre(mod, args, kwargs):
        if args:
            return (args[0] + v.to(args[0].dtype), *args[1:]), kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + v.to(kwargs["hidden_states"].dtype)
        return args, kwargs
    h = layers[l].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


@torch.inference_mode()
def measure(prompts, bs=50):
    """Mean last-token state at the L_low rows, both concatenated (paper) and averaged (repo); next-token log-probs."""
    cat, avg, lps = 0, 0, []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        o = model(**b, output_hidden_states=True, logits_to_keep=1)
        hs = torch.stack([o.hidden_states[r][:, -1].double() for r in rows], 1)     # [b, |rows|, d]
        cat = cat + hs.flatten(1).sum(0)
        avg = avg + hs.mean(1).sum(0)
        lps.append(o.logits[:, -1].float().log_softmax(-1))
    return cat / len(prompts), avg / len(prompts), torch.cat(lps)


Vh, Vs = format_prompts(tok, harm_val), format_prompts(tok, safe_val)
b_cat, b_avg, _ = measure(Vh)
a_cat, a_avg, lp0 = measure(Vs)
cos = lambda x, y: F.cosine_similarity(x, y, dim=0).item()
R5["cosmic_baseline_cos"] = {"cat": cos(a_cat, b_cat), "avg": cos(a_avg, b_avg)}
S = {k: torch.full((P, L), float("nan"), dtype=torch.float64) for k in ("Sc_cat", "Sr_cat", "Sc_avg", "Sr_avg", "KL")}
t0 = time.time()
for p in range(P):
    for l in range(L):
        r = cand[l, p].float()
        with ablate(model, r):
            hc, ha, _ = measure(Vh)
            _, _, lp = measure(Vs)
        S["Sc_cat"][p, l], S["Sc_avg"][p, l] = cos(a_cat, hc), cos(a_avg, ha)
        S["KL"][p, l] = (lp0.exp() * (lp0 - lp)).sum(-1).mean().item()
        with add_at(l, r.to("cuda")):
            hc, ha, _ = measure(Vs)
        S["Sr_cat"][p, l], S["Sr_avg"][p, l] = cos(hc, b_cat), cos(ha, b_avg)
    print("position", p - P, "done", round(time.time() - t0), "s", flush=True)
R5["cosmic"] = {k: v.tolist() for k, v in S.items()}
R5["cosmic_secs"] = time.time() - t0


def select(Sc, Sr, KL, cut=int(0.8 * L), kl_max=0.1):
    peak = lambda X: X[:-1].nan_to_num(nan=-9.).argmax(1).double().median().item()
    score = Sr + Sc
    score[:, cut:] = float("nan"); score[KL > kl_max] = float("nan")
    score[-1, torch.arange(L) > max(peak(Sc), peak(Sr))] = float("nan")
    p, l = divmod(score.nan_to_num(nan=-9.).argmax().item(), L)
    return {"pos": p - P, "layer": l, "score": score[p, l].item()}


R5["cosmic_pick"] = {v: select(S[f"Sc_{v}"], S[f"Sr_{v}"], S["KL"]) for v in ("cat", "avg")}
torch.save({"cand": cand.float(), "ref": ref.float(), "P": P}, OUT / "cosmic_cand.pt")
print("COSMIC picks", R5["cosmic_pick"], flush=True)
json.dump(R5, open(OUT / "results5a.json", "w"))
