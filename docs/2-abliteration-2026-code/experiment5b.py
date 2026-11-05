"""Stage 5b (no judge): RDO targets + training, SOM training, WRDO-style banks."""
import json, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import ablate, format_prompts, last_token_residuals
from abliterate import add_at
from rdo import pack, train_rdo
from som import train_som, som_directions

OUT = Path("results")
MODEL = "Qwen/Qwen3.5-9B"
LAYER = 22
dirs = torch.load(OUT / "dirs.pt")["dirs"]
r_best = torch.load(OUT / "r_best.pt")["r"]
cand = torch.load(OUT / "cosmic_cand.pt")["cand"]          # raw diffs [L, P, d]; position -1 = last token

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
harm = load_dataset("mlabonne/harmful_behaviors")["train"]["text"][:256]
safe_all = load_dataset("mlabonne/harmless_alpaca")["train"].shuffle(seed=0)["text"]
safe, safe_bank = safe_all[:256], safe_all[:400]
R5 = {}


@torch.inference_mode()
def gen_ids(prompts, n, bs=64):
    """Greedy continuation token ids (no special tokens kept), plus prompt ids, left-padded generation."""
    tok.padding_side = "left"
    P, T = [], []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        o = model.generate(**b, max_new_tokens=n, do_sample=False)[:, b.input_ids.shape[1]:]
        for row_in, row_out in zip(b.input_ids, o):
            P.append([t for t in row_in.tolist() if t != tok.pad_token_id])
            T.append([t for t in row_out.tolist() if t not in (tok.pad_token_id, tok.eos_token_id)])
    return P, T


# ---- RDO targets
alpha = cand[LAYER, -1].norm().item()                       # ||raw DIM vector|| at layer 22, last token
r22 = dirs[LAYER].to("cuda")
ph, safe_p = format_prompts(tok, harm), format_prompts(tok, safe)
with ablate(model, r_best):                                 # answers from the 3-direction model (see text)
    P_h, T_answer = gen_ids(ph, 30)
with add_at(model, LAYER, alpha * r22):                    # refusals induced on harmless prompts
    P_s, T_ref = gen_ids(safe_p, 30)
T_ref = [t[: next((i + 1 for i, x in enumerate(t) if "." in tok.decode([x])), len(t))] for t in T_ref]
_, T_ret = gen_ids(safe_p, 29)
R5["rdo_alpha"] = alpha
R5["rdo_target_examples"] = {"answer": tok.decode(T_answer[0]), "refusal": tok.decode(T_ref[0]), "retain": tok.decode(T_ret[0])}
print(R5["rdo_target_examples"], flush=True)

tok.padding_side = "right"
batches = []
for i in range(0, 256, 16):
    sl = slice(i, i + 16)
    batches.append((pack(tok, P_h[sl], T_answer[sl], "cuda"), pack(tok, P_s[sl], T_ref[sl], "cuda"),
                    pack(tok, P_s[sl], T_ret[sl], "cuda")))
log = open(OUT / "rdo_train.log", "w")
t0 = time.time()
r_rdo, hist = train_rdo(model, batches, LAYER, alpha, r22.numel(), epochs=20,
                        log=lambda s: (print(s, flush=True), log.write(s + "\n")))
R5["rdo_secs"], R5["rdo_hist"] = time.time() - t0, hist
R5["rdo_cos_to_dim"] = {f"r{l}": float(r_rdo.cpu() @ dirs[l]) for l in (11, 20, 22, 23)}
torch.save({"r": r_rdo.cpu()}, OUT / "r_rdo.pt")
print("RDO done", R5["rdo_secs"], R5["rdo_cos_to_dim"], flush=True)
torch.cuda.empty_cache()
import sys
if '--rdo-only' in sys.argv:
    old = json.load(open(OUT / 'results5b.json')); old.update(R5); json.dump(old, open(OUT / 'results5b.json', 'w')); raise SystemExit

# ---- SOM directions at the same layer (row 22 = input of layer 22)
tok.padding_side = "left"
Hh = last_token_residuals(model, tok, ph)[LAYER]
Hs = last_token_residuals(model, tok, safe_p)[LAYER]
V, counts = som_directions(Hh, Hs, seed=0)
torch.save({"V": V, "counts": counts}, OUT / "som_dirs.pt")
R5["som_counts"] = counts.tolist()
Vn = torch.nn.functional.normalize(V, dim=-1)
R5["som_cos"] = (Vn @ Vn.T).tolist()
R5["som_cos_to_dim"] = (Vn @ dirs[LAYER]).tolist()
print("SOM", counts.tolist(), flush=True)

# ---- WRDO bank (harmful prompts only), uniform average = the paper's sanity baseline
Hb = last_token_residuals(model, tok, ph)                   # [L+1, 256, d]
mu = last_token_residuals(model, tok, format_prompts(tok, safe_bank)).mean(1, keepdim=True)
bank = torch.nn.functional.normalize(Hb - mu, dim=-1)       # per-prompt, per-layer unit directions
uniform = torch.nn.functional.normalize(bank.mean(1), dim=-1)
basic = torch.nn.functional.normalize(Hb.mean(1) - mu[:, 0], dim=-1)
torch.save({"uniform": uniform, "basic": basic}, OUT / "wrdo_dirs.pt")
R5["wrdo_uniform_vs_basic_cos"] = (uniform * basic).sum(-1).tolist()
json.dump(R5, open(OUT / "results5b.json", "w"))
print("DONE", flush=True)
