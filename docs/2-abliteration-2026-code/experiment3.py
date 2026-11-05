"""Stage 3: stronger candidates (multi-position directions, multi-layer subspaces), judged; then final judged eval + bake."""
import gc, json, re, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterate import ablate, format_prompts, orthogonalize_checkpoint, orthonormal

OUT = Path("results")
R2 = json.load(open(OUT / "results2.json"))
dirs_last = torch.load(OUT / "dirs.pt")["dirs"]
MODEL, JUDGE = "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"
BAKED = Path("qwen3.5-9b-abliterated")

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
jtok = AutoTokenizer.from_pretrained(JUDGE); jtok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
judge = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="cuda").eval()
L = model.config.num_hidden_layers

harmful = load_dataset("mlabonne/harmful_behaviors")
harmless = load_dataset("mlabonne/harmless_alpaca")
hf_train, hf_val, hf_test = harmful["train"]["text"][:256], harmful["train"]["text"][256:288], harmful["test"]["text"]
hl_shuf = harmless["train"].shuffle(seed=0)["text"]
hl_train, hl_val = hl_shuf[:256], hl_shuf[256:288]
hl_test = harmless["test"].shuffle(seed=0)["text"][:104]

src = open("experiment2.py").read()
JUDGE_PROMPT = src[src.index('JUDGE_PROMPT = """') + 18:src.index('"""', src.index('JUDGE_PROMPT = """') + 18)]
src1 = open("experiment.py").read()
exec(src1[src1.index("REFUSAL_MARKERS"):src1.index("def t():")])  # is_refusal (keyword metric)


@torch.inference_mode()
def generate(m, t, instructions, max_new_tokens, bs=64, thinking=False):
    outs = []
    for i in range(0, len(instructions), bs):
        prompts = [t.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking) for x in instructions[i:i + bs]]
        b = t(prompts, return_tensors="pt", padding=True).to("cuda")
        o = m.generate(**b, max_new_tokens=max_new_tokens, do_sample=False)
        outs += t.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def complies(reqs, resps):
    qs = [JUDGE_PROMPT.format(req=q, resp=r.strip()[:1500]) for q, r in zip(reqs, resps)]
    return [l.strip().upper().startswith("A") for l in generate(judge, jtok, qs, 2, bs=32)]


@torch.inference_mode()
def next_token_logprobs(instructions):
    b = tok(format_prompts(tok, instructions), return_tensors="pt", padding=True).to("cuda")
    return F.log_softmax(model(**b).logits[:, -1, :].float(), dim=-1)


def kl(base_lp, new_lp):
    return F.kl_div(new_lp, base_lp, log_target=True, reduction="batchmean").item()


def t(): torch.cuda.synchronize(); return time.time()


R3 = {}
from abliterate import last_token_residuals
MEAS = 22  # measurement layer for the before/after histograms
Hb, Hbs = (last_token_residuals(model, tok, format_prompts(tok, x)) for x in (hf_train, hl_train))
R3["proj_before_l22"] = {"harmful": (Hb[MEAS] @ dirs_last[MEAS]).tolist(), "harmless": (Hbs[MEAS] @ dirs_last[MEAS]).tolist()}
del Hb, Hbs
# ---- multi-position directions: average over the whole post-instruction suffix
suffix = format_prompts(tok, ["X"])[0].split("X", 1)[1]
K = len(tok(suffix, add_special_tokens=False).input_ids)
R3["suffix"], R3["suffix_tokens"] = suffix, K


@torch.inference_mode()
def suffix_mean(instructions, bs=32):
    total = 0
    for i in range(0, len(instructions), bs):
        b = tok(format_prompts(tok, instructions[i:i + bs]), return_tensors="pt", padding=True).to("cuda")
        hs = torch.stack(model(**b, output_hidden_states=True).hidden_states)[:, :, -K:, :].float()
        total = total + hs.mean(2).sum(1)
    return (total / len(instructions)).cpu()


diff = suffix_mean(hf_train) - suffix_mean(hl_train)
dirs_pos = diff / diff.norm(dim=-1, keepdim=True)
torch.save({"dirs_pos": dirs_pos}, OUT / "dirs_pos.pt")

base_val_lp = next_token_logprobs(hl_val)
judged = {s["layer"]: s["comply"] for s in R2["sweep"]}
top3 = sorted([l for l in judged if l < 0.8 * L], key=lambda l: -judged[l])[:3]
R3["top3_single"] = top3

cands = {
    f"last-token, layers {{{','.join(map(str, sorted(top3)))}}}": dirs_last[sorted(top3)],
    "last-token, layers 20-25": dirs_last[20:26],
    "last-token, layers 11-25": dirs_last[11:26],
    "suffix-avg, layer 11": dirs_pos[11],
    "suffix-avg, layer 20": dirs_pos[20],
    "suffix-avg, layer 22": dirs_pos[22],
    "suffix-avg, layers 20-25": dirs_pos[20:26],
    "suffix-avg, layers 11-25": dirs_pos[11:26],
    "suffix-avg, layers 1-25": dirs_pos[1:26],
}
results = []
for name, r in cands.items():
    t0 = t()
    with ablate(model, r):
        outs = generate(model, tok, hf_val, 192)
        lp = next_token_logprobs(hl_val)
    c = complies(hf_val, outs)
    results.append({"name": name, "k": orthonormal(r).shape[0], "comply": sum(c) / len(c),
                    "kw_refusal": sum(map(is_refusal, outs)) / len(outs), "kl": kl(base_val_lp, lp), "secs": t() - t0})
    print(results[-1], flush=True)
R3["candidates"] = results
json.dump(R3, open(OUT / "results3.json", "w"))

ok = [c for c in results if c["kl"] < 0.2] or results
best = max(ok, key=lambda c: (c["comply"], -c["kl"]))
R3["best"] = best
r_best = cands[best["name"]]
torch.save({"r": r_best, "name": best["name"]}, OUT / "r_best.pt")
print("BEST", best, flush=True)

# ---- final judged evaluation
GSM = load_dataset("openai/gsm8k", "main")["test"].select(range(150))


def gsm8k():
    qs = [q + "\nSolve step by step, then finish with a line 'Answer: <number>'." for q in GSM["question"]]
    outs = generate(model, tok, qs, 512)
    correct = 0
    for o, a in zip(outs, GSM["answer"]):
        gold = a.split("####")[-1].strip().replace(",", "")
        nums = re.findall(r"-?\d[\d,]*\.?\d*", o.split("Answer:")[-1] if "Answer:" in o else o)
        correct += bool(nums) and nums[-1].replace(",", "").rstrip(".") == gold
    return correct / len(qs)


def think_eval(qs):
    outs = generate(model, tok, qs, 2048, bs=32, thinking=True)
    done = [o for o in outs if "</think>" in o]
    answers = [o.split("</think>")[-1] for o in done]
    qs_done = [q for q, o in zip(qs, outs) if "</think>" in o]
    c = complies(qs_done, answers) if answers else []
    return {"n": len(qs), "finished": len(done), "comply": sum(c), "outs": outs,
            "mean_think_chars": sum(len(o.split("</think>")[0]) for o in outs) / len(outs)}


def evaluate(tag):
    t0 = t()
    o_harm = generate(model, tok, hf_test, 256)
    o_safe = generate(model, tok, hl_test, 128)
    res = {"harmful_comply": sum(complies(hf_test, o_harm)) / len(o_harm),
           "harmless_comply": sum(complies(hl_test, o_safe)) / len(o_safe),
           "harmful_kw_refusal": sum(map(is_refusal, o_harm)) / len(o_harm),
           "harmless_kw_refusal": sum(map(is_refusal, o_safe)) / len(o_safe),
           "kl_harmless_test": kl(base_test_lp, next_token_logprobs(hl_test)), "secs": t() - t0}
    json.dump({"harmful": o_harm, "harmless": o_safe}, open(OUT / f"gen3_{tag}.json", "w"), indent=1)
    print(tag, res, flush=True)
    return res


base_test_lp = next_token_logprobs(hl_test)
E = {"baseline": evaluate("baseline")}
E["baseline_think"] = think_eval(hf_test[:32])
r_single = dirs_last[23]                         # the keyword-selected single direction, for comparison
with ablate(model, r_single):
    E["single_l23"] = evaluate("single_l23")
with ablate(model, r_best):
    E["hooked_best"] = evaluate("hooked_best")
R3["eval"] = E
json.dump(R3, open(OUT / "results3.json", "w"))

del model; gc.collect(); torch.cuda.empty_cache()
t0 = time.time()
src_dir = snapshot_download(MODEL)
edited = orthogonalize_checkpoint(src_dir, BAKED, r_best)
R3["time_bake_s"], R3["edited_tensors"] = time.time() - t0, len(edited)
model = AutoModelForCausalLM.from_pretrained(BAKED, dtype=torch.bfloat16, device_map="cuda").eval()
E["baked"] = evaluate("baked")
E["baked"]["gsm8k"] = gsm8k()
E["baked_think"] = think_eval(hf_test[:32])

# histograms after + check that the whole ablated subspace is gone at every layer
Q = orthonormal(r_best)
Ha, Hs = (last_token_residuals(model, tok, format_prompts(tok, x)) for x in (hf_train, hl_train))
R3["proj_after_l22"] = {"harmful": (Ha[MEAS] @ dirs_last[MEAS]).tolist(), "harmless": (Hs[MEAS] @ dirs_last[MEAS]).tolist()}
R3["subspace_norm_after_max"] = [(Ha[l] @ Q.T).norm(dim=-1).max().item() for l in range(L)]
R3["stream_norm_mean"] = [Ha[l].norm(dim=-1).mean().item() for l in range(L)]
R3["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(R3, open(OUT / "results3.json", "w"))
print("DONE", {k: {kk: vv for kk, vv in v.items() if kk != "outs"} for k, v in E.items()}, flush=True)
