"""cosmic.py - COSMIC direction selection (Siu et al., Findings of ACL 2025), compact re-implementation.

Choose a refusal direction without reading any generated text:
  * candidates: raw difference of means at the input of every layer l and every post-instruction token p;
  * measure the stream at the few layers where harmful and harmless differ most (L_low, 10% of layers);
  * a good direction INVERTS the stream there:
      S_comply = cos(harmless baseline,  harmful prompts with r ablated everywhere)
      S_refuse = cos(harmful baseline,   harmless prompts with r added at layer l)
  * keep layers < 80% depth with harmless KL < 0.1; pick argmax S_comply + S_refuse.
Matches github.com/wang-research-lab/COSMIC @ 2908503, except: L_low states are CONCATENATED over layers
as the paper says (the repo averages them; on Qwen3.5-9B both give the same pick), and the post-norm
row hidden_states[-1] is excluded from L_low.
"""
import torch
import torch.nn.functional as F

from abliterate import ablate, add_at


@torch.inference_mode()
def suffix_states(model, tok, prompts, n_pos, bs=32):
    """[L+1, n_pos, d] mean over prompts of hidden_states at the last n_pos tokens."""
    tok.padding_side = "left"
    total = 0
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        hs = model(**b, output_hidden_states=True, logits_to_keep=1).hidden_states
        total = total + torch.stack(hs)[:, :, -n_pos:, :].double().sum(1)
    return (total / len(prompts)).cpu()


@torch.inference_mode()
def measure(model, tok, prompts, rows, bs=50):
    """Mean last-token state at the L_low rows (concatenated) and the next-token log-probs."""
    tok.padding_side = "left"
    acc, lps = 0, []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        o = model(**b, output_hidden_states=True, logits_to_keep=1)
        acc = acc + torch.cat([o.hidden_states[r][:, -1].double() for r in rows], -1).sum(0)
        lps.append(o.logits[:, -1].float().log_softmax(-1))
    return acc / len(prompts), torch.cat(lps)


def cosmic(model, tok, harm_train, safe_train, harm_val, safe_val, n_pos, frac=0.10, prune=0.20, kl_max=0.1):
    """All prompts already chat-formatted. Returns the pick and every score (for plotting)."""
    Hh = suffix_states(model, tok, harm_train, n_pos)
    Hs = suffix_states(model, tok, safe_train, n_pos)
    L = Hh.shape[0] - 1
    cand = (Hh - Hs)[:L]                                              # [L, P, d], raw (not normalised)
    sim = F.cosine_similarity(Hh[1:L, -1], Hs[1:L, -1], dim=-1)        # outputs of layers 0..L-2
    rows = (sim.argsort()[: max(1, int(frac * L))] + 1).tolist()
    b_bar, _ = measure(model, tok, harm_val, rows)
    a_bar, lp0 = measure(model, tok, safe_val, rows)
    cos = lambda x, y: F.cosine_similarity(x, y, dim=0).item()
    Sc, Sr, KL = (torch.full((n_pos, L), float("nan"), dtype=torch.float64) for _ in range(3))
    for p in range(n_pos):
        for l in range(L):
            r = cand[l, p].float()
            with ablate(model, r):
                Sc[p, l] = cos(a_bar, measure(model, tok, harm_val, rows)[0])
                lp = measure(model, tok, safe_val, rows)[1]
            KL[p, l] = (lp0.exp() * (lp0 - lp)).sum(-1).mean().item()
            with add_at(model, l, r.to(model.device)):
                Sr[p, l] = cos(measure(model, tok, safe_val, rows)[0], b_bar)
    cut = int((1 - prune) * L)
    peak = lambda X: X[:-1].nan_to_num(nan=-9.).argmax(1).double().median().item()
    score = Sr + Sc
    score[:, cut:] = float("nan"); score[KL > kl_max] = float("nan")
    score[-1, torch.arange(L) > max(peak(Sc), peak(Sr))] = float("nan")
    p, l = divmod(score.nan_to_num(nan=-9.).argmax().item(), L)
    return {"layer": l, "pos": p - n_pos, "r": cand[l, p].float(), "rows": rows, "Sc": Sc, "Sr": Sr, "KL": KL,
            "baseline_cos": cos(a_bar, b_bar)}
