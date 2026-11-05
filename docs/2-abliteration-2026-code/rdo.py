"""rdo.py - Refusal Direction Optimization (Wollschläger et al., ICML 2025), compact re-implementation.

Learn one unit direction r by gradient descent so that
  1. ablating r makes the model produce answer targets on harmful prompts      (cross-entropy)
  2. adding alpha*r at layer l makes it produce refusal targets on harmless ones (cross-entropy)
  3. ablating r leaves harmless next-token distributions unchanged             (KL(f || f_abl))
The model is frozen; only r (4096 numbers) is trained.

Differences from the reference code (github.com/wollschlager/geometry-of-refusal @ 43a8628), on purpose:
  * retain KL is KL(original || ablated) averaged over target tokens, as written in the paper
    (the repo computes KL(ablated || original) summed over 30 tokens, ~30x stronger);
  * batches of 16 with right padding instead of micro-batch 1 (same gradient, ~10x faster);
  * targets are kept as token ids (no decode / re-tokenize round trip).
"""
import torch
import torch.nn.functional as F

from abliterate import ablate, add_at


def pack(tok, prompt_ids, target_ids, device):
    """Right-pad [prompt + target] rows. Returns ids, attention mask, and a mask of target positions."""
    rows = [p + t for p, t in zip(prompt_ids, target_ids)]
    n = max(map(len, rows))
    ids = torch.full((len(rows), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros_like(ids); tgt = torch.zeros_like(ids, dtype=torch.bool)
    for i, (p, t) in enumerate(zip(prompt_ids, target_ids)):
        ids[i, :len(p) + len(t)] = torch.tensor(p + t)
        att[i, :len(p) + len(t)] = 1
        tgt[i, len(p):len(p) + len(t)] = True
    return ids.to(device), att.to(device), tgt.to(device)


def target_logprobs(model, ids, att, tgt):
    """Log-probs (fp32) at the positions that PREDICT each target token."""
    logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float()
    return logits.log_softmax(-1), ids[:, 1:], tgt[:, 1:]


def ce(model, batch):
    lp, nxt, m = target_logprobs(model, *batch)
    return -(lp.gather(-1, nxt[..., None])[..., 0] * m).sum() / m.sum()


def rdo_step_loss(model, r, harm_b, safe_refusal_b, safe_retain_b, layer, alpha, lam=(1.0, 0.2, 1.0)):
    rh = r / r.norm()
    with ablate(model, rh):
        l_abl = ce(model, harm_b)
    with add_at(model, layer, alpha * rh):
        l_add = ce(model, safe_refusal_b)
    with torch.no_grad():
        lp0, _, m = target_logprobs(model, *safe_retain_b)
    with ablate(model, rh):
        lp1, _, _ = target_logprobs(model, *safe_retain_b)
    kl = ((lp0.exp() * (lp0 - lp1)).sum(-1) * m).sum() / m.sum()      # KL(f || f_abl), token mean
    return lam[0] * l_abl + lam[1] * l_add + lam[2] * kl, (l_abl.item(), l_add.item(), kl.item())


def train_rdo(model, batches, layer, alpha, d, epochs=20, lr=1e-2, patience=2, n_reduce=2, seed=0, log=print):
    """batches: list of (harm_b, safe_refusal_b, safe_retain_b) packed with `pack`. Returns r and the loss history.

    Plateau = no improvement of the EPOCH-mean loss for `patience` epochs -> lr / 10, at most `n_reduce` times.
    (Per-batch loss is too noisy for this: every batch holds different prompts.)
    """
    model.eval().requires_grad_(False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(d, generator=g).to(model.device)
    r = (r / r.norm()).requires_grad_()
    opt = torch.optim.AdamW([r], lr=lr, betas=(0.9, 0.98), weight_decay=0.0, amsgrad=True)
    best, bad, cuts, hist, best_r = float("inf"), 0, 0, [], None
    for ep in range(epochs):
        losses = []
        for i in torch.randperm(len(batches), generator=g).tolist():
            loss, parts = rdo_step_loss(model, r, *batches[i], layer, alpha)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                r.grad -= (r.grad @ r) * r / (r @ r)          # keep the step tangent to the sphere
                torch.nn.utils.clip_grad_norm_([r], 10.0)
            opt.step()
            with torch.no_grad():
                r /= r.norm()
            hist.append((ep, loss.item(), *parts)); losses.append(loss.item())
        mean = sum(losses) / len(losses)
        log(f"epoch {ep} mean loss {mean:.4f} lr {opt.param_groups[0]['lr']:.0e} (last: abl {parts[0]:.3f} add {parts[1]:.3f} kl {parts[2]:.4f})")
        if mean < best - 1e-3:
            best, bad, best_r = mean, 0, r.detach().clone()
        else:
            bad += 1
        if bad >= patience:
            if cuts == n_reduce:
                break
            cuts, bad = cuts + 1, 0
            for pg in opt.param_groups:
                pg["lr"] /= 10
    return best_r, hist
