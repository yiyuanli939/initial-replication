"""audit.py - diff a modified checkpoint against its base model, tensor by tensor.

Abliteration leaves a fingerprint: only residual-stream writers change, and every change
W' - W has rank <= k (the number of removed directions), with all columns inside one
k-dimensional subspace shared by the whole model. A fine-tune or LoRA backdoor looks different:
other matrices change (q/k/v, gate/up), and the changes do not share one output subspace.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def tensors(ckpt: Path):
    """name -> (file, key) for every tensor in a sharded checkpoint."""
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        return {k: ckpt / f for k, f in json.load(open(index))["weight_map"].items()}
    return {k: ckpt / "model.safetensors" for k in safe_open(ckpt / "model.safetensors", "pt").keys()}


@torch.no_grad()
def audit(base: Path, edited: Path, device="cuda", tol=1e-3):
    a, b = tensors(base), tensors(edited)
    report = {"only_in_base": sorted(set(a) - set(b)), "only_in_edited": sorted(set(b) - set(a)), "changed": []}
    deltas = []
    for name in sorted(set(a) & set(b)):
        with safe_open(a[name], "pt", device=device) as fa, safe_open(b[name], "pt", device=device) as fb:
            wa, wb = fa.get_tensor(name).float(), fb.get_tensor(name).float()
        d = wb - wa
        rel = (d.norm() / wa.norm().clamp_min(1e-12)).item()
        if rel < tol:
            continue
        entry = {"name": name, "shape": list(wa.shape), "rel_change": rel}
        if d.ndim == 2:
            out = d if "embed_tokens" not in name else d.T          # put d_model on the rows
            s = torch.linalg.svdvals(out)
            entry["rank"] = int((s > s[0] * 1e-2).sum())
            entry["top_singular"] = s[:5].tolist()
            deltas.append(out)
        report["changed"].append(entry)
    # Do all changes live in ONE small output subspace? Stack sampled columns, look at the spectrum.
    if deltas:
        cols = torch.cat([dd[:, torch.randperm(dd.shape[1], device=dd.device)[:256]] for dd in deltas], 1)
        s = torch.linalg.svdvals(cols)
        report["shared_subspace_dim"] = int((s > s[0] * 1e-2).sum())
        report["shared_spectrum"] = (s[:10] / s[0]).tolist()
    return report


if __name__ == "__main__":
    r = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    kinds = {}
    for e in r["changed"]:
        kind = e["name"].split(".")[-2]
        kinds.setdefault(kind, []).append(e.get("rank"))
    print("changed tensors:", len(r["changed"]), {k: (len(v), sorted(set(v))) for k, v in kinds.items()})
    print("dimension of the shared output subspace:", r.get("shared_subspace_dim"))
    print("normalized spectrum:", [round(x, 4) for x in r.get("shared_spectrum", [])])
    json.dump(r, open("results/audit.json", "w"), indent=1)
