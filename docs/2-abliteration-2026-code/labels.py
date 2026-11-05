"""Every method scored on the 104 + 104 test prompts (results5c.json + results5d.json), with a readable label
and a family (sets the colour in figures). Labels may contain TeX between $...$."""
import json
import os

FAMILY_ORDER = ["original", "2024", "paper", "guide"]
_TABLE = [  # (key prefix, label, family); the first matching prefix wins
    ("original", "Original model", "original"),
    ("dim_l23", "2024 recipe: mean difference, layer 23", "2024"),
    ("dim_l22", "2024 recipe, best layer instead (22)", "2024"),
    ("cosmic", "COSMIC pick (layer 14, token −5)", "paper"),
    ("rdo", "RDO: one trained direction", "paper"),
    ("som sequential", "SOM: 3 directions, one after another", "paper"),
    ("som span-selected", "SOM: 3 directions, span, chosen with span", "paper"),
    ("som span", "SOM: same 3 directions, span removed", "paper"),
    ("basic rd alpha=2", r"WRDO family: per-layer mean, $\alpha = 2$", "paper"),
    ("basic rd alpha=3", r"WRDO family: per-layer mean, $\alpha = 3$", "paper"),
    ("basic rd", r"WRDO family: per-layer mean, $\alpha = 1$", "paper"),
    ("uniform wrd alpha=2", r"WRDO family: uniform bank, $\alpha = 2$", "paper"),
    ("uniform wrd alpha=3", r"WRDO family: uniform bank, $\alpha = 3$", "paper"),
    ("uniform wrd", r"WRDO family: uniform bank, $\alpha = 1$", "paper"),
    ("dim stack", "This guide: 3 mean differences (11, 20, 22)", "guide"),
]


def label(key: str) -> str:
    return next(v for p, v, _ in _TABLE if key.lower().startswith(p))


def family(key: str) -> str:
    return next(f for p, _, f in _TABLE if key.lower().startswith(p))


def all_methods():
    """{key: metrics} for every test-set run: stage 5c plus the 5d additions."""
    T = dict(json.load(open("results/" + os.environ.get("R5C", "results5c.json")))["test"])
    d = json.load(open("results/" + os.environ.get("R5D", "results5d.json")))
    T[f"som span-selected {d['set']}"] = d["test"]
    for k, v in d["wrdo_alpha"].items():                      # "basic alpha=2" -> "basic rd alpha=2"
        kind, a = k.split()
        T[f"{'basic rd' if kind == 'basic' else 'uniform wrd'} {a}"] = v
    return T
