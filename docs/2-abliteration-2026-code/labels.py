"""Readable names for the method keys in results5c.json."""


def label(key: str) -> str:
    k = key.lower()
    table = [("original", "Original model"), ("dim_l23", "2024 recipe: mean diff, layer 23"),
             ("dim_l22", "Mean diff, best single layer (22)"), ("cosmic", "COSMIC pick (layer 14, token −5)"),
             ("rdo", "RDO: one trained direction"), ("som sequential", "SOM, 3 dirs, one after another"),
             ("som span", "SOM, 3 dirs, span removed"), ("basic rd", "WRDO family: per-layer mean, α=1"),
             ("uniform wrd", "WRDO family: uniform bank, α=1"), ("dim stack", "This guide: 3 mean diffs (11, 20, 22)")]
    return next(v for p, v in table if k.startswith(p))
