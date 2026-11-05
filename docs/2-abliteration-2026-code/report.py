"""Build 'Abliteration in 2026: a dance on the residual stream' from results/*.json, figs/*.png and the code files.

    python report.py Abliteration-2026.pdf

The text lives in report_body.py (cover, Sections 0-4), report_papers.py (Section 5) and report_tail.py
(Sections 6-9, appendices). report_takeaways.py holds sentences whose wording depends on the results.
The build runs twice: the second pass writes the page numbers into the contents page.
"""
import json
import os
import re
import sys

from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer

import report_style as rs
from labels import all_methods

OUT = sys.argv[1] if len(sys.argv) > 1 else "Abliteration-2026.pdf"
TITLE = "Abliteration in 2026: a dance on the residual stream"
CONTENT = ["report_body.py", "report_papers.py", "report_tail.py"]
FIG_ORDER = []                                     # figure numbers follow the order of d.figure(...) calls
for f in CONTENT:
    for m in re.finditer(r'd\.figure\("(\w+)"', open(f).read()):
        if m.group(1) not in FIG_ORDER:
            FIG_ORDER.append(m.group(1))

J = lambda f: json.load(open(f"results/{f}"))
GH = {
    "geometry": ("github.com/wollschlager/geometry-of-refusal", "43a8628"),
    "cosmic": ("github.com/wang-research-lab/COSMIC", "2908503"),
    "som": ("github.com/pralab/som-refusal-directions", "d244c7d"),
    "wrdo": ("github.com/brain-lab-research/WRDO", "a67cf00"),
    "more": ("github.com/fjoad/more-than-one-direction", "dd73fc7"),
}
DATA = dict(
    R1=J("results.json"), R2=J("results2.json"), R3=J("results3.json"), R4=J("results4.json"), R5a=J("results5a.json"),
    R5b=J("results5b.json"), R5c=J(os.environ.get("R5C", "results5c.json")), R5d=J(os.environ.get("R5D", "results5d.json")),
    NN=J("nnsight.json"), AU=J("audit.json"), ST=J("stream.json"), FS=J("figs3_stats.json"), T5=all_methods(),
    GH=GH, cm=cm, PageBreak=PageBreak, Paragraph=Paragraph, Spacer=Spacer,
    pct=lambda x: f"{100 * x:.0f}%",
    link=lambda k: f"<link href='https://{GH[k][0]}/tree/{GH[k][1]}' color='#2a78d6'>{GH[k][0]}</link> @ {GH[k][1]}",
)
DATA["E3"] = DATA["R3"]["eval"]
DATA["tkey"] = lambda prefix: next(k for k in DATA["T5"] if k.startswith(prefix))
DATA.update({k: getattr(rs, k) for k in ("W", "S", "para", "mono", "esc", "fn_source", "M", "LazyContents")})


def build(pages):
    ns = dict(DATA, d=rs.Doc(FIG_ORDER, pages), PAGES=pages)
    for f in ["report_takeaways.py"] + CONTENT:
        exec(compile(open(f).read(), f, "exec"), ns)
    doc = rs.GuideDoc(OUT, TITLE, author="yiyuanli939")
    doc.build(ns["d"].story)
    return doc.pages


pages = build({})
final = build(pages)
if final != pages:                                 # the contents page changed length: one more pass
    final = build(final)
print("wrote", OUT, "| figures:", len(FIG_ORDER), "| sections:", len(final))
