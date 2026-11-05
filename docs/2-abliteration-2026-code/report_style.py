"""Shared PDF styling and flowable helpers for report.py (reportlab)."""
import ast
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, Preformatted, Spacer, Table, TableStyle)

D = "/usr/share/fonts/truetype/dejavu/"
for name, f in [("Sans", "DejaVuSans"), ("Sans-Bold", "DejaVuSans-Bold"), ("Sans-Oblique", "DejaVuSans-Oblique"),
                ("Sans-BoldOblique", "DejaVuSans-BoldOblique"), ("Mono", "DejaVuSansMono"), ("Mono-Bold", "DejaVuSansMono-Bold")]:
    pdfmetrics.registerFont(TTFont(name, D + f + ".ttf"))
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique", boldItalic="Sans-BoldOblique")
registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

H = colors.HexColor
INK, INK2, MUTED, RULE = H("#0b0b0b"), H("#52514e"), H("#898781"), H("#e1e0d9")
BLUE, ORANGE, VIOLET, CODEBG = H("#2a78d6"), H("#eb6834"), H("#6250d6"), H("#f4f3ef")
BLUEBG, ORANGEBG, VIOLETBG, GREYBG = H("#eef4fc"), H("#fdf0ea"), H("#efedfb"), H("#f4f3ef")

W = A4[0] - 4.0 * cm
S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=25, leading=30, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=12, leading=17, textColor=INK2, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=20, textColor=INK, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=9.6, leading=13, textColor=INK2, spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.3, leading=13.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.1, leading=11, textColor=INK2, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8, leading=11, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.0, leading=9.0, textColor=INK),
    "codes": ParagraphStyle("codes", fontName="Mono", fontSize=6.3, leading=8.0, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.1, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.1, leading=10.6, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9.3, leading=13.2, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "toc": ParagraphStyle("toc", fontName="Sans", fontSize=10, leading=16, textColor=INK),
    "key": ParagraphStyle("key", fontName="Sans", fontSize=9.8, leading=14, textColor=INK),
}


def _wrap(src, size, width):
    """Soft-wrap code lines wider than the box; continuation lines start with '↪'. The source is not changed."""
    n = int(width / (0.6 * size)) - 1                  # DejaVu Sans Mono advance = 0.6 em
    out = []
    for line in src.splitlines():
        while len(line) > n:
            cut = line.rfind(" ", 0, n)
            cut = cut if cut > n * 0.6 else n
            out.append(line[:cut])
            indent = len(line) - len(line.lstrip())
            line = " " * indent + "↪ " + line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


class _Story(list):
    """A story where headings stick to the next flowable (no heading alone at the bottom of a page)."""
    def __init__(self):
        super().__init__(); self.pending = []

    def append(self, f):
        if getattr(f, "_heading", False):
            self.pending.append(f); return
        if self.pending:
            inner = list(f._content) if isinstance(f, KeepTogether) else [f]   # flatten: nested KeepTogether misbehaves
            f = KeepTogether(self.pending + inner); self.pending = []
        super().append(f)

    def extend(self, fs):
        for f in fs: self.append(f)


class Doc:
    def __init__(self):
        self.story = _Story()

    def add(self, *f):
        self.story.extend(f)

    def P(self, text, st="body"):
        self.story.append(Paragraph(text, S[st]))

    def _h(self, t, st, need):
        self.story.append(CondPageBreak(need * cm))
        para = Paragraph(t, S[st]); para._heading = True
        self.story.append(para)

    def H1(self, t, need=7):
        self._h(t, "h1", need)

    def H2(self, t, need=4.5):
        self._h(t, "h2", need)

    def H3(self, t):
        self._h(t, "h3", 3)

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(it, S["bullet"], bulletText="•"))

    def box(self, html, kind="info", style="body"):
        bg, edge = {"info": (BLUEBG, BLUE), "warn": (ORANGEBG, ORANGE), "key": (VIOLETBG, VIOLET),
                    "grey": (GREYBG, MUTED)}[kind]
        t = Table([[Paragraph(html, S[style])]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        self.story.append(KeepTogether([t, Spacer(1, 8)]))

    def key(self, html):
        """One-sentence 'move on the residual stream' that opens a section."""
        self.box("<b>On the residual stream:</b> " + html, "key", "key")

    def code(self, src, title=None, small=False, edge=BLUE):
        st = S["codes" if small else "code"]
        t = Table([[Preformatted(_wrap(src.strip("\n"), st.fontSize, W - 14), st)]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODEBG), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBEFORE", (0, 0), (0, -1), 2, edge)]))
        parts = [Paragraph(title, S["small"])] if title else []
        self.story.append(KeepTogether(parts + [t, Spacer(1, 7)]))

    def code_long(self, src, title, chunk=72, small=True):
        lines = src.strip("\n").splitlines()
        for i in range(0, len(lines), chunk):
            self.code("\n".join(lines[i:i + chunk]), title if i == 0 else title + " (continued)", small=small)

    def side_by_side(self, left, right, lt, rt):
        hdr = [Paragraph(f"<b>{lt}</b>", S["small"]), Paragraph(f"<b>{rt}</b>", S["small"])]
        t = Table([hdr, [Preformatted(_wrap(left.strip("\n"), 6.3, W / 2 - 12), S["codes"]),
                         Preformatted(_wrap(right.strip("\n"), 6.3, W / 2 - 12), S["codes"])]],
                  colWidths=[W / 2 - 2, W / 2 - 2])
        t.setStyle(TableStyle([("BACKGROUND", (0, 1), (-1, 1), CODEBG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBEFORE", (0, 1), (0, 1), 2, BLUE), ("LINEBEFORE", (1, 1), (1, 1), 2, VIOLET),
                               ("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 1), (-1, 1), 5),
                               ("BOTTOMPADDING", (0, 1), (-1, 1), 5)]))
        self.story.append(KeepTogether([t, Spacer(1, 8)]))

    def figure(self, name, caption, width=W):
        from PIL import Image as PI
        w, h = PI.open(f"figs/{name}.png").size
        self.story.append(KeepTogether([Image(f"figs/{name}.png", width=width, height=width * h / w),
                                        Paragraph(caption, S["cap"])]))

    def table(self, rows, widths, header=True, hl_rows=()):
        data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else "cell"])
                 for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
              ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
        for r in hl_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), VIOLETBG))
        t.setStyle(TableStyle(st))
        self.story.append(KeepTogether([t, Spacer(1, 8)]))

    def contrast(self, title_a, a, title_b, b, label="Contrast"):
        """Two columns, same question, two answers."""
        hdr = [Paragraph(f"<b>{title_a}</b>", S["cell"]), Paragraph(f"<b>{title_b}</b>", S["cell"])]
        t = Table([[Paragraph(f"<b>{label}</b>", S["small"]), ""], hdr,
                   [Paragraph(a, S["cell"]), Paragraph(b, S["cell"])]], colWidths=[W / 2, W / 2])
        t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 1), (0, -1), GREYBG),
                               ("BACKGROUND", (1, 1), (1, -1), VIOLETBG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 1), (-1, -1), 5), ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                               ("LINEAFTER", (0, 1), (0, -1), 1.5, colors.white)]))
        self.story.append(KeepTogether([t, Spacer(1, 8)]))


def fn_source(path, name):
    src = Path(path).read_text()
    for node in ast.parse(src).body:
        if getattr(node, "name", None) == name:
            start = node.lineno - 1 - len(getattr(node, "decorator_list", []))
            return "\n".join(src.splitlines()[start:node.end_lineno])
    raise KeyError(name)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono(s):
    return f"<font name='Mono' size='8.3'>{s}</font>"
