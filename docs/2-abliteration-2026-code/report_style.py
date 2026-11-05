"""The look of the PDF (reportlab): fonts, LaTeX math, boxes, highlighted code, figures, tables, contents.

Body text is Noto Serif; headings, captions, tables and boxes are Noto Sans; code is DejaVu Sans Mono.
Math is written as TeX between $...$ anywhere in the text and typeset by matplotlib's mathtext engine in
Computer Modern (no TeX installation is needed). Short macros used in the text:
  \\rh -> hat r,  \\rh22 -> hat r_22,  \\rhl -> hat r_l,  \\bx, \\bx22, \\bxl,  \\be, \\ba, \\bm, \\bmu,  \\KL,  \\T (transpose),
  \\muh / \\mus (harmful / harmless mean),  \\RR (blackboard R)
"""
import ast
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from fontTools.ttLib import TTFont as _FT
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from pygments.lexers import PythonLexer
from pygments.token import Token
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted)

# ------------------------------------------------------------------------------------------------ fonts
NOTO, DV = "/usr/share/fonts/truetype/noto/", "/usr/share/fonts/truetype/dejavu/"
_FAMILIES = {
    "Serif": [NOTO + f"NotoSerif-{s}.ttf" for s in ("Regular", "Bold", "Italic", "BoldItalic")],
    "Sans": [NOTO + f"NotoSans-{s}.ttf" for s in ("Regular", "Bold", "Italic", "BoldItalic")],
    "DVSerif": [DV + f for f in ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf",
                                 "DejaVuSerif-BoldItalic.ttf")],
    "DVSans": [DV + f for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
                                "DejaVuSans-BoldOblique.ttf")],
    "Mono": [DV + f for f in ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf",
                              "DejaVuSansMono-BoldOblique.ttf")],
}
for fam, files in _FAMILIES.items():
    names = [fam, fam + "-Bold", fam + "-Italic", fam + "-BoldItalic"]
    for n, f in zip(names, files):
        pdfmetrics.registerFont(TTFont(n, f))
    registerFontFamily(fam, normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
_CMAP = {fam: set(_FT(files[0]).getBestCmap()) for fam, files in _FAMILIES.items() if fam in ("Serif", "Sans")}

# ------------------------------------------------------------------------------------------------ colours
H = colors.HexColor
INK, INK2, MUTED, RULE = H("#0b0b0b"), H("#52514e"), H("#898781"), H("#e1e0d9")
BLUE, ORANGE, VIOLET, AQUA, VIOLET_D = H("#2a78d6"), H("#eb6834"), H("#6250d6"), H("#1baf7a"), H("#4a3aa7")
CODEBG, BLUEBG, ORANGEBG, VIOLETBG, GREYBG = H("#f6f5f1"), H("#eef4fc"), H("#fdf0ea"), H("#f1effc"), H("#f4f3ef")

# ------------------------------------------------------------------------------------------------ page + styles
PAGE = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 2.3 * cm, 2.4 * cm, 2.2 * cm
W = PAGE[0] - 2 * MARGIN_X
try:                                    # hyphenation for justified text (pip install pyphen); optional
    import pyphen  # noqa: F401
    _hy = dict(hyphenationLang="en_US", embeddedHyphenation=1, uriWasteReduce=0.3)
except ImportError:
    _hy = {}
S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=31, leading=36, textColor=INK, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName="Serif-Italic", fontSize=15.5, leading=21, textColor=INK2),
    "eyebrow": ParagraphStyle("eyebrow", fontName="Sans-Bold", fontSize=8.4, leading=11, textColor=VIOLET_D),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=18, leading=22, textColor=INK),
    "badge": ParagraphStyle("badge", fontName="Sans-Bold", fontSize=15, leading=18, textColor=colors.white,
                            alignment=TA_CENTER),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12, leading=15.5, textColor=INK, spaceBefore=13,
                         spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=9.8, leading=13, textColor=INK2, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.8, leading=14.6, textColor=INK, spaceAfter=6.5,
                           alignment=TA_JUSTIFY, **_hy),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=9.8, leading=14.4, textColor=INK, leftIndent=13,
                             bulletIndent=3, spaceAfter=3, alignment=TA_LEFT, **_hy),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8, leading=11, textColor=INK2, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.2, leading=11.6, textColor=INK2, spaceBefore=4,
                          spaceAfter=13),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=8.8, leading=12.8, textColor=INK),
    "boxhead": ParagraphStyle("boxhead", fontName="Sans-Bold", fontSize=8.4, leading=11, textColor=INK2),
    "boxbullet": ParagraphStyle("boxbullet", fontName="Sans", fontSize=8.8, leading=12.6, textColor=INK,
                                leftIndent=11, bulletIndent=1, spaceAfter=2),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.1, leading=10.9, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.1, leading=10.9, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.0, leading=9.1, textColor=INK),
    "codes": ParagraphStyle("codes", fontName="Mono", fontSize=6.4, leading=8.2, textColor=INK),
    "toc": ParagraphStyle("toc", fontName="Sans-Bold", fontSize=10.2, leading=13, textColor=INK),
    "tocsub": ParagraphStyle("tocsub", fontName="Sans", fontSize=8, leading=10.6, textColor=INK2),
    "tocnum": ParagraphStyle("tocnum", fontName="Sans-Bold", fontSize=10.2, leading=13, textColor=VIOLET_D),
}

# ------------------------------------------------------------------------------------------------ LaTeX math
_MP = MathTextParser("path")
MATHDIR = Path("figs/math"); MATHDIR.mkdir(parents=True, exist_ok=True)
_MACROS = [(r"\\rhl(?![A-Za-z])", lambda m: r"\hat{\mathbf{r}}_\ell"),
           (r"\\rh(\d+)", lambda m: r"\hat{\mathbf{r}}_{%s}" % m.group(1)),
           (r"\\rh(?![A-Za-z])", lambda m: r"\hat{\mathbf{r}}"),
           (r"\\bxl(?![A-Za-z])", lambda m: r"\mathbf{x}_\ell"),
           (r"\\bx(\d+)", lambda m: r"\mathbf{x}_{%s}" % m.group(1)),
           (r"\\bx(?![A-Za-z])", lambda m: r"\mathbf{x}"),
           (r"\\bmu(?![A-Za-z])", lambda m: r"\boldsymbol{\mu}"),
           (r"\\be(?![A-Za-z])", lambda m: r"\mathbf{e}"),
           (r"\\ba(?![A-Za-z])", lambda m: r"\mathbf{a}"),
           (r"\\bm(?![A-Za-z])", lambda m: r"\mathbf{m}"),
           (r"\\KL(?![A-Za-z])", lambda m: r"D_{\mathrm{KL}}"),
           (r"\\muh(?![A-Za-z])", lambda m: r"\boldsymbol{\mu}^{\mathrm{harm}}"),
           (r"\\mus(?![A-Za-z])", lambda m: r"\boldsymbol{\mu}^{\mathrm{safe}}"),
           (r"\\RR(?![A-Za-z])", lambda m: r"\mathbb{R}"),
           (r"\\T(?![A-Za-z])", lambda m: r"^{\top}")]


def tex(s):
    """Expand the guide's macros into plain TeX."""
    for pat, rep in _MACROS:
        s = re.sub(pat, rep, s)
    return s


def math_png(src, size):
    """Render TeX with mathtext (Computer Modern). Returns (path, width, height, depth) in points."""
    s = f"${tex(src)}$"
    prop = FontProperties(size=size, math_fontfamily="cm")
    w, h, d, *_ = _MP.parse(s, dpi=72, prop=prop)
    f = MATHDIR / (hashlib.md5(f"{s}|{size}".encode()).hexdigest()[:16] + ".png")
    if not f.exists():
        fig = Figure(figsize=(w / 72, h / 72)); FigureCanvasAgg(fig)
        fig.text(0, d / h, s, fontproperties=prop, color="#0b0b0b")
        fig.savefig(f, dpi=900, transparent=True)
    return str(f), w, h, d


def M(src, size=11.4):
    """Inline math as an <img> tag on the text baseline."""
    f, w, h, d = math_png(src, size)
    return f'<img src="{f}" width="{w:.2f}" height="{h:.2f}" valign="{-d:.2f}"/>'


def _fallback(text, base):
    """Characters missing from Noto get DejaVu (same style), outside tags and outside Mono runs."""
    miss_ok = _CMAP[base]; face = "DVSerif" if base == "Serif" else "DVSans"
    out, i, stack = [], 0, []
    while i < len(text):
        if text[i] == "<":
            j = text.index(">", i); tag = text[i:j + 1]
            if tag.startswith("<font"):
                stack.append("Mono" in tag)
            elif tag.startswith("</font"):
                stack and stack.pop()
            out.append(tag); i = j + 1; continue
        if text[i] == "&":
            j = text.index(";", i); out.append(text[i:j + 1]); i = j + 1; continue
        ch = text[i]
        if ch == "'" and not any(stack):
            ch = "\u2019"                                         # typographic apostrophe
        out.append(f'<font face="{face}">{ch}</font>' if (ord(ch) not in miss_ok and ch not in "\n " and not any(stack))
                   else ch)
        i += 1
    return "".join(out)


def rich(text, st):
    """TeX between $...$ becomes math images; missing glyphs fall back to DejaVu."""
    size = st.fontSize * 1.17
    text = re.sub(r"\$([^$]+)\$", lambda m: M(m.group(1), size), text)
    base = "Serif" if st.fontName.startswith("Serif") else ("Sans" if not st.fontName.startswith("Mono") else None)
    return _fallback(text, base) if base else text


def para(text, st="body", **kw):
    style = S[st] if isinstance(st, str) else st
    if re.search(r"name=.Mono", text):                # inline code: no justification, never hyphenate code
        style = style.clone(style.name + "_code", alignment=TA_LEFT, hyphenationLang=None)
    return Paragraph(rich(text, style), style, **kw)


class FloatGroup(Flowable):
    """A figure plus the one or two paragraphs after it. If the figure does not fit on the page, the paragraphs
    go first and the figure moves to the next page (a LaTeX-style float, limited to two paragraphs)."""
    def __init__(self, fig):
        super().__init__(); self.fig, self.followers = fig, []

    def wrap(self, aw, ah):
        return aw, 0x7fffffff                       # always ask to be split; split() decides the order

    def split(self, aw, ah):
        self.fig.wrapOn(self.canv, aw, ah)
        if self.fig._H <= ah:                           # KeepTogether records its real content height in _H
            return list(self.fig._content) + self.followers
        if self.followers:                              # the doc places S[0] without splitting: it must fit
            f0 = self.followers[0]
            _, h0 = f0.wrapOn(self.canv, aw, ah)
            if h0 + f0.getSpaceBefore() <= ah:
                return self.followers + [self.fig]
        return []

    def draw(self):
        pass


# ------------------------------------------------------------------------------------------------ code
_CODE_COL = [(Token.Comment, "#7d7b74", "Mono-Italic"), (Token.Literal.String, "#2f7d4f", None),
             (Token.Literal.Number, "#b0461f", None), (Token.Keyword, "#4a3aa7", "Mono-Bold"),
             (Token.Name.Builtin, "#4a3aa7", None), (Token.Name.Function, "#1c5cab", "Mono-Bold"),
             (Token.Name.Class, "#1c5cab", "Mono-Bold"), (Token.Name.Decorator, "#8a5d00", None),
             (Token.Operator.Word, "#4a3aa7", "Mono-Bold")]


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "\u00a0")


def _span(tt, v):
    for t, col, font in _CODE_COL:
        if tt in t:
            return f'<font color="{col}"{f" name={chr(34)}{font}{chr(34)}" if font else ""}>{_esc(v)}</font>'
    return _esc(v)


def highlight(src, size, width):
    """Python source -> XPreformatted markup, soft-wrapped to the box width ('↪' marks continuations)."""
    lines = [[]]
    for tt, v in PythonLexer(stripnl=False, ensurenl=False).get_tokens(src):
        for k, part in enumerate(v.split("\n")):
            if k:
                lines.append([])
            if part:
                lines[-1].append((tt, part))
    n = int(width / (0.602 * size)) - 1
    out = []
    for segs in lines:
        text = "".join(v for _, v in segs)
        indent = len(text) - len(text.lstrip())
        start, first = 0, True
        while True:
            avail = n if first else n - indent - 2
            if not first:
                while start < len(text) and text[start] == " ":
                    start += 1
            rest = text[start:]
            if len(rest) <= avail:
                end = len(text)
            else:
                cut = rest.rfind(" ", 0, avail); cut = cut if cut > avail * 0.6 else avail
                end = start + cut
            piece, pos = [], 0
            for tt, v in segs:
                a, b = max(pos, start), min(pos + len(v), end)
                if a < b:
                    piece.append(_span(tt, v[a - pos:b - pos]))
                pos += len(v)
            prefix = "" if first else _esc(" " * indent) + '<font color="#a3a198">↪\u00a0</font>'
            out.append((prefix + "".join(piece)) or '<font color="#f6f5f1">.</font>')   # invisible: keeps blank lines
            if end >= len(text):
                break
            start, first = end, False
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ story
class _Story(list):
    """Headings stick to the next flowable, so no heading is left alone at the bottom of a page."""
    def __init__(self):
        super().__init__(); self.pending, self.float = [], None

    def append(self, f):
        if self.float is not None and getattr(f, "_follow", False) and not self.pending and len(self.float.followers) < 2:
            self.float.followers.append(f); return
        self.float = None
        if getattr(f, "_heading", False):
            self.pending.append(f); return
        if self.pending and getattr(f, "_long", False):  # a long splittable block: headings go first, no keep
            super().extend(self.pending); self.pending = []
        if self.pending:
            inner = list(f._content) if isinstance(f, KeepTogether) else [f]
            f = KeepTogether(self.pending + inner); self.pending = []
        super().append(f)

    def extend(self, fs):
        for f in fs:
            self.append(f)


MOVES = ("READ", "CHOOSE", "SUBTRACT", "ADD", "BAKE")


class Doc:
    def __init__(self, fig_order, pages=None):
        self.story, self.fig_order, self.pages = _Story(), fig_order, pages or {}
        self.n_sections, self.eq_n, self.toc = 0, 0, []

    # -------- references
    def F(self, name):
        return self.fig_order.index(name) + 1

    def add(self, *f):
        self.story.extend(f)

    def P(self, text, st="body"):
        p = para(text, st); p._follow = st == "body"
        self.story.append(p)

    # -------- headings
    def section(self, num, title, blurb="", new_page=True, in_toc=True):
        """Numbered section on a new page; recorded for the contents page and the PDF outline."""
        if new_page and len(self.story):
            self.story.append(PageBreak())
        key = f"sec{len(self.toc)}"
        if in_toc:
            self.toc.append((num, title, key, blurb))
        cells = [[para(num, "badge"), para(title, "h1")]] if num else [[para(title, "h1")]]
        t = Table(cells, colWidths=[30, W - 30] if num else [W])
        st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 3),
              ("LINEBELOW", (0, 0), (-1, -1), 0.8, RULE)]
        if num:
            st += [("BACKGROUND", (0, 0), (0, 0), VIOLET), ("LEFTPADDING", (1, 0), (1, 0), 11),
                   ("BOTTOMPADDING", (0, 0), (0, 0), 5)]
        t.setStyle(TableStyle(st))
        if in_toc:
            t._toc = (0, (f"{num}. " if num and num[0].isdigit() else (f"{num}  " if num else "")) + title, key)
        t._heading = True
        self.story.append(t)
        self.story.pending.append(Spacer(1, 8))

    def H2(self, t, need=4.5, key=None):
        self.story.append(CondPageBreak(need * cm))
        p = para(t, "h2"); p._heading = True
        self.n_h2 = getattr(self, "n_h2", 0) + 1
        p._toc = (1, re.sub(r"<[^>]+>", "", t), key or f"h2_{self.n_h2}")
        self.story.append(p)

    def H3(self, t):
        self.story.append(CondPageBreak(3 * cm))
        p = para(t, "h3"); p._heading = True
        self.story.append(p)

    # -------- text blocks
    def bullets(self, items, st="bullet"):
        for it in items:
            self.story.append(para(it, st, bulletText="•"))
        self.story.float = None

    def eq(self, src, number=True, size=12.5):
        """Display equation, centred, with an equation number on the right."""
        f, w, h, d = math_png(src, size)
        if number:
            self.eq_n += 1
        t = Table([["", Image(f, width=w, height=h), para(f"({self.eq_n})" if number else "", "small")]],
                  colWidths=[40, W - 80, 40])
        t.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.story.append(t)
        return self.eq_n

    def _boxed(self, rows, bg, edge, pad=9, edge_w=3):
        t = Table(rows, colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), edge_w, edge),
                               ("LEFTPADDING", (0, 0), (-1, -1), pad + 1), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                               ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
        self.story.append(KeepTogether([t, Spacer(1, 9)]))

    def box(self, html, kind="info", title=None):
        bg, edge = {"info": (BLUEBG, BLUE), "warn": (ORANGEBG, ORANGE), "grey": (GREYBG, MUTED)}[kind]
        rows = ([[para(title, "boxhead")]] if title else []) + [[para(html, "box")]]
        self._boxed(rows, bg, edge)

    def key(self, moves, html):
        """The opening box of every section: which moves on the residual stream it uses, in one sentence."""
        chips = "&nbsp;".join(f'<font backColor="#ffffff" color="#4a3aa7"><b>&nbsp;{m}&nbsp;</b></font>'
                              for m in moves)
        html = html[0].upper() + html[1:]
        self._boxed([[para(f"<b>On the residual stream</b>&nbsp;&nbsp;&nbsp;{chips}", "boxhead")],
                     [para(html, "box")]], VIOLETBG, VIOLET)

    def review(self, points, check=None, answer=None):
        """End-of-section review: what to remember, and one question to test it."""
        rows = [[para("REVIEW", "boxhead")]] + [[para(p, "boxbullet", bulletText="✓")] for p in points]
        if check:
            rows.append([para(f"<b>Check yourself.</b> {check}", "box")])
            rows.append([para(f"<i>Answer:</i> {answer}", S["box"].clone("ans", textColor=INK2))])
        t = Table(rows, colWidths=[W])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, RULE), ("BACKGROUND", (0, 0), (-1, 0), GREYBG),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
                               ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                               ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                               ("TOPPADDING", (0, 1), (-1, 1), 7), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
        self.story.append(Spacer(1, 6)); self.story.append(KeepTogether([t, Spacer(1, 6)]))

    def contrast(self, title_a, a, title_b, b, label="Contrast"):
        """Two columns, one question, two answers: grey = the naive or 2024 answer, violet = what we measured."""
        t = Table([[para(f"<b>{label}</b>", "boxhead"), ""],
                   [para(f"<b>{title_a}</b>", "cell"), para(f"<b>{title_b}</b>", "cell")],
                   [para(a, "cell"), para(b, "cell")]], colWidths=[W / 2, W / 2])
        t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 1), (0, -1), GREYBG),
                               ("BACKGROUND", (1, 1), (1, -1), VIOLETBG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("LEFTPADDING", (0, 0), (0, 0), 0), ("TOPPADDING", (0, 0), (-1, 0), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, 0), 4), ("TOPPADDING", (0, 1), (-1, 1), 6),
                               ("BOTTOMPADDING", (0, -1), (-1, -1), 8), ("LINEAFTER", (0, 1), (0, -1), 2, colors.white),
                               ("LINEABOVE", (0, 1), (0, 1), 2, MUTED), ("LINEABOVE", (1, 1), (1, 1), 2, VIOLET)]))
        self.story.append(KeepTogether([t, Spacer(1, 10)]))

    # -------- code
    def code(self, src, title=None, small=False, rows_of=6):
        """Highlighted code in a box that can break across pages between groups of `rows_of` lines."""
        st = S["codes" if small else "code"]
        lines = highlight(src.strip("\n"), st.fontSize, W - 16).split("\n")
        head = [[para(f'<font name="Mono" size="7.4">{title}</font>', "small")]] if title else []
        o = len(head)
        rows = head + [[XPreformatted("\n".join(lines[i:i + rows_of]), st)] for i in range(0, len(lines), rows_of)]
        t = Table(rows, colWidths=[W])
        style = [("BACKGROUND", (0, o), (-1, -1), CODEBG), ("LEFTPADDING", (0, o), (-1, -1), 9),
                 ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 0),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, o), (-1, o), 6),
                 ("BOTTOMPADDING", (0, -1), (-1, -1), 6), ("LINEBEFORE", (0, o), (0, -1), 2, VIOLET),
                 ("NOSPLIT", (0, 0), (0, min(o + 1, len(rows) - 1)))]
        if o:
            style += [("LEFTPADDING", (0, 0), (0, 0), 0), ("BOTTOMPADDING", (0, 0), (0, 0), 3)]
        t.setStyle(TableStyle(style))
        if len(lines) <= 40:
            self.story.append(KeepTogether([t, Spacer(1, 8)]))
        else:                                           # long listing: may start mid-page and break between rows
            if not self.story.pending:
                self.story.append(CondPageBreak(4 * cm))
            t._long = True
            self.story.append(t); self.story.append(Spacer(1, 8))

    def code_long(self, src, title, small=True):
        self.code(src, title, small=small)

    def side_by_side(self, left, right, lt, rt):
        cw = W / 2 - 3
        t = Table([[para(f"<b>{lt}</b>", "small"), para(f"<b>{rt}</b>", "small")],
                   [XPreformatted(highlight(left.strip("\n"), 6.4, cw - 12), S["codes"]),
                    XPreformatted(highlight(right.strip("\n"), 6.4, cw - 12), S["codes"])]], colWidths=[cw + 3, cw + 3])
        t.setStyle(TableStyle([("BACKGROUND", (0, 1), (-1, 1), CODEBG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBEFORE", (0, 1), (0, 1), 2, MUTED), ("LINEBEFORE", (1, 1), (1, 1), 2, VIOLET),
                               ("LINEAFTER", (0, 1), (0, 1), 3, colors.white),
                               ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 1), (-1, 1), 6),
                               ("BOTTOMPADDING", (0, 1), (-1, 1), 6)]))
        self.story.append(KeepTogether([t, Spacer(1, 9)]))

    # -------- figures and tables
    def figure(self, name, caption, width=W):
        from PIL import Image as PI
        w, h = PI.open(f"figs/{name}.png").size
        width = min(width, W)
        img = Image(f"figs/{name}.png", width=width, height=width * h / w)
        fig = KeepTogether([Spacer(1, 3), img, para(f"<b>Figure {self.F(name)}.</b> {caption}", "cap")])
        if self.story.pending:                          # a heading must stay with its figure: no float
            self.story.append(fig); return
        g = FloatGroup(fig); self.story.append(g); self.story.float = g

    _NUM = re.compile(r"^[\s≈~<>+\-−]*[\d.,]+\s*(%|s|min|GB|×)?$")

    def table(self, rows, widths, header=True, hl_rows=()):
        plain = lambda c: re.sub(r"<[^>]+>", "", c) if isinstance(c, str) else ""
        ncol = len(rows[0])
        numcol = [all(self._NUM.match(plain(r[j]).strip() or "0") for r in rows[1 if header else 0:]) for j in range(ncol)]
        data = []
        for i, r in enumerate(rows):
            row = []
            for j, c in enumerate(r):
                st = S["cellb" if (header and i == 0) else "cell"]
                if numcol[j]:
                    st = st.clone(st.name + "r", alignment=2)
                row.append(para(c, st) if isinstance(c, str) else c)
            data.append(row)
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2),
              ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.6),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 3.8), ("LEFTPADDING", (0, 0), (-1, -1), 4),
              ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
        for r in hl_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), VIOLETBG))
        t.setStyle(TableStyle(st))
        self.story.append(KeepTogether([t, Spacer(1, 10)]))


# ------------------------------------------------------------------------------------------------ document
class GuideDoc(BaseDocTemplate):
    """A4, running header with the current section, page numbers, PDF outline from the section headings."""
    def __init__(self, filename, title, **kw):
        super().__init__(filename, pagesize=PAGE, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
                         bottomMargin=MARGIN_BOTTOM, title=title, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body", leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("page", [frame], onPage=self._footer, onPageEnd=self._header)])
        self.pages, self.cur, self.starts, self.short_title = {}, "", set(), title.split(":")[0]

    def afterFlowable(self, f):
        t = getattr(f, "_toc", None)
        if t:
            level, text, key = t
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=True)
            self.pages[key] = self.page
            if level == 0:
                self.pages[key] = self.page; self.cur = text; self.starts.add(self.page)

    def _footer(self, c, doc):
        c.saveState(); c.setFont("Sans", 7.8); c.setFillColor(MUTED)
        c.drawCentredString(PAGE[0] / 2, 1.25 * cm, str(doc.page)); c.restoreState()

    def _header(self, c, doc):
        if doc.page == 1 or doc.page in self.starts or not self.cur:
            return
        c.saveState(); c.setFont("Sans", 7.6); c.setFillColor(MUTED)
        y = PAGE[1] - 1.5 * cm
        c.drawString(MARGIN_X, y, self.short_title)
        c.drawRightString(PAGE[0] - MARGIN_X, y, self.cur)
        c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(MARGIN_X, y - 5, PAGE[0] - MARGIN_X, y - 5)
        c.restoreState()


class LazyContents(Flowable):
    """The contents table, built when it is laid out (by then every section has been recorded)."""
    def __init__(self, doc_obj, pages):
        super().__init__(); self.d, self.pages, self._t = doc_obj, pages, None

    def _table(self):
        if self._t is None:
            self._t = contents_table(self.d.toc, self.pages)
        return self._t

    def wrap(self, aw, ah):
        return self._table().wrap(aw, ah)

    def split(self, aw, ah):
        return self._table().split(aw, ah)

    def draw(self):
        self._table().drawOn(self.canv, 0, 0)


def contents_table(toc, pages):
    rows = []
    for num, title, key, blurb in toc:
        rows.append([para(num, "tocnum"), [para(title, "toc")] + ([para(blurb, "tocsub")] if blurb else []),
                     para(str(pages.get(key, "")), S["toc"].clone("tocp", alignment=2, fontName="Sans"))])
    t = Table(rows, colWidths=[26, W - 66, 40])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


# ------------------------------------------------------------------------------------------------ helpers
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
    return f"<font name='Mono' size='8.4'>{s}</font>"
