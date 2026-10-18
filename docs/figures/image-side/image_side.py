"""The image side of the model as a set of explanatory SVG panels: from the photo to the point where its patch tokens
join TabICL's row transformer. One file per panel, an overview, preview.html and (with Chrome) preview.png.

    python3 docs/figures/image-side/image_side.py [out_dir]

Every number is the one in table_image_icl/model.py and inputs.py (ModelConfig: 224 px photos, 14 px patches, a 4-layer
encoder, 2 label reads, top-32 of up to 256 sources, width 128, 4 heads) and training.py (lr 1e-5 / 1e-4).

Style, shared with the component library: 2 px rounded strokes, pastel fill + darker stroke per module type, Helvetica
text as real <text> elements, transparent background, a <title> per file, no filters, images or external fonts. Change
the palette or the fonts here and rerun; every SVG stays hand-editable.
"""
import html
import math
import os
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------------- style
PAD = 8
W, H = 884, 404                       # drawing area; the file is 900 x 420 with the padding
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, WHITE = "#1F2937", "#6B7280", "#FFFFFF"
P = {  # fill, stroke
    "blue": ("#DBEAFE", "#2563EB"),       # encoders, data, input
    "orange": ("#FFEDD5", "#EA580C"),     # attention, trainable
    "green": ("#DCFCE7", "#16A34A"),      # MLP
    "grey": ("#E5E7EB", "#6B7280"),       # frozen, inactive
    "purple": ("#EDE9FE", "#7C3AED"),     # embeddings, tokens
    "yellow": ("#FEF3C7", "#D97706"),     # labels
}
FLAME_FILL, FLAME_CORE = "#F97316", "#FDE68A"
FILES = []


# ----------------------------------------------------------------------------------------------- primitives
def L(s):
    return "    " + s + "\n"


def esc(s):
    return html.escape(str(s), quote=False)


def rect(x, y, w, h, kind, rx=8, dashed=False, sw=2, fill=None, stroke=None, extra=""):
    f, s = P[kind] if kind in P else (fill, stroke)
    f, s = fill or f, stroke or s
    d = ' stroke-dasharray="6 4"' if dashed else ""
    return L(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}{extra}/>')


def text(x, y, s, size=13, color=INK, anchor="middle", weight=None, italic=False, font=None):
    """Text centred vertically on y."""
    w = f' font-weight="{weight}"' if weight else ""
    i = ' font-style="italic"' if italic else ""
    f = f' font-family="{font}"' if font else ""
    return L(f'<text x="{x:.1f}" y="{y + size * 0.36:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}{i}{f}>{esc(s)}</text>')


def lines(x, y, rows, size=11, color=INK, anchor="middle", lead=None, weight=None):
    lead = lead or size + 3
    return "".join(text(x, y + i * lead, r, size, color, anchor, weight) for i, r in enumerate(rows))


def line(x1, y1, x2, y2, color=INK, sw=2, dashed=False):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    return L(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')


def head(x, y, angle, color=INK, size=9):
    """Filled arrowhead with its tip at (x, y), pointing along `angle` degrees (0 = right, 90 = down)."""
    a = math.radians(angle)
    bx, by = x - size * math.cos(a), y - size * math.sin(a)
    px, py = -math.sin(a) * size * 0.5, math.cos(a) * size * 0.5
    return L(f'<polygon points="{x:.1f},{y:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{color}" stroke="none"/>')


def arrow(x1, y1, x2, y2, color=INK, sw=2, dashed=False, size=9):
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    a = math.radians(ang)
    return line(x1, y1, x2 - 0.75 * size * math.cos(a), y2 - 0.75 * size * math.sin(a), color, sw, dashed) + head(x2, y2, ang, color, size)


def arc(x1, y1, x2, y2, bend=-30, color=P["orange"][1], sw=1.5, dashed=False, arrowhead=True):
    """Quadratic arc from (x1, y1) to (x2, y2); bend = control-point offset normal to the chord (negative = up/left)."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1
    cx, cy = mx + bend * (-dy / n), my + bend * (dx / n)
    d = ' stroke-dasharray="5 4"' if dashed else ""
    s = L(f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{sw}"{d}/>')
    if arrowhead:
        s += head(x2, y2, math.degrees(math.atan2(y2 - cy, x2 - cx)), color, 7)
    return s


def path(d, color=INK, sw=2, dashed=False, fill="none"):
    dd = ' stroke-dasharray="6 4"' if dashed else ""
    return L(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{dd}/>')


def block(x, y, w, h, kind, label, sub=None, size=12, sub_size=10, dashed=False):
    """A module box with a bold label and an optional muted sub-label; label and sub may be lists of lines."""
    s = rect(x, y, w, h, kind, dashed=dashed)
    labels = label if isinstance(label, list) else [label]
    subs = [] if sub is None else (sub if isinstance(sub, list) else [sub])
    total = len(labels) * (size + 2) + len(subs) * (sub_size + 2)
    cy = y + h / 2 - total / 2 + (size + 2) / 2
    s += lines(x + w / 2, cy, labels, size, P[kind][1], lead=size + 2, weight="bold")
    if subs:
        s += lines(x + w / 2, cy + len(labels) * (size + 2) - (size + 2) / 2 + (sub_size + 2) / 2, subs, sub_size, MUTED, lead=sub_size + 2)
    return s


def snowflake(cx, cy, r, color=P["grey"][1], sw=2):
    d = ""
    for k in range(6):
        a = math.radians(k * 60)
        d += f"M{cx:.1f},{cy:.1f} L{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f} "
        mx, my = cx + 0.6 * r * math.cos(a), cy + 0.6 * r * math.sin(a)
        for sgn in (-1, 1):
            b = a + sgn * math.radians(50)
            d += f"M{mx:.1f},{my:.1f} L{mx + 0.35 * r * math.cos(b):.1f},{my + 0.35 * r * math.sin(b):.1f} "
    return L(f'<path d="{d.strip()}" fill="none" stroke="{color}" stroke-width="{sw}"/>')


def flame(cx, cy, s):
    """Flame glyph centred on (cx, cy); s ~ half-height."""
    outer = (f'<path transform="translate({cx:.1f} {cy:.1f}) scale({s / 11:.3f})" d="M0,-11 C5,-6 8,-1 8,4 C8,9 4.5,12 0,12 C-4.5,12 -8,9 -8,4 '
             f'C-8,0 -5.5,-2 -4,-5 C-3.5,-2 -2,-0.5 -0.5,0.5 C1,-3 0.5,-7 0,-11 Z" fill="{FLAME_FILL}" stroke="{P["orange"][1]}" stroke-width="{11 / s * 2:.2f}"/>')
    inner = (f'<path transform="translate({cx:.1f} {cy:.1f}) scale({s / 11:.3f})" d="M0.5,2 C3,4.5 4,6.5 4,8.5 C4,10.5 2,12 0,12 C-2,12 -4,10.5 -4,8.5 '
             f'C-4,6.5 -2.5,5 -1,2.5 C-0.8,4 0,4.5 0.5,2 Z" fill="{FLAME_CORE}" stroke="none"/>')
    return L(outer) + L(inner)


def badge(cx, cy, kind):
    """A 20 px circle with a snowflake (kind "frozen") or a flame (kind "trained") to sit on a block's corner."""
    s = L(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{WHITE}" stroke="{P["grey"][1] if kind == "frozen" else P["orange"][1]}" stroke-width="1.5"/>')
    return s + (snowflake(cx, cy, 6.5, sw=1.5) if kind == "frozen" else flame(cx, cy + 0.5, 6))


def token(x, y, size=24, kind="purple", label=None, rx=5, text_size=10, hollow=False):
    s = rect(x, y, size, size, kind, rx=rx, sw=1.5, fill=WHITE if hollow else None)
    if label is not None:
        s += text(x + size / 2, y + size / 2, label, text_size, P[kind][1], weight="bold")
    return s


def chip(x, y, w, h, kind, label, size=10):
    s = rect(x, y, w, h, kind, rx=h / 2, sw=1.5)
    return s + text(x + w / 2, y + h / 2, label, size, P[kind][1], weight="bold")


def photo_icon(x, y, w, h, rx=4, sw=2, stroke=None, uid=""):
    """Landscape picture: sky, hill, sun, clipped to a rounded rectangle."""
    stroke = stroke or P["blue"][1]
    cid = f"c{uid}{x:.0f}x{y:.0f}x{w:.0f}x{h:.0f}".replace("-", "n")
    s = L(f'<clipPath id="{cid}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/></clipPath>')
    s += L(f'<g clip-path="url(#{cid})">')
    s += L(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#BFDBFE"/>')
    s += L(f'<circle cx="{x + w * 0.72:.1f}" cy="{y + h * 0.3:.1f}" r="{min(w, h) * 0.12:.1f}" fill="{FLAME_CORE}" stroke="{P["yellow"][1]}" stroke-width="1.5"/>')
    s += L(f'<path d="M{x:.1f},{y + h * 0.78:.1f} L{x + w * 0.3:.1f},{y + h * 0.42:.1f} L{x + w * 0.52:.1f},{y + h * 0.68:.1f} L{x + w * 0.66:.1f},{y + h * 0.55:.1f} '
           f'L{x + w:.1f},{y + h * 0.85:.1f} L{x + w:.1f},{y + h:.1f} L{x:.1f},{y + h:.1f} Z" fill="#86EFAC" stroke="{P["green"][1]}" stroke-width="1.5"/>')
    s += L("</g>")
    return s + L(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')


def patch_grid(x, y, size, n=4, stroke=None, sw=1.5, grid_sw=1.5, uid=""):
    """A photo cut into n x n patches (white grid lines over the picture)."""
    s = photo_icon(x, y, size, size, rx=4, sw=sw, stroke=stroke, uid=uid)
    for k in range(1, n):
        s += line(x + k * size / n, y, x + k * size / n, y + size, WHITE, grid_sw) + line(x, y + k * size / n, x + size, y + k * size / n, WHITE, grid_sw)
    return s


def cell_center(x, y, size, n, col, row):
    return x + (col + 0.5) * size / n, y + (row + 0.5) * size / n


def highlight(x, y, size, n, col, row, kind="orange", opacity=0.75):
    c = size / n
    return L(f'<rect x="{x + col * c:.1f}" y="{y + row * c:.1f}" width="{c:.1f}" height="{c:.1f}" fill="{P[kind][0]}" fill-opacity="{opacity}" stroke="{P[kind][1]}" stroke-width="1.5"/>')


def hbrace(x1, x2, y, color=MUTED, depth=6):
    m = (x1 + x2) / 2
    return path(f"M{x1:.1f},{y:.1f} Q{x1:.1f},{y + depth:.1f} {x1 + depth:.1f},{y + depth:.1f} H{m - depth:.1f} Q{m:.1f},{y + depth:.1f} {m:.1f},{y + 2 * depth:.1f} "
                f"Q{m:.1f},{y + depth:.1f} {m + depth:.1f},{y + depth:.1f} H{x2 - depth:.1f} Q{x2:.1f},{y + depth:.1f} {x2:.1f},{y:.1f}", color, 1.5)


def plus(cx, cy, r=9, color=INK):
    return (L(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{WHITE}" stroke="{color}" stroke-width="1.5"/>')
            + line(cx - r * 0.55, cy, cx + r * 0.55, cy, color, 1.5) + line(cx, cy - r * 0.55, cx, cy + r * 0.55, color, 1.5))


def cross(cx, cy, r=8, color="#DC2626"):
    return (L(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{WHITE}" stroke="{color}" stroke-width="1.5"/>')
            + line(cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5, color, 1.5) + line(cx - r * 0.5, cy + r * 0.5, cx + r * 0.5, cy - r * 0.5, color, 1.5))


def shape(x, y, s, anchor="middle"):
    return text(x, y, s, 10, MUTED, anchor, font=MONO)


# ----------------------------------------------------------------------------------------------- the panel frame
def panel(name, title, body, math, caption, file_title):
    """Title top left, the body, a monospace math box, a one-line caption at the bottom."""
    math_rows = math if isinstance(math, list) else [math]
    box_h = 14 + 16 * len(math_rows)
    box_y = 380 - 22 - box_h
    s = text(0, 12, title, 16, INK, "start", weight="bold")
    s += body
    s += rect(0, box_y, W, box_h, "grey", rx=6, sw=1, fill="#F9FAFB", stroke=P["grey"][0])
    for i, row in enumerate(math_rows):
        s += text(12, box_y + 7 + 8 + 16 * i, row, 11.5, INK, "start", font=MONO)
    s += text(0, 394, caption, 12, MUTED, "start")
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W + 2 * PAD} {H + 2 * PAD}" width="{W + 2 * PAD}" height="{H + 2 * PAD}" '
           f'font-family="{FONT}" font-size="13" fill="{INK}" stroke-linecap="round" stroke-linejoin="round">\n'
           f"  <title>{esc(file_title)}</title>\n  <g transform=\"translate({PAD} {PAD})\">\n{s}  </g>\n</svg>\n")
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write(doc)
    FILES.append((name, file_title))


# =============================================================================================== 0. overview
def overview():
    b = ""
    ya, yb, bh = 48, 176, 58
    # row A: photo -> frozen DINOv2 -> tokens -> LN + linear -> tokens -> per-photo encoder
    b += photo_icon(2, ya, bh, bh, uid="o")
    b += text(31, ya + bh + 12, "photo", 10, MUTED) + shape(31, ya + bh + 24, "224×224 px")
    b += arrow(64, ya + bh / 2, 86, ya + bh / 2)
    b += block(88, ya, 134, bh, "grey", ["DINOv2 ViT-B/14", "blocks 1–4"], "frozen, run once outside")
    b += badge(222, ya, "frozen")
    b += shape(155, ya + bh + 12, "(rows, 261, 768)") + shape(155, ya + bh + 24, "→ drop CLS + 4 reg")
    b += arrow(224, ya + bh / 2, 246, ya + bh / 2)
    b += block(248, ya, 96, bh, "purple", "patch tokens", "t, 768 wide")
    b += shape(296, ya + bh + 12, "(rows, 256, 768)")
    b += arrow(346, ya + bh / 2, 368, ya + bh / 2)
    b += block(370, ya, 116, bh, "orange", ["LayerNorm", "+ linear"], "768 → 128")
    b += badge(486, ya, "trained")
    b += shape(428, ya + bh + 12, "(rows, 256, 128)")
    b += arrow(488, ya + bh / 2, 510, ya + bh / 2)
    b += block(512, ya, 96, bh, "purple", "patch tokens", "z, 128 wide")
    b += shape(560, ya + bh + 12, "(rows, 256, 128)")
    b += arrow(610, ya + bh / 2, 632, ya + bh / 2)
    b += block(634, ya, 134, bh, "orange", ["per-photo encoder", "4 layers"], "+ position, 4 heads")
    b += badge(768, ya, "trained")
    b += shape(701, ya + bh + 12, "(rows, 256, 128)")
    # the return line into row B
    mid = ya + bh + 44
    b += path(f"M{701},{ya + bh + 30} V{mid} H{70} V{yb - 10}", MUTED, 1.5)
    b += head(70, yb - 2, 90, MUTED, 8)
    b += text(386, mid - 8, "the image side so far looks at one photo at a time", 10, MUTED)
    # row B: retrieval -> labelled read -> + marker -> append after cells -> row transformer -> in-context transformer
    b += block(2, yb, 136, bh, "blue", ["retrieval", "top-32 neighbours"], ["by cosine of the mean", "token; no weights"])
    b += shape(70, yb + bh + 12, "N(i): (rows, 32)")
    b += arrow(140, yb + bh / 2, 162, yb + bh / 2)
    b += block(164, yb, 118, bh, "orange", ["labelled read", "2 layers"], "values = labels")
    b += badge(282, yb, "trained")
    b += shape(223, yb + bh + 12, "(rows, 256, 128)")
    b += arrow(284, yb + bh / 2, 306, yb + bh / 2)
    b += block(308, yb, 82, bh, "purple", "+ marker m", "learned, 128")
    b += badge(390, yb, "trained")
    b += shape(349, yb + bh + 12, "(rows, 256, 128)")
    b += arrow(392, yb + bh / 2, 414, yb + bh / 2)
    b += block(416, yb, 136, bh, "purple", ["appended after", "the row's cells"], "[4 CLS | cells | 256 patches]")
    b += shape(484, yb + bh + 12, "(rows, 4+cols+256, 128)")
    b += arrow(554, yb + bh / 2, 576, yb + bh / 2)
    b += block(578, yb, 140, bh, "grey", ["TabICL row", "transformer"], "3 layers, 8 heads; pools")
    b += badge(718, yb, "trained")
    b += shape(648, yb + bh + 12, "(rows, 4×128 = 512)")
    b += arrow(720, yb + bh / 2, 742, yb + bh / 2)
    b += block(744, yb, 130, bh, "grey", ["TabICL in-context", "transformer"], "12 layers; rows → ŷ")
    b += badge(874, yb, "trained")
    b += shape(809, yb + bh + 12, "(test rows, classes)")
    # learning-rate legend
    ly = yb + bh + 40
    b += flame(10, ly, 7) + text(24, ly, "orange and + m: the image side, trained from scratch at lr 1e-4", 10.5, INK, "start")
    b += flame(368, ly, 7) + text(382, ly, "grey: TabICL v2's pretrained table stages, trained on at lr 1e-5", 10.5, INK, "start")
    b += snowflake(736, ly, 6, sw=1.5) + text(750, ly, "snowflake: frozen DINOv2", 10.5, INK, "start")
    panel("0-overview", "Overview: from the photo to TabICL's row transformer", b,
          ["t = DINO_1..4(x) ;   z = W·LN(t) + b ;   z = Enc(z + p) ;   z ← Read(z, N(z), y_ctx) + m",
           "row_i = RowTransformer([CLS ; cells_i ; z_i]) ;   ŷ = ICL(rows, y_ctx)"],
          "Shapes are (rows, patches, width): every row of the table, context or test, carries one photo through the same strip; only the labelled read looks across photos.",
          "Image side overview: photo, frozen DINOv2 blocks 1–4, LayerNorm + linear, per-photo encoder, retrieval, labelled read, marker, join with the row's cells, TabICL row and in-context transformers")


# =============================================================================================== 1. tokens
def tokens():
    b = ""
    # the photo cut into 16 x 16 patches
    gx, gy, gs = 4, 52, 170
    b += text(gx + gs / 2, 38, "photo, letterboxed to 224 × 224 px", 11, INK)
    b += patch_grid(gx, gy, gs, n=16, grid_sw=0.8, uid="t")
    b += highlight(gx, gy, gs, 16, 5, 6, "orange", 0.9)
    b += text(gx + gs / 2, gy + gs + 12, "16 × 16 = 256 patches of 14 px", 11, INK)
    b += text(gx + gs / 2, gy + gs + 26, "never stretched; padded to a square", 10, MUTED)
    b += arrow(gx + gs + 4, gy + gs / 2, gx + gs + 30, gy + gs / 2)
    # the frozen DINOv2 with the token row and the four blocks run
    fx, fy, fw, fh = 204, 46, 250, 214
    b += rect(fx, fy, fw, fh, "grey", dashed=False)
    b += badge(fx + fw, fy, "frozen")
    b += text(fx + 12, fy + 14, "DINOv2 ViT-B/14 with registers", 12, P["grey"][1], "start", weight="bold")
    b += text(fx + 12, fy + 29, "frozen; computed once, outside the model, no gradient", 9.5, MUTED, "start")
    # token row: CLS, 4 registers, 256 patches
    ty = fy + 52
    b += token(fx + 12, ty, 20, "yellow", "CLS", text_size=7)
    for i in range(4):
        b += token(fx + 36 + i * 22, ty, 20, "yellow", f"r{i + 1}", text_size=8)
    for i in range(6):
        b += token(fx + 130 + i * 16, ty + 3, 14, "blue", rx=3)
    b += text(fx + 230, ty + 10, "…", 12, P["blue"][1])
    b += text(fx + 12, ty + 32, "1 + 4 + 256 = 261 tokens × 768", 10, MUTED, "start")
    # the blocks
    by = ty + 50
    for i in range(4):
        b += rect(fx + 12 + i * 58, by, 52, 24, "grey", rx=5, sw=1.5, fill=WHITE)
        b += text(fx + 38 + i * 58, by + 12, f"block {i + 1}", 10, P["grey"][1], weight="bold")
        if i < 3:
            b += arrow(fx + 64 + i * 58, by + 12, fx + 70 + i * 58, by + 12, P["grey"][1], 1.5, size=6)
    b += rect(fx + 12, by + 36, 226, 22, "grey", rx=5, sw=1.2, dashed=True, fill="none")
    b += text(fx + 125, by + 47, "blocks 5 – 12 are not run", 9.5, MUTED)
    b += text(fx + 125, by + 74, "the chosen encoder is block4: a 4-block run", 9.5, MUTED)
    b += text(fx + 125, by + 86, "on 261 tokens, the output of block 4 taken", 9.5, MUTED)
    # out of the frozen box: drop CLS and registers
    b += arrow(fx + fw + 4, fy + fh / 2, fx + fw + 30, fy + fh / 2)
    ox = fx + fw + 36
    b += token(ox, 76, 20, "yellow", "CLS", text_size=7) + token(ox + 24, 76, 20, "yellow", "r1–4", text_size=6)
    b += line(ox - 2, 98, ox + 46, 74, "#DC2626", 2)
    b += text(ox + 22, 108, "dropped", 9.5, "#DC2626")
    # 256 x 768 token stack
    sx, sy = ox + 2, 126
    for i in range(5):
        b += rect(sx + i * 3, sy + i * 3, 46, 46, "purple", rx=5, sw=1.5)
    b += text(sx + 29, sy + 29, "t", 13, P["purple"][1], weight="bold", italic=True)
    b += text(sx + 30, sy + 76, "256 × 768", 10.5, INK)
    b += shape(sx + 30, sy + 90, "(rows, 256, 768)")
    # LayerNorm, linear
    b += arrow(sx + 64, sy + 30, sx + 88, sy + 30)
    lx = sx + 90
    b += block(lx, sy + 6, 72, 48, "orange", "LayerNorm", "over 768")
    b += arrow(lx + 74, sy + 30, lx + 96, sy + 30)
    b += block(lx + 98, sy + 6, 92, 48, "orange", "linear", "768 → 128")
    b += badge(lx + 190, sy + 6, "trained")
    b += text(lx + 144, sy + 68, "trained, lr 1e-4", 9.5, MUTED)
    b += arrow(lx + 192, sy + 30, lx + 214, sy + 30)
    zx = lx + 216
    for i in range(5):
        b += rect(zx + i * 3, sy + i * 3, 46, 46, "purple", rx=5, sw=1.5)
    b += text(zx + 29, sy + 29, "z", 13, P["purple"][1], weight="bold", italic=True)
    b += text(zx + 30, sy + 76, "256 × 128", 10.5, INK)
    b += shape(zx + 30, sy + 90, "(rows, 256, 128)")
    b += text(W, sy + 132, "the model's width: TabICL's", 10, MUTED, "end") + text(W, sy + 146, "cell tokens are 128 wide too", 10, MUTED, "end")
    b += text(sx - 2, sy + 132, "one token per patch, row-major;", 10, MUTED, "start") + text(sx - 2, sy + 146, "no photo sees another here", 10, MUTED, "start")
    panel("1-tokens", "Patch tokens: a frozen DINOv2 run, then LayerNorm and a linear map to width 128", b,
          "z_i = W · LN(t_i) + b,   t_i ∈ R^{256×768} (block 4 of DINOv2, CLS and registers dropped),   z_i ∈ R^{256×128}",
          "The same for every row's photo, context and test alike; the frozen tokens are computed once, outside the model.",
          "Patch tokens: one photo cut into 16 × 16 = 256 patches of 14 px, DINOv2 adds CLS + 4 registers (261 tokens), runs blocks 1–4 only, drops CLS and registers, then LayerNorm and a linear map 768 → 128")


# =============================================================================================== 2. per-photo encoder
def encoder():
    b = ""
    gs, n = 112, 4
    ax, ay = 24, 62
    bx, by = 232, 62
    b += text(ax + gs / 2, 44, "photo A", 11, INK, weight="bold") + text(bx + gs / 2, 44, "photo B", 11, INK, weight="bold")
    b += patch_grid(ax, ay, gs, n, uid="a") + patch_grid(bx, by, gs, n, uid="b")
    # position embedding added first: a small purple grid with "+ p" under each photo
    for x0 in (ax, bx):
        b += plus(x0 + gs / 2, ay + gs + 16, 8)
        px, py = x0 + gs / 2 - 20, ay + gs + 32
        for r in range(4):
            for c in range(4):
                b += rect(px + c * 10, py + r * 10, 9, 9, "purple", rx=2, sw=1)
        b += text(x0 + gs / 2, py + 50, "learned position p", 10, INK) + text(x0 + gs / 2, py + 62, "one per patch, 256 × 128", 9.5, MUTED)
    # arcs inside A: one patch attends to the others of its own photo
    qa = cell_center(ax, ay, gs, n, 1, 1)
    b += highlight(ax, ay, gs, n, 1, 1)
    for (c, r), bend in (((3, 0), -14), ((0, 3), 14), ((3, 3), -18), ((2, 2), -6), ((0, 0), 10)):
        tx, ty = cell_center(ax, ay, gs, n, c, r)
        b += arc(qa[0], qa[1], tx, ty, bend)
    # arcs inside B
    qb = cell_center(bx, by, gs, n, 2, 1)
    b += highlight(bx, by, gs, n, 2, 1)
    for (c, r), bend in (((0, 0), 10), ((3, 3), 12), ((0, 3), -16), ((1, 2), 8)):
        tx, ty = cell_center(bx, by, gs, n, c, r)
        b += arc(qb[0], qb[1], tx, ty, bend)
    # no arcs between the photos
    mx = (ax + gs + bx) / 2
    b += line(mx, ay - 6, mx, ay + gs + 6, P["grey"][1], 1.5, dashed=True)
    b += cross(mx, ay + gs / 2, 9)
    b += text(mx, ay + gs + 20, "no attention", 9.5, "#DC2626") + text(mx, ay + gs + 31, "between photos", 9.5, "#DC2626")
    # the layer, as a horizontal chain with residual bypasses, x 4
    lx, ly, lh = 392, 108, 44
    b += rect(lx - 8, 44, 492, 232, "grey", rx=10, sw=1.2, dashed=True, fill="none")
    b += text(lx, 58, "one layer, applied to each photo's 256 tokens; × 4 layers", 11, INK, "start", weight="bold")
    b += badge(lx + 484, 44, "trained")
    cy = ly + lh / 2
    b += text(lx + 4, cy, "z", 12, P["purple"][1], "start", weight="bold", italic=True)
    b += arrow(lx + 14, cy, lx + 34, cy)
    b += block(lx + 36, ly, 44, lh, "orange", "LN", size=11)
    b += arrow(lx + 82, cy, lx + 102, cy)
    b += block(lx + 104, ly, 100, lh, "orange", "self-attention", "4 heads, Q = K = V", size=11, sub_size=9)
    b += arrow(lx + 206, cy, lx + 232 - 9, cy)
    b += plus(lx + 232, cy, 9)
    b += path(f"M{lx + 24},{cy} V{cy - 34} H{lx + 232} V{cy - 10}", INK, 1.5)
    b += head(lx + 232, cy - 9, 90, INK, 7)
    b += arrow(lx + 241, cy, lx + 262, cy)
    b += block(lx + 264, ly, 44, lh, "orange", "LN", size=11)
    b += arrow(lx + 310, cy, lx + 330, cy)
    b += block(lx + 332, ly, 112, lh, "green", "MLP", "GELU, 128 → 256 → 128", size=11, sub_size=8.5)
    b += arrow(lx + 446, cy, lx + 470 - 9, cy)
    b += plus(lx + 470, cy, 9)
    b += path(f"M{lx + 252},{cy} V{cy - 34} H{lx + 470} V{cy - 10}", INK, 1.5)
    b += head(lx + 470, cy - 9, 90, INK, 7)
    b += arrow(lx + 479, cy, lx + 482, cy)
    b += text(lx + 236, cy + 42, "pre-norm, as in ViT and DINOv2; every token keeps its own position in the 256", 9.5, MUTED)
    b += shape(lx + 236, cy + 56, "(rows, 256, 128) in  →  (rows, 256, 128) out")
    b += text(lx + 236, cy + 84, "the keys of each attention are the 256 patches of the same photo and nothing else:", 10, INK)
    b += text(lx + 236, cy + 98, "photos of all rows are flattened to (rows × 256, 128) and encoded independently", 10, INK)
    b += text(lx + 236, cy + 112, "(no label is involved: this is where the model learns what a patch is)", 10, MUTED)
    panel("2-per-photo-encoder", "Per-photo encoder: 4 transformer layers over the patches of one photo", b,
          "z ← z + p ;   for l in 1..4:   z ← z + Attn(LN z, LN z, LN z) ;   z ← z + MLP(LN z)",
          "Patches of one photo attend to each other; photos do not see each other yet.",
          "Per-photo encoder: a learned position embedding is added, then 4 pre-norm transformer layers in which the patches of one photo attend to each other and never to another photo")


# =============================================================================================== 3. retrieval
def retrieval():
    b = ""
    ps, rh = 34, 44
    # readers: every photo, context and test
    rx, ry = 4, 56
    b += text(rx, 40, "every photo (readers)", 11, INK, "start", weight="bold")
    readers = [("c1", "yellow", "1"), ("c2", "yellow", "0"), ("c3", "yellow", "1"), ("t1", "grey", "?"), ("t2", "grey", "?")]
    for i, (name, kind, lab) in enumerate(readers):
        y = ry + i * rh
        b += text(rx + 8, y + ps / 2, name, 10, MUTED)
        b += photo_icon(rx + 20, y, ps, ps, rx=3, sw=1.5, uid="r")
        b += chip(rx + 58, y + 9, 20, 16, kind, lab)
        b += arrow(rx + 84, y + ps / 2, rx + 104, y + ps / 2, MUTED, 1.2, size=6)
        b += token(rx + 106, y + 5, 24, "purple", "m", text_size=10)
    b += text(rx + 118, ry + 5 * rh + 2, "m_i: mean of its", 9.5, MUTED) + text(rx + 118, ry + 5 * rh + 13, "256 patch tokens", 9.5, MUTED)
    b += text(rx + 40, ry + 5 * rh + 2, "context rows", 9.5, MUTED) + text(rx + 40, ry + 5 * rh + 13, "and test rows", 9.5, MUTED)
    # sources: the context photos
    sx, sy = 392, 56
    b += text(sx + 70, 40, "the context photos (sources, up to 256)", 11, INK, "start", weight="bold")
    sources = [("c1", "1"), ("c2", "0"), ("c3", "1"), ("c4", "0"), ("c5", "1")]
    for j, (name, lab) in enumerate(sources):
        y = sy + j * rh
        b += token(sx, y + 5, 24, "purple", "m", text_size=10)
        b += arrow(sx + 30, y + ps / 2, sx + 46, y + ps / 2, MUTED, 1.2, size=6)
        b += photo_icon(sx + 50, y, ps, ps, rx=3, sw=1.5, uid="s")
        b += chip(sx + 88, y + 9, 20, 16, "yellow", lab)
        b += text(sx + 122, y + ps / 2, name, 10, MUTED)
    # cosine arrows from test photo t1 (reader 4) to its 3 nearest sources
    qx, qy = rx + 130, ry + 3 * rh + 17
    hits = ((1, "0.91", -10, (326, 150)), (3, "0.84", -4, (300, 196)), (0, "0.79", 22, (206, 150)))
    for jj, s, bend, (tx, yy) in hits:
        ty = sy + jj * rh + 17
        b += arc(qx, qy, sx - 2, ty, bend, P["orange"][1], 1.8)
        b += text(tx, yy, f"cos = {s}", 9.5, P["orange"][1])
    for jj in (2, 4):
        ty = sy + jj * rh + 17
        b += arc(qx, qy, sx - 2, ty, 6, P["grey"][1], 1.2, dashed=True, arrowhead=False)
    b += text(300, 282, "s_ij = cos(m_i, m_j) for every source j", 10, INK)
    b += text(300, 295, "one 128-d dot product per pair; the grey pairs lose", 9.5, MUTED)
    # a context photo's nearest is itself
    cx0, cy0 = rx + 130, ry + 17
    b += arc(cx0, cy0, sx - 2, sy + 17, -16, P["yellow"][1], 1.5)
    b += text(262, 48, "c1 → c1: cos = 1, a context photo's nearest is itself", 9.5, P["yellow"][1])
    # the result
    ox, oy = 572, 56
    b += rect(ox, oy, 308, 60, "orange", rx=8, sw=1.5)
    b += text(ox + 154, oy + 14, "N(t1) = top-3 by s = {c2, c4, c1}", 11.5, P["orange"][1], weight="bold", font=MONO)
    b += text(ox + 154, oy + 32, "k = 3 drawn here; k = 32 in the model", 10, INK)
    b += text(ox + 154, oy + 47, "ModelConfig.patch_top_k = 32, max_sources = 256", 9.5, MUTED, font=MONO)
    b += lines(ox + 4, oy + 84, ["what retrieval does and does not do:",
                                  "• no weights: the mean, two normalisations, one matmul, topk",
                                  "• runs once per table on the stage's input tokens, before layer 1",
                                  "• runs only when there are more than k sources; with ≤ 32",
                                  "  context photos every source is read (nearest = None)",
                                  "• a context photo retrieves itself, so it will read its own label",
                                  "• the attention a photo faces is then the same size for any",
                                  "  context, which is what lets a model trained on 256-row",
                                  "  tables be scored on 10,000-row ones"], 9.8, INK, "start", lead=13.5)
    panel("3-retrieval", "Retrieval: every photo picks its k nearest context photos by their mean token", b,
          "m_i = (1/256) Σ_p z_{i,p} ;   s_ij = cos(m_i, m_j), j ∈ context ;   N(i) = top-k_j s_ij   (k = 32)",
          "Retrieval is by content only: no label is used to pick the neighbours; it costs one 128-dimensional dot product per pair.",
          "Retrieval: every photo, context or test, is summarised by the mean of its 256 patch tokens; cosine similarity to the mean tokens of the context photos picks its top-32 neighbours; a context photo's nearest is itself")


# =============================================================================================== 4. labelled read
def labelled_read():
    b = ""
    n = 4
    # the reading photo: queries
    rx, ry, rs = 10, 60, 120
    b += text(rx + rs / 2, 42, "reading photo i: queries", 11, INK, weight="bold")
    b += patch_grid(rx, ry, rs, n, uid="q")
    qc, qr = 2, 1
    b += highlight(rx, ry, rs, n, qc, qr)
    qx, qy = cell_center(rx, ry, rs, n, qc, qr)
    b += text(rx + rs / 2, ry + rs + 12, "q = LN_q(z_i), one query per patch", 9.8, INK)
    b += text(rx + rs / 2, ry + rs + 24, "context or test: every photo reads", 9.5, MUTED)
    # the neighbours: keys
    kx, ks = 262, 96
    kys = (50, 166)
    b += text(kx + ks / 2 + 10, 42, "its k neighbours j ∈ N(i): keys", 11, INK, weight="bold")
    labs = ("1", "0")
    keys = (((0, 1), (2, 2)), ((1, 0),))
    weights = (("0.5", "0.3"), ("0.2",))
    for j, ky in enumerate(kys):
        b += patch_grid(kx, ky, ks, n, uid=f"k{j}")
        b += chip(kx + ks + 6, ky + 2, 22, 16, "yellow", labs[j])
        b += text(kx + ks + 17, ky + 30, f"y = {labs[j]}", 9.5, MUTED)
        for (c, r), wgt in zip(keys[j], weights[j]):
            tx, ty = cell_center(kx, ky, ks, n, c, r)
            b += highlight(kx, ky, ks, n, c, r, "orange", 0.45)
            b += arc(qx, qy, tx - 6, ty, -10 if j == 0 else 14, P["orange"][1], 1.8)
            dx, dy = (8, -13) if (c, r) == (0, 1) else ((8, 11) if j == 0 else (-16, 26))
            b += text((qx + tx) / 2 + dx, (qy + ty) / 2 + dy, f"a = {wgt}", 9.5, P["orange"][1])
    b += text(kx + ks / 2, 154, "…", 16, MUTED) + text(kx + ks / 2 + 48, 278, "k = 32 photos × 256 patches of keys", 9.5, MUTED)
    b += text(kx + ks / 2 + 4, 291, "k_j = LN_k(z_j): the stage's input tokens, label-free", 9.5, MUTED)
    # the values: the neighbour's label embedding repeated per patch
    vx = 456
    b += text(vx + ks / 2 + 10, 42, "values: the neighbour's label, per patch", 11, INK, weight="bold")
    for j, ky in enumerate(kys):
        b += arrow(kx + ks + 36, ky + ks / 2, vx - 8, ky + ks / 2, MUTED, 1.2, size=6)
        for r in range(n):
            for c in range(n):
                b += token(vx + c * 24, ky + r * 24, 22, "yellow", f"e{labs[j]}", rx=3, text_size=8.5)
        b += text(vx + ks + 24, ky + ks / 2 - 7, f"e_y = Embed({labs[j]})", 9.5, INK, "start")
        b += text(vx + ks + 24, ky + ks / 2 + 7, "16 copies (256 in the model)", 9, MUTED, "start")
    b += text(vx + ks / 2, 154, "…", 16, MUTED)
    b += text(vx + ks / 2 + 60, 278, "not the patches: the label embedding e_{y_j} ∈ R^128", 9.5, MUTED)
    b += text(vx + ks / 2 + 60, 291, "of the photo a key patch belongs to (nn.Embedding(10, 128))", 9.5, MUTED)
    # what flows back: the weighted label embeddings into r_i, added to the query patch
    ox, oy = 700, 112
    b += rect(ox, oy, 180, 96, "yellow", rx=8, sw=1.5)
    b += text(ox + 90, oy + 14, "what flows back", 11, P["yellow"][1], weight="bold")
    b += text(ox + 90, oy + 32, "r = 0.5·e1 + 0.3·e1 + 0.2·e0", 10, INK, font=MONO)
    b += text(ox + 90, oy + 48, "a soft vote: mostly \"1\"", 10, INK)
    b += text(ox + 90, oy + 64, "z_i ← z_i + r  (query patch)", 10, INK, font=MONO)
    b += text(ox + 90, oy + 80, "then z_i ← z_i + MLP(LN z_i)", 10, INK, font=MONO)
    for j, ky in enumerate(kys):
        for (c, r) in keys[j]:
            vxx, vyy = vx + c * 24 + 22, ky + r * 24 + 11
            b += arc(vxx, vyy, ox - 2, oy + 48 + (j * 2 - 1) * 10, 10 if j == 0 else -10, P["yellow"][1], 1.5, dashed=True)
    b += text(ox + 90, oy + 110, "4 heads, head width 32; two separate", 9.5, MUTED)
    b += text(ox + 90, oy + 122, "LayerNorms (readers, sources); the", 9.5, MUTED)
    b += text(ox + 90, oy + 134, "attention has its own q, k, v, o maps", 9.5, MUTED)
    b += text(ox + 90, oy + 160, "a context photo is among its own", 9.5, INK)
    b += text(ox + 90, oy + 172, "neighbours: it reads its own label too", 9.5, INK)
    panel("4-labelled-read", "Labelled read: content decides where to look, labels are what is read back", b,
          ["q = LN_q(z_i),   k_j = LN_k(z_j) for j ∈ N(i),   v_{j,p} = e_{y_j} for every patch p of photo j",
           "r_i = softmax(q kᵀ / √d) v ;   z_i ← z_i + r_i ;   z_i ← z_i + MLP(LN z_i)     (two such layers; keys = stage input both times)"],
          "A soft nearest-neighbour vote per patch. A context photo reads its own label too; a test photo is never a source, so it cannot read its own.",
          "Labelled read: a reading photo's patches are the queries, the patches of its k neighbour photos are the keys, and the values are each neighbour's label embedding repeated for every one of its patches; what flows back is a weighted sum of labels")


# =============================================================================================== 5. join
def join():
    b = ""
    ts, gap = 26, 4
    y = 150
    x = 8
    # the row's tokens: 4 CLS, the cells, 256 patch tokens + m
    b += text(x, 42, "row i, as the row transformer sees it: one token sequence", 11, INK, "start", weight="bold")
    cls_x = x
    for i in range(4):
        b += token(x, y, ts, "purple", "CLS", hollow=True, text_size=7.5)
        x += ts + gap
    cls_end = x - gap
    x += 10
    cells_x = x
    for i in range(5):
        b += token(x, y, ts, "purple", f"c{i + 1}", text_size=9)
        x += ts + gap
    cells_end = x - gap
    x += 10
    patch_x = x
    for i in range(7):
        b += token(x, y, ts, "blue", f"p{i + 1}", text_size=8.5)
        x += ts + gap
    b += text(x + 8, y + ts / 2, "…", 14, P["blue"][1])
    x += 24
    b += token(x, y, ts, "blue", "p256", text_size=7)
    x += ts
    patch_end = x

    # braces
    b += hbrace(cls_x, cls_end, y + ts + 6)
    b += text((cls_x + cls_end) / 2, y + ts + 26, "4 CLS tokens", 9.5, MUTED)
    b += text((cls_x + cls_end) / 2, y + ts + 37, "TabICL's readout", 9, MUTED)
    b += hbrace(cells_x, cells_end, y + ts + 6)
    b += text((cells_x + cells_end) / 2, y + ts + 26, "cell tokens of row i", 9.5, MUTED)
    b += text((cells_x + cells_end) / 2, y + ts + 37, "from the column stage", 9, MUTED)
    b += hbrace(patch_x, patch_end, y + ts + 6)
    b += text((patch_x + patch_end) / 2, y + ts + 26, "256 patch tokens z_i + m, after the labelled read", 9.5, MUTED)
    b += text((patch_x + patch_end) / 2, y + ts + 37, "concatenated along the token axis: torch.cat([cells, patches], 2)", 9, MUTED)
    b += text((patch_x + patch_end) / 2, y + ts + 49, "m: one learned 128-vector added to every patch token, so the row stage can tell a patch from a cell", 9, INK)
    b += shape(8, y + ts + 62, "(rows, 4 + columns + 256, 128)", "start")
    # the row transformer above: arcs from a CLS to cells and patches
    qx, qy = cls_x + ts / 2, y
    for tx, bend in ((cells_x + ts / 2, -14), (cells_x + 3 * (ts + gap) + ts / 2, -24), (patch_x + ts / 2, -34), (patch_x + 4 * (ts + gap) + ts / 2, -46), (patch_end - ts / 2, -60)):
        b += arc(qx, qy, tx, y, bend)
    b += arc(cells_x + 2 * (ts + gap) + ts / 2, y, patch_x + 2 * (ts + gap) + ts / 2, y, -22, P["orange"][1], 1.2, dashed=True)
    b += rect(8, 54, 452, 52, "grey", rx=8)
    b += badge(460, 54, "trained")
    b += text(16, 68, "TabICL row transformer: 3 layers, 8 heads", 11.5, P["grey"][1], "start", weight="bold")
    b += text(16, 83, "attends across all tokens of the row (cells and patches alike, with rotary positions);", 9.5, MUTED, "start")
    b += text(16, 95, "the 4 CLS outputs, concatenated, are the row embedding. Pretrained weights, trained on at lr 1e-5", 9.5, MUTED, "start")
    # pooling into the row embedding
    ex = patch_end + 30
    b += arrow(patch_end + 6, y + ts / 2, ex - 4, y + ts / 2)
    b += rect(ex, y - 10, 56, 46, "purple", rx=8, sw=2)
    b += text(ex + 28, y + 7, "row_i", 11, P["purple"][1], weight="bold")
    b += text(ex + 28, y + 22, "4 × 128", 9, MUTED)
    b += shape(ex + 28, y + 50, "(rows, 512)")
    # the in-context transformer on the right
    ix, iy = ex + 76, 54
    b += arrow(ex + 58, y + ts / 2, ix - 4, y + ts / 2)
    b += rect(ix, iy, W - ix, 262, "grey", rx=8, dashed=False)
    b += badge(W, iy, "trained")
    b += text(ix + 10, iy + 14, "in-context transformer", 11.5, P["grey"][1], "start", weight="bold")
    b += text(ix + 10, iy + 28, "12 layers; rows attend to rows", 9.5, MUTED, "start")
    rx0, ry0 = ix + 44, iy + 46
    ctx = (("1", "yellow"), ("0", "yellow"), ("1", "yellow"))
    for r, (lab, kind) in enumerate(ctx):
        yy = ry0 + r * 32
        b += rect(rx0, yy, 44, 20, "purple", rx=5, sw=1.5) + text(rx0 + 22, yy + 10, f"row c{r + 1}", 8.5, P["purple"][1], weight="bold")
        b += chip(rx0 + 50, yy + 2, 20, 16, kind, lab)
    b += text(rx0 - 4, ry0 + 3 * 32 - 4, "context rows + their labels", 9, MUTED, "start")
    tys = (ry0 + 3 * 32 + 10, ry0 + 3 * 32 + 42)
    for r, ty in enumerate(tys):
        b += rect(rx0, ty, 44, 20, "purple", rx=5, sw=1.5) + text(rx0 + 22, ty + 10, f"row t{r + 1}", 8.5, P["purple"][1], weight="bold")
        b += chip(rx0 + 50, ty + 2, 20, 16, "grey", "?")
    for r in range(3):
        b += arc(rx0 - 3, tys[0] + 10, rx0 - 3, ry0 + r * 32 + 12, -(14 + 5 * (2 - r)), P["orange"][1], 1.3)
    b += text(rx0 + 36, tys[1] + 32, "test rows read the", 9, MUTED) + text(rx0 + 36, tys[1] + 43, "context rows; ŷ per test row", 9, MUTED)
    b += text(ix + (W - ix) / 2, iy + 250, "no photo here: only row embeddings", 9.5, INK)
    panel("5-join", "Join: the patch tokens enter the row as extra tokens; the in-context stage sees rows only", b,
          "tokens_i = [CLS ; cells_i ; z_i + m],   row_i = RowTransformer(tokens_i),   ŷ = ICL(rows, y_context)",
          "Every row, context or test, carries its photo inside its row embedding; the in-context stage never touches photos.",
          "Join: a row's 4 CLS tokens, its cell tokens and its 256 patch tokens plus the marker are one token sequence for TabICL's row transformer, which pools them into the row embedding; the in-context transformer then attends row to row")


# =============================================================================================== 6. who attends to what
def who_attends():
    b = ""
    cols = [("stage", 142), ("queries", 150), ("keys", 178), ("values", 182), ("across photos?", 104), ("uses labels?", 128)]
    rows = [
        ("per-photo encoder", ["the 256 patches", "of photo i"], ["the 256 patches", "of photo i"], ["the 256 patches", "of photo i"], ["no"], ["no"], "orange"),
        ("retrieval", ["mean token m_i", "of photo i"], ["mean tokens of the", "context photos (≤ 256)"], ["none: cosine + top-k", "gives the indices N(i)"], ["yes"], ["no"], "blue"),
        ("labelled read", ["the 256 patches", "of photo i"], ["the patches of its", "k = 32 neighbours", "(stage input, label-free)"], ["the neighbours' label", "embeddings e_y, one per", "neighbour, repeated per patch"], ["yes"], ["yes", "(values only)"], "orange"),
        ("row transformer", ["4 CLS + cells +", "256 patches of row i"], ["the same tokens", "of row i"], ["the same tokens", "of row i"], ["no", "(one row, one photo)"], ["no"], "grey"),
        ("in-context transformer", ["row embeddings", "(context and test)"], ["context row", "embeddings"], ["context row embeddings", "(label added to each)"], ["across rows:", "yes"], ["yes", "(context labels)"], "grey"),
    ]
    x0, y0, hh, rh = 0, 38, 24, 50
    xs = [x0]
    for _, w in cols:
        xs.append(xs[-1] + w)
    # header
    b += rect(x0, y0, xs[-1] - x0, hh, "grey", rx=6, sw=1.2, fill="#F3F4F6", stroke=P["grey"][0])
    for (name, w), x in zip(cols, xs):
        b += text(x + w / 2, y0 + hh / 2, name, 11, INK, weight="bold")
    # rows
    for r, (stage, q, k, v, across, labels, kind) in enumerate(rows):
        y = y0 + hh + r * rh
        b += line(x0, y + rh, xs[-1], y + rh, P["grey"][0], 1)
        b += rect(xs[0] + 4, y + rh / 2 - 13, cols[0][1] - 8, 26, kind, rx=6, sw=1.5)
        b += text(xs[0] + cols[0][1] / 2, y + rh / 2, stage, 10.5, P[kind][1], weight="bold")
        for cells, (name, w), x in zip((q, k, v, across, labels), cols[1:], xs[1:]):
            yy = y + rh / 2 - (len(cells) - 1) * 6.5
            colour = INK
            if name in ("across photos?", "uses labels?"):
                colour = P["orange"][1] if cells[0].startswith("yes") or cells[0].startswith("across") else P["green"][1]
            b += lines(x + w / 2, yy, cells, 10, colour, lead=13, weight="bold" if colour != INK else None)
    for x in xs[1:-1]:
        b += line(x, y0, x, y0 + hh + len(rows) * rh, P["grey"][0], 1)
    b += rect(x0, y0, xs[-1] - x0, hh + len(rows) * rh, "grey", rx=6, sw=1.2, fill="none", stroke=P["grey"][1])
    panel("6-who-attends-to-what", "Who attends to what: queries, keys and values of every stage", b,
          "Attn(Q, K, V) = softmax(Q Kᵀ / √d) V  —  the table says where Q, K and V come from at each stage",
          "Only the labelled read mixes photos, and only through its values; the table stages are TabICL's, trained on at lr 1e-5 with the photo tokens in the rows.",
          "Who attends to what: a table of the queries, keys and values of the per-photo encoder, retrieval, the labelled read, the row transformer and the in-context transformer, and whether each looks across photos or uses labels")


# =============================================================================================== preview sheet
def preview():
    cards = "\n".join(
        f'<figure><img src="{n}.svg" alt="{esc(t)}" width="{W + 2 * PAD}" height="{H + 2 * PAD}"><figcaption><code>{n}.svg</code><span>{esc(t)}</span></figcaption></figure>'
        for n, t in FILES)
    swatches = "".join(f'<div class="sw"><i style="background:{P[k][0]};border-color:{P[k][1]}"></i>{lab}</div>' for k, lab in (
        ("blue", "data, parameter-free"), ("orange", "attention, trained from scratch (lr 1e-4)"), ("green", "MLP"), ("grey", "TabICL stages (lr 1e-5) / frozen DINOv2"),
        ("purple", "tokens"), ("yellow", "labels")))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image side of the model: panels</title>
<style>
  body {{ margin: 0; padding: 24px; font-family: {FONT}; color: {INK}; background: #fff; width: {2 * (W + 2 * PAD) + 18}px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, {W + 2 * PAD}px); gap: 18px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p {{ margin: 0 0 14px; color: {MUTED}; font-size: 13px; }}
  figure {{ margin: 0; border: 1px solid #E5E7EB; border-radius: 10px; overflow: hidden; background: #fff; }}
  figure img {{ display: block; background: repeating-conic-gradient(#F3F4F6 0 25%, #fff 0 50%) 0 0 / 16px 16px; }}
  figcaption {{ padding: 8px 10px; border-top: 1px solid #E5E7EB; font-size: 12px; }}
  figcaption code {{ display: block; font-weight: bold; color: {P["blue"][1]}; }}
  figcaption span {{ color: {MUTED}; }}
  .swatches {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }}
  .sw {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
  .sw i {{ width: 26px; height: 18px; border-radius: 5px; border: 2px solid; display: inline-block; }}
</style></head><body>
<h1>The image side of the model</h1>
<p>{len(FILES)} SVG panels in <code>docs/figures/image-side/</code>: from the photo to the point where its patch tokens join TabICL's row transformer. Shapes are (rows, patches, width).</p>
<div class="swatches">{swatches}</div>
<div class="grid">
{cards}
</div>
</body></html>
"""
    with open(os.path.join(OUT, "preview.html"), "w") as f:
        f.write(page)
    return 24 * 2 + 130 + ((len(FILES) + 1) // 2) * (H + 2 * PAD + 40 + 18)


def preview_png(height):
    """Render preview.html with headless Chrome when it is installed; otherwise leave the old preview.png alone."""
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        print("Chrome not found; preview.png not rendered")
        return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={2 * (W + 2 * PAD) + 18 + 48},{height}",
                    f"--screenshot={os.path.join(OUT, 'preview.png')}", "file://" + os.path.join(OUT, "preview.html")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    overview()
    tokens()
    encoder()
    retrieval()
    labelled_read()
    join()
    who_attends()
    height = preview()
    preview_png(height)
    print(len(FILES), "panels + preview.html written to", OUT)
