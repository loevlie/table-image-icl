"""The image in-context read (class LabelledRead in table_image_icl/model.py) explained in detail: one tall data-flow
figure with every tensor and operation of a forward pass, and three worked examples whose numbers come from running
the real module on a tiny hand-set instance.

    PYTHONPATH=. python3 docs/figures/image-icl/image_icl.py [out_dir]

Writes full-dataflow.svg, example-1.svg, example-2.svg, example-3.svg, examples.json (every number drawn in the
example panels, recomputed from the code on each run), preview.html and, with Chrome installed, preview.png.
Style, shared with docs/figures/image-side: 2 px rounded strokes, pastel fill + darker stroke per module type,
Helvetica text as real <text> elements, transparent background, a <title> per file, no filters, images or fonts.
"""
import html
import json
import math
import os
import subprocess
import sys

import torch

from table_image_icl.model import LabelledRead, nearest_photos

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------------- style
PAD = 8
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, WHITE, RED = "#1F2937", "#6B7280", "#FFFFFF", "#DC2626"
P = {  # fill, stroke
    "blue": ("#DBEAFE", "#2563EB"),       # data, parameter-free operations
    "orange": ("#FFEDD5", "#EA580C"),     # attention, trained
    "green": ("#DCFCE7", "#16A34A"),      # MLP
    "grey": ("#E5E7EB", "#6B7280"),       # notes, inactive
    "purple": ("#EDE9FE", "#7C3AED"),     # tensors of content only
    "yellow": ("#FEF3C7", "#D97706"),     # tensors that involve labels
}
FILES = []


# ----------------------------------------------------------------------------------------------- primitives
def L(s):
    return "    " + s + "\n"


def esc(s):
    return html.escape(str(s), quote=False)


def rect(x, y, w, h, kind, rx=8, dashed=False, sw=2, fill=None, stroke=None):
    f, s = P[kind] if kind in P else (fill, stroke)
    f, s = fill or f, stroke or s
    d = ' stroke-dasharray="6 4"' if dashed else ""
    return L(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, color=INK, anchor="middle", weight=None, italic=False, font=None):
    """Text centred vertically on y."""
    w = f' font-weight="{weight}"' if weight else ""
    i = ' font-style="italic"' if italic else ""
    f = f' font-family="{font}"' if font else ""
    return L(f'<text x="{x:.1f}" y="{y + size * 0.36:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}{i}{f}>{esc(s)}</text>')


def lines(x, y, rows, size=11, color=INK, anchor="middle", lead=None, weight=None, font=None):
    lead = lead or size + 3
    return "".join(text(x, y + i * lead, r, size, color, anchor, weight, font=font) for i, r in enumerate(rows))


def line(x1, y1, x2, y2, color=INK, sw=2, dashed=False):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    return L(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')


def head(x, y, angle, color=INK, size=9):
    a = math.radians(angle)
    bx, by = x - size * math.cos(a), y - size * math.sin(a)
    px, py = -math.sin(a) * size * 0.5, math.cos(a) * size * 0.5
    return L(f'<polygon points="{x:.1f},{y:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{color}" stroke="none"/>')


def arrow(x1, y1, x2, y2, color=INK, sw=2, dashed=False, size=9):
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    a = math.radians(ang)
    return line(x1, y1, x2 - 0.75 * size * math.cos(a), y2 - 0.75 * size * math.sin(a), color, sw, dashed) + head(x2, y2, ang, color, size)


def elbow(x1, y1, x2, y2, color=INK, sw=1.5, via="v"):
    """Orthogonal connector: down then across ("v") or across then down ("h"), ending in an arrowhead."""
    if via == "v":
        ym = (y1 + y2) / 2
        d = f"M{x1:.1f},{y1:.1f} V{ym:.1f} H{x2:.1f} V{y2 - 7:.1f}"
        return L(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"/>') + head(x2, y2, 90, color, 8)
    d = f"M{x1:.1f},{y1:.1f} H{x2:.1f} V{y2 - 7:.1f}"
    return L(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"/>') + head(x2, y2, 90, color, 8)


def chip(x, y, w, h, kind, label, size=10):
    return rect(x, y, w, h, kind, rx=h / 2, sw=1.5) + text(x + w / 2, y + h / 2, label, size, P[kind][1], weight="bold")


def flame(cx, cy, s):
    outer = (f'<path transform="translate({cx:.1f} {cy:.1f}) scale({s / 11:.3f})" d="M0,-11 C5,-6 8,-1 8,4 C8,9 4.5,12 0,12 C-4.5,12 -8,9 -8,4 '
             f'C-8,0 -5.5,-2 -4,-5 C-3.5,-2 -2,-0.5 -0.5,0.5 C1,-3 0.5,-7 0,-11 Z" fill="#F97316" stroke="{P["orange"][1]}" stroke-width="{11 / s * 2:.2f}"/>')
    inner = (f'<path transform="translate({cx:.1f} {cy:.1f}) scale({s / 11:.3f})" d="M0.5,2 C3,4.5 4,6.5 4,8.5 C4,10.5 2,12 0,12 C-2,12 -4,10.5 -4,8.5 '
             f'C-4,6.5 -2.5,5 -1,2.5 C-0.8,4 0,4.5 0.5,2 Z" fill="#FDE68A" stroke="none"/>')
    return L(outer) + L(inner)


def node(x, y, w, h, kind, label, shape=None, sub=None, size=11, dashed=False, trained=False):
    """A box with a bold label (str or lines), a monospace shape line and an optional muted sub-line."""
    s = rect(x, y, w, h, kind, dashed=dashed)
    labels = label if isinstance(label, list) else [label]
    subs = [] if sub is None else (sub if isinstance(sub, list) else [sub])
    n = len(labels) * (size + 2) + (12 if shape else 0) + len(subs) * 11
    cy = y + h / 2 - n / 2 + (size + 2) / 2
    s += lines(x + w / 2, cy, labels, size, P[kind][1], lead=size + 2, weight="bold")
    cy += len(labels) * (size + 2) - (size + 2) / 2
    if shape:
        cy += 6
        s += text(x + w / 2, cy, shape, 9.5, INK, font=MONO)
        cy += 6
    if subs:
        s += lines(x + w / 2, cy + 5.5, subs, 9, MUTED, lead=11)
    if trained:
        s += flame(x + w - 9, y + 9, 5)
    return s


def svg_file(name, title, w, h, body):
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w + 2 * PAD} {h + 2 * PAD}" width="{w + 2 * PAD}" height="{h + 2 * PAD}" '
           f'font-family="{FONT}" font-size="13" fill="{INK}" stroke-linecap="round" stroke-linejoin="round">\n'
           f"  <title>{esc(title)}</title>\n  <g transform=\"translate({PAD} {PAD})\">\n{body}  </g>\n</svg>\n")
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write(doc)
    FILES.append((name, title, w + 2 * PAD, h + 2 * PAD))


def legend(x, y):
    s = ""
    for i, (kind, lab) in enumerate((("purple", "content only"), ("yellow", "involves labels"), ("orange", "trained attention / norms"),
                                     ("green", "MLP"), ("blue", "no weights"))):
        xx = x + i * 150
        s += rect(xx, y - 7, 18, 14, kind, rx=4, sw=1.5) + text(xx + 24, y, lab, 10, INK, "start")
    return s


# =============================================================================================== parameter counts
def parameter_counts():
    """From the module as the model builds it: LabelledRead(128, 4, 2, 10, 256) and the 128-wide patch marker."""
    m = LabelledRead(128, 4, 2, 10, 256)
    c = {name: p.numel() for name, p in m.named_parameters()}
    read = sum(v for n, v in c.items() if n.startswith("reads.0."))
    norms_in_read = sum(v for n, v in c.items() if n.startswith("reads.0.norm"))
    attn = sum(v for n, v in c.items() if n.startswith("reads.0.attention"))
    layer_norm = sum(v for n, v in c.items() if n.startswith("norms.0."))
    mlp = sum(v for n, v in c.items() if n.startswith("mlps.0."))
    label = c["label.weight"]
    total = sum(c.values())
    return {"read": read, "read_norms": norms_in_read, "attention": attn, "layer_norm": layer_norm, "mlp": mlp, "label": label,
            "per_layer": read + layer_norm + mlp, "total": total, "marker": 128, "total_with_marker": total + 128}


# =============================================================================================== 1. full data flow
def full_dataflow(pc):
    W, H = 884, 1384
    b = text(0, 12, "Image in-context read (LabelledRead.forward): every tensor and operation of one table's forward pass", 15, INK, "start", weight="bold")
    b += text(0, 32, "ModelConfig: width 128, 4 heads, 2 layers, up to 256 sources, top-32; shapes for one table, (photos, patches, width) with the batch axis dropped", 10, MUTED, "start")
    b += legend(0, 50)
    ca, cb, cc = 26, 318, 610          # three columns, with a gutter on the left for the h line
    wa, wb, wc = 262, 262, 274
    gx = 10                            # the gutter
    # ---- inputs
    y = 70
    b += node(ca, y, wa, 44, "purple", "tokens h  (stage input)", "(rows, 256, 128)", "context photos first, then test photos")
    b += node(cb, y, wb, 44, "yellow", "y_context", "(n_ctx,)  label slots 0..9", "the context rows' labels; no test label")
    b += node(cc, y, wc, 44, "yellow", "E = nn.Embedding(10, 128)", "(10, 128)", f"orthogonal init; {pc['label']:,} parameters", trained=True)
    # ---- pick sources
    y = 134
    b += arrow(cb + wb / 2, 114, cb + wb / 2, y, INK, 1.5)
    b += node(cb, y, wb, 46, "blue", "pick_sources(n_ctx)", "picked: (S,),  S = min(n_ctx, 256)", "past 256: a random subset per training step", size=10.5)
    # ---- sources and source labels
    y = 204
    b += arrow(ca + wa / 2, 114, ca + wa / 2, y, INK, 1.5)
    b += text(ca + wa / 2 + 6, 158, "index with picked", 9, MUTED, "start")
    b += node(ca, y, wa, 44, "purple", "sources = h[picked]", "(S, 256, 128)", "the context photos' tokens, label-free")
    b += arrow(cb + wb / 2, 180, cb + wb / 2, y, INK, 1.5)
    b += elbow(cc + wc / 2, 114, cb + wb - 40, y, INK, 1.5, via="v")
    b += node(cb, y, wb, 44, "yellow", "source_labels = E[y_context[picked]]", "(S, 128)", "one label embedding per source photo", size=10.5)
    # the h line down the gutter to the first read
    y0 = 428
    b += L(f'<path d="M{ca},{92} H{gx} V{y0 + 50 - 0} H{ca - 7}" fill="none" stroke="{INK}" stroke-width="1.5"/>') + head(ca, y0 + 50, 0, INK, 8)
    b += L(f'<text transform="translate({gx - 3} {300}) rotate(-90)" font-size="9" fill="{MUTED}" text-anchor="middle">h, the stage input, also feeds the reads directly</text>')
    # ---- retrieval group
    y = 272
    b += rect(ca - 2, y, cc + wc - ca + 2, 140, "blue", rx=10, sw=1.2, dashed=True, fill="none")
    b += text(ca + 8, y + 12, "nearest_photos(h, sources, k = 32): retrieval, no weights, once per table on the stage input", 10.5, P["blue"][1], "start", weight="bold")
    b += arrow(ca + wa / 2, 248, ca + wa / 2, y + 28, INK, 1.5)
    b += node(ca, y + 28, 126, 40, "purple", "m = h.mean(2)", "(rows, 128)")
    b += node(ca + 136, y + 28, 126, 40, "purple", "m_S = sources.mean(2)", "(S, 128)", size=10)
    b += arrow(ca + 63, y + 68, ca + 63, y + 86, INK, 1.5) + arrow(ca + 199, y + 68, ca + 199, y + 86, INK, 1.5)
    b += node(ca, y + 86, wa, 40, "blue", "F.normalize both, then m̂ · m̂_Sᵀ", "sim: (rows, S)  cosine similarity")
    b += arrow(ca + wa, y + 106, cb + 20, y + 106, INK, 1.5)
    b += node(cb + 22, y + 86, 240, 40, "purple", "N = sim.topk(32).indices", "(rows, 32)  the 32 nearest sources", size=10.5)
    b += lines(cc + 6, y + 30, ["only when S > 32 (top_k < len(picked)); with",
                                "32 or fewer sources N = None and every patch",
                                "attends to every source patch instead (the",
                                "plain read: keys and values flattened over S).",
                                "No label is used here. A context photo's cosine",
                                "with itself is 1, so it is among its own nearest.",
                                "N is computed once and reused by both layers."], 9.2, INK, "start", lead=11.5)
    # ---- layer 1 group
    b += rect(ca - 2, y0, cc + wc - ca + 2, 650, "orange", rx=10, sw=1.2, dashed=True, fill="none")
    b += text(cc + wc - 24, y0 + 12, "layer 1: LabelRead (reads[0]), then norms[0] and mlps[0]", 10.5, P["orange"][1], "end", weight="bold")
    b += flame(cc + wc - 12, y0 + 12, 6)
    y = y0 + 28
    b += node(ca, y, wa, 44, "purple", "q = norm_readers(h)", "(rows, 256, 128)", "LayerNorm, its own affine (256 params)", trained=True)
    b += node(cb, y, wb, 44, "purple", "keys = norm_sources(sources)", "(S, 256, 128)", "a second LayerNorm (256 params)", trained=True)
    b += node(cc, y, wc, 44, "yellow", "values = source_labels[:, None].expand", "(S, 256, 128)", "the photo's label, one copy per patch", size=10)
    ga, gb = (ca + wa + cb) / 2, (cb + wb + cc) / 2          # the gaps between the columns
    b += L(f'<path d="M{ca + wa},{226} H{ga} V{y - 10} H{cb + 40} V{y - 7}" fill="none" stroke="{INK}" stroke-width="1.5"/>') + head(cb + 40, y, 90, INK, 8)
    b += text(ga + 5, 262, "sources", 9, MUTED, "start")
    b += L(f'<path d="M{cb + wb / 2 + 40},{248} V{258} H{gb} V{y - 10} H{cc + wc / 2} V{y - 7}" fill="none" stroke="{INK}" stroke-width="1.5"/>') + head(cc + wc / 2, y, 90, INK, 8)
    b += text(gb + 5, 266, "source_labels", 9, MUTED, "start")
    # gather per reader
    y = y0 + 96
    b += arrow(cb + wb / 2, y0 + 72, cb + wb / 2, y, INK, 1.5) + arrow(cc + wc / 2, y0 + 72, cc + wc / 2, y, INK, 1.5)
    b += L(f'<path d="M{cb + 40},{412} V{420} H{cb - 14} V{y + 23} H{cb - 7}" fill="none" stroke="{INK}" stroke-width="1.5"/>') + head(cb, y + 23, 0, INK, 8)
    b += text(cb - 10, y - 10, "N(i)", 9, MUTED, "start")
    b += node(cb, y, wb, 46, "purple", "K_i = keys[N(i)].flatten", "(32 · 256 = 8192, 128)", "per reading photo i: its neighbours' patches")
    b += node(cc, y, wc, 46, "yellow", "V_i = values[N(i)].flatten", "(8192, 128)", "the neighbours' labels, one per key")
    b += node(ca, y, wa, 46, "grey", "chunk loop (self.chunk = 32)", None, ["readers go through the attention 32 photos at a time;", "memory only: the result does not depend on the chunk"], size=10)
    # projections
    y = y0 + 166
    for x, w, lab, shp, sub in ((ca, wa, "Wq · q_i + b_q", "(256, 128)", "q_i: the 256 patches of photo i"), (cb, wb, "Wk · K_i + b_k", "(8192, 128)", None), (cc, wc, "Wv · V_i + b_v", "(8192, 128)", None)):
        b += arrow(x + w / 2, y - 24, x + w / 2, y, INK, 1.5)
        b += node(x, y, w, 44, "orange" if x != cc else "yellow", lab, shp, sub, trained=True)
    # split heads
    y = y0 + 230
    for x, w, shp in ((ca, wa, "(4, 256, 32)"), (cb, wb, "(4, 8192, 32)"), (cc, wc, "(4, 8192, 32)")):
        b += arrow(x + w / 2, y - 20, x + w / 2, y, INK, 1.5)
        b += node(x, y, w, 36, "purple" if x != cc else "yellow", "split into 4 heads of 32", shp, size=10)
    # scores
    y = y0 + 290
    b += arrow(ca + wa / 2, y - 24, ca + wa / 2, y, INK, 1.5)
    b += elbow(cb + wb / 2, y - 24, ca + wa - 30, y, INK, 1.5)
    b += node(ca, y, wa, 44, "purple", "scores = q kᵀ / √32", "(4, 256, 8192) per head", "F.scaled_dot_product_attention; √(head width)")
    b += arrow(ca + wa / 2, y + 44, ca + wa / 2, y + 64, INK, 1.5)
    b += node(ca, y + 64, wa, 44, "purple", "weights = softmax over the 8192 keys", "(4, 256, 8192)", "content decides where to look: no label so far", size=10)
    # weighted sum
    y = y0 + 424
    b += arrow(ca + wa / 2, y - 26, ca + wa / 2, y, INK, 1.5)
    b += elbow(cc + wc / 2, y0 + 266, ca + wa - 30, y, INK, 1.5)
    b += text(cc + wc / 2 - 6, y - 50, "V flows across to the weighted sum", 9, MUTED, "end")
    b += node(ca, y, wa, 44, "yellow", "weights · v", "(4, 256, 32) → merge heads (256, 128)", "a soft vote: each patch's mix of its neighbours' labels", size=10)
    b += arrow(ca + wa / 2, y + 44, ca + wa / 2, y + 64, INK, 1.5)
    b += node(ca, y + 64, wa, 40, "yellow", "r_i = Wo · (…) + b_o", "(256, 128)", None, trained=True)
    # residual and MLP, in the middle column
    yr = y0 + 424
    b += elbow(ca + wa, y + 84, cb + wb / 2, yr + 4, INK, 1.5, via="h")
    b += node(cb, yr + 4, wb, 44, "yellow", "h ← h + r", "(rows, 256, 128)", "content + what it read")
    b += arrow(cb + wb / 2, yr + 48, cb + wb / 2, yr + 68, INK, 1.5)
    b += node(cb, yr + 68, wb, 44, "green", "mlps[0](norms[0](h))", "128 → 256 → GELU → 128", f"LayerNorm {pc['layer_norm']} + MLP {pc['mlp']:,} params", trained=True)
    b += arrow(cb + wb / 2, yr + 112, cb + wb / 2, yr + 132, INK, 1.5)
    b += node(cb, yr + 132, wb, 44, "yellow", "h ← h + MLP(LN h)", "(rows, 256, 128)", "every row's photo, context and test alike")
    b += lines(cc + 6, yr + 10, ["parameters of one LabelRead:", f"  norm_readers + norm_sources   {pc['read_norms']}",
                                f"  Wq, Wk, Wv, Wo (128×128 + 128 each)   {pc['attention']:,}", f"  = {pc['read']:,}",
                                "the attention has no label of its own: labels", "enter only through the values.",
                                "", "a test photo is never a source: it cannot read", "its own label. A context photo is among its", "own sources and reads it (context is given).",
                                "", "the plain read (patch_read = \"all\") is the same", "with N dropped: keys and values flattened over", "every source photo, (S · 256, 128)."],
                9.2, INK, "start", lead=11.5)
    # ---- layer 2
    y = 1096
    b += elbow(cb + wb / 2, yr + 176, cb + wb / 2, y, INK, 1.5)
    b += rect(ca - 2, y, cc + wc - ca + 2, 74, "orange", rx=10, sw=1.2, dashed=True, fill="none")
    b += flame(cc + wc - 12, y + 12, 6)
    b += text(ca + 8, y + 12, "layer 2: reads[1], norms[1], mlps[1] — identical in structure, its own weights", 10.5, P["orange"][1], "start", weight="bold")
    b += lines(ca + 8, y + 32, ["queries = norm_readers of layer 2 applied to layer 1's output h; keys = norm_sources of layer 2 applied to sources, and values = source_labels, both from the STAGE INPUT:",
                                "the same sources, the same source_labels and the same N as layer 1 (model.py: sources and nearest are computed once, before the loop over layers).",
                                f"Same parameter count as layer 1: {pc['per_layer']:,}.  Output h: (rows, 256, 128)."], 9.4, INK, "start", lead=12)
    # ---- output
    y = 1192
    b += arrow(cb + wb / 2, 1170, cb + wb / 2, y, INK, 1.5)
    b += node(ca, y, wa, 44, "yellow", "stage output h", "(rows, 256, 128)", "LabelledRead.forward returns here")
    b += elbow(cb + wb / 2, 1170, ca + wa / 2, y, INK, 1.5)
    b += arrow(ca + wa, y + 22, cb - 2, y + 22, INK, 1.5)
    b += node(cb, y, wb, 44, "yellow", "+ patch_marker m", "m: (128,)  broadcast over every patch", "TableImageICL.photo_tokens; 128 params", trained=True)
    b += arrow(cb + wb, y + 22, cc - 2, y + 22, INK, 1.5)
    b += node(cc, y, wc, 44, "grey", "to TabICL's row transformer", "cat([4 CLS | cells | 256 patches])", "the row stage sees patches next to cells")
    # ---- parameter summary
    y = 1262
    b += rect(ca - 2, y, cc + wc - ca + 2, 112, "grey", rx=10, sw=1.2, fill="#F9FAFB", stroke=P["grey"][0])
    b += text(ca + 8, y + 14, "all parameters of the stage (counted from the module as built by the model)", 10.5, INK, "start", weight="bold")
    b += lines(ca + 8, y + 34, [f"per layer: LabelRead {pc['read']:,} (two LayerNorms {pc['read_norms']}, four 128×128 linears with bias {pc['attention']:,})  +  LayerNorm {pc['layer_norm']}  +  MLP 128→256→128 {pc['mlp']:,}   =   {pc['per_layer']:,}",
                                f"two layers {2 * pc['per_layer']:,}  +  label table E 10×128 = {pc['label']:,}   =   {pc['total']:,} in LabelledRead;  plus the patch marker m {pc['marker']} = {pc['total_with_marker']:,}",
                                "all trained from scratch at lr 1e-4; nothing in this stage is pretrained. The label table is the only place a label becomes a vector:",
                                "every yellow tensor above is a sum of rows of E (and, after the MLP, a function of such sums)."], 9.4, INK, "start", lead=13)
    svg_file("full-dataflow", "Image in-context read: the full data flow of LabelledRead.forward for one table, every tensor with its shape, every operation, with the tensors that involve labels in yellow and the content-only ones in purple", W, H, b)


# =============================================================================================== 2. worked examples from the real code
A, B_, C = [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]     # three zero-mean, unit-variance patterns
PATTERN_NAME = {tuple(A): "A", tuple(B_): "B", tuple(C): "C"}
SMALL = dict(width=4, heads=1, layers=1, n_slots=3, max_sources=8, read="top_k", top_k=2)


def small_module():
    """LabelledRead with hand-set weights: identity Wq/Wk/Wv/Wo with zero bias, LayerNorms at their default affine
    (weight 1, bias 0), E = one-hot rows, and the MLP's second linear zeroed so the example stays readable."""
    m = LabelledRead(**SMALL)
    with torch.no_grad():
        att = m.reads[0].attention
        for lin in (att.q, att.k, att.v, att.o):
            lin.weight.copy_(torch.eye(4))
            lin.bias.zero_()
        m.label.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]))
        m.mlps[0][2].weight.zero_()
        m.mlps[0][2].bias.zero_()
    return m.eval()


def run_example(name, photos, reader, story):
    """photos: list of (name, label or None, [patch, patch]); context photos first. reader: index of the photo whose read
    is drawn. Every intermediate is recomputed from the module's parts and the final output is checked against the
    module's own forward."""
    m = small_module()
    tokens = torch.tensor([[p[2] for p in photos]])                                   # (1, photos, 2, 4)
    y_context = torch.tensor([[p[1] for p in photos if p[1] is not None]])            # (1, n_ctx)
    with torch.no_grad():
        picked = m.pick_sources(y_context.shape[1], tokens.device)
        sources = tokens[:, picked]
        source_labels = m.label(y_context[:, picked])
        means = tokens.mean(2)[0]
        means_s = sources.mean(2)[0]
        sim = torch.nn.functional.normalize(means, dim=-1) @ torch.nn.functional.normalize(means_s, dim=-1).T
        nearest = nearest_photos(tokens, sources, m.top_k)
        assert torch.equal(nearest[0], sim.topk(m.top_k, dim=-1).indices)
        read = m.reads[0]
        queries, keys = read.norm_readers(tokens)[0], read.norm_sources(sources)[0]
        i = reader
        nb = nearest[0, i].tolist()
        K = keys[nb].flatten(0, 1)                                                    # (k · patches, 4)
        V = source_labels[0, nb][:, None].expand(-1, 2, -1).flatten(0, 1)
        att = read.attention
        Q = att.q(queries[i])
        logits = (Q @ att.k(K).T) / math.sqrt(4)
        weights = logits.softmax(-1)
        r = att.o(weights @ att.v(V))
        r_module = read(tokens, sources, source_labels, nearest)[0, i]
        assert torch.allclose(r, r_module, atol=1e-6), "hand recomputation differs from LabelRead"
        out = m(tokens, y_context)
        assert torch.allclose(out[0, i], tokens[0, i] + r, atol=1e-6), "the MLP was meant to be zero"
    labels = [p[1] for p in photos]
    key_labels = [labels[j] for j in nb for _ in range(2)]
    mix = [[round(float(sum(w for w, lab in zip(row.tolist(), key_labels) if lab == c)), 4) for c in range(3)] for row in weights]
    avg = [round(sum(m[c] for m in mix) / 2, 4) for c in range(3)]
    cos_by_name = {photos[j][0]: round(float(sim[i, j]), 2) for j in range(len(picked))}
    story = [line.format(reader=photos[i][0], n0=photos[nb[0]][0], n1=photos[nb[1]][0], cos0=cos_by_name[photos[nb[0]][0]], cos1=cos_by_name[photos[nb[1]][0]],
                         other=next(photos[j][0] for j in range(len(picked)) if j not in nb), cos_other=cos_by_name[next(photos[j][0] for j in range(len(picked)) if j not in nb)],
                         p1e0=mix[0][0], p1e1=mix[0][1], p2e0=mix[1][0], p2e1=mix[1][1], avg0=avg[0], avg1=avg[1]) for line in story]
    rd = lambda t: [[round(v, 4) for v in row] for row in t.tolist()] if t.dim() == 2 else [round(v, 4) for v in t.tolist()]
    return {"name": name, "story": story, "reader": i, "reader_name": photos[i][0],
            "photos": [{"name": p[0], "label": p[1], "patches": [[PATTERN_NAME[tuple(x)], x] for x in p[2]], "mean": rd(means[j])} for j, p in enumerate(photos)],
            "picked": picked.tolist(), "cos": rd(sim[i]), "cos_all": rd(sim), "N": nb, "N_names": [photos[j][0] for j in nb],
            "keys": [f"{photos[j][0]}·p{p + 1}" for j in nb for p in range(2)], "key_labels": key_labels,
            "ln_query": rd(queries[i]), "logits": rd(logits), "weights": rd(weights), "r": rd(r), "r_mix": mix, "r_mix_photo_average": avg, "out": rd(out[0, i]),
            "checked_against_forward": True}


def examples():
    ctx = [("c1", 1, [A, B_]), ("c2", 0, [B_, B_]), ("c3", 1, [A, C])]
    ex1 = run_example("example-1", ctx + [("t", None, [A, C])], 3,
                      ["{reader}'s mean token points where {n0}'s and {n1}'s do: cos {cos0:.2f} and {cos1:.2f}, {cos_other:.2f} for {other}, so N({reader}) = {{{n0}, {n1}}}.",
                       "Each test patch softmaxes over the 4 neighbour patches by content; both neighbours are labelled 1, so every value is e1.",
                       "The read is {p1e1:.2f} e1 on patch 1 and {p2e1:.2f} e1 on patch 2, whatever the weights: a unanimous vote. {reader}'s tokens leave as z + e1."])
    ex2 = run_example("example-2", [("c1", 1, [A, C]), ("c2", 0, [B_, B_]), ("c3", 1, [C, C]), ("t", None, [A, B_])], 3,
                      ["{reader} has one patch like c1 (label 1) and one like c2 (label 0): cos {cos1:.2f} with c1 and {cos0:.2f} with c2, {cos_other:.2f} with {other}, so N({reader}) = {{{n0}, {n1}}}.",
                       "Patch 1 attends mostly to c1's matching patch, patch 2 mostly to c2's: their reads are {p1e1:.2f} e1 + {p1e0:.2f} e0 and {p2e0:.2f} e0 + {p2e1:.2f} e1.",
                       "Averaged over the photo the read is {avg0:.2f} e0 + {avg1:.2f} e1: the vote is soft, and it is per patch."])
    ex3 = run_example("example-3", ctx + [("t", None, [A, C])], 1,
                      ["{reader} is a context photo (label 0). Its own mean token has cosine {cos0:.2f} with itself and {cos1:.2f} with {n1}, so N({reader}) = {{{n0}, {n1}}}.",
                       "Its patches attend to their own two copies (label 0) and to {n1}'s matching patch (label 1): r = {p1e0:.2f} e0 + {p1e1:.2f} e1, mostly e0.",
                       "Allowed: context labels are inputs and no loss is taken on context rows. y_test is not an argument of the stage."])
    return [ex1, ex2, ex3]


def fmt(v, d=2):
    return f"{v:+.{d}f}" if d else (f"{int(round(v)):+d}" if abs(v - round(v)) < 1e-9 else f"{v:+.2f}")


def vec(x, y, values, w=84, h=20, kind="purple", digits=0, size=9.5):
    s = rect(x, y, w, h, kind, rx=4, sw=1.2)
    return s + text(x + w / 2, y + h / 2, "  ".join(fmt(v, digits) for v in values), size, P[kind][1], font=MONO)


def matrix(x, y, col_names, row_names, rows, cw=58, rh=20, highlight=(), kind="purple", digits=2, header_kind="grey", left=92):
    """A table of numbers with a header row and a row-label column; highlight = set of column indices to colour."""
    s = ""
    s += rect(x, y, left + len(col_names) * cw, rh, header_kind, rx=4, sw=1, fill="#F3F4F6", stroke=P["grey"][0])
    for j, c in enumerate(col_names):
        s += text(x + left + j * cw + cw / 2, y + rh / 2, c, 9.5, INK, weight="bold")
    for i, (rn, row) in enumerate(zip(row_names, rows)):
        yy = y + rh * (i + 1)
        s += text(x + left - 6, yy + rh / 2, rn, 9.5, MUTED, "end")
        for j, v in enumerate(row):
            k = kind if j in highlight else None
            if k:
                s += rect(x + left + j * cw + 1, yy + 1, cw - 2, rh - 2, k, rx=3, sw=1)
            s += text(x + left + j * cw + cw / 2, yy + rh / 2, v if isinstance(v, str) else f"{v:.{digits}f}", 9.5, INK, font=MONO)
    s += rect(x, y, left + len(col_names) * cw, rh * (len(rows) + 1), "grey", rx=4, sw=1, fill="none", stroke=P["grey"][1])
    return s


def example_panel(ex, number):
    W, H = 884, 584
    ph = ex["photos"]
    i = ex["reader"]
    b = text(0, 12, f"Example {number}: {ex['reader_name']} reads — " + {1: "a test photo whose neighbours agree", 2: "a test photo between a 1-photo and a 0-photo", 3: "a context photo reads its own label"}[number], 15, INK, "start", weight="bold")
    b += text(0, 32, "LabelledRead(width 4, heads 1, layers 1, n_slots 3, max_sources 8, top_k 2); Wq = Wk = Wv = Wo = I, b = 0; LayerNorms weight 1, bias 0; E = one-hot rows e0, e1, e2; MLP zeroed. Numbers: examples.json", 9, MUTED, "start")
    # ---- the photos
    x0, y0 = 0, 56
    b += text(x0, y0, "the table: 2-patch photos, 4-wide tokens", 11, INK, "start", weight="bold")
    b += text(x0 + 88, y0 + 18, "patch 1", 9, MUTED) + text(x0 + 170, y0 + 18, "patch 2", 9, MUTED) + text(x0 + 268, y0 + 18, "mean token", 9, MUTED)
    for j, p in enumerate(ph):
        yy = y0 + 30 + j * 46
        if j == i:
            b += rect(x0 - 4, yy - 6, 326, 40, "orange", rx=6, sw=1.5, fill="none")
            b += text(x0 + 318, yy + 31, "reader", 8.5, P["orange"][1], "end")
        b += text(x0 + 10, yy + 10, p["name"], 11, INK, weight="bold")
        b += chip(x0 + 22, yy + 2, 22, 16, "yellow" if p["label"] is not None else "grey", "?" if p["label"] is None else str(p["label"]))
        for k, (pname, vals) in enumerate(p["patches"]):
            b += vec(x0 + 50 + k * 82, yy, vals, 76, 20, "blue", 0, 9)
            b += text(x0 + 50 + k * 82 + 38, yy + 27, f"pattern {pname}", 8, MUTED)
        b += vec(x0 + 216, yy, p["mean"], 104, 20, "purple", 1, 8.5)
    yy = y0 + 30 + len(ph) * 46 + 4
    pat = ", ".join(f"{n} = ({' '.join(fmt(v, 0) for v in v4)})" for v4, n in ((A, "A"), (B_, "B"), (C, "C")))
    b += lines(x0, yy, [pat + ":", "zero mean and unit variance, so LayerNorm leaves them as they are;",
                        "A·B = A·C = B·C = 0 and X·X = 4. Labels are the context photos' slots;",
                        "the test photo t has none: it is not an input of the stage."], 8.8, MUTED, "start", lead=11)
    # ---- retrieval
    x1, y1 = 350, 56
    src = [p["name"] for p in ph if p["label"] is not None]
    b += text(x1, y1, f"retrieval: cosine of {ex['reader_name']}'s mean token with each source", 11, INK, "start", weight="bold")
    hi = set(ex["N"])
    b += matrix(x1, y1 + 12, src, [f"cos(m_{ex['reader_name']}, m_c)"], [ex["cos"]], cw=64, highlight=hi, left=96)
    b += text(x1, y1 + 66, f"top-2  →  N({ex['reader_name']}) = {{{', '.join(ex['N_names'])}}}  (k = 2 here, 32 in the model)", 10, INK, "start")
    # ---- attention
    y2 = y1 + 90
    b += text(x1, y2, "the read: one query patch at a time over the neighbours' 4 patches (keys)", 11, INK, "start", weight="bold")
    rows, names = [], []
    for pi in range(2):
        rows.append(ex["logits"][pi]); names.append(f"p{pi + 1}: q·k / √4")
        rows.append(ex["weights"][pi]); names.append(f"p{pi + 1}: softmax")
    labs = [f"e{l}" for l in ex["key_labels"]]
    rows.append(labs); names.append("value = label")
    b += matrix(x1, y2 + 12, ex["keys"], names, rows, cw=64, left=96)
    b += text(x1, y2 + 12 + 20 * 6 + 12, "keys and queries are LayerNormed content tokens; the values are the key photo's label embedding, repeated for its patches", 8.8, MUTED, "start")
    # ---- what is read back
    y3 = y2 + 12 + 20 * 6 + 34
    b += text(x1, y3, "what flows back and what the photo carries out", 11, INK, "start", weight="bold")
    for pi in range(2):
        yy = y3 + 14 + pi * 48
        mix = ex["r_mix"][pi]
        terms = " + ".join(f"{w:.2f} e{c}" for c, w in enumerate(mix) if w > 0.005)
        b += text(x1, yy + 10, f"patch {pi + 1}:  r = {terms}", 10, INK, "start", font=MONO)
        b += text(x1 + 54, yy + 30, "r =", 9, MUTED, "end") + vec(x1 + 60, yy + 20, ex["r"][pi], 136, 20, "yellow", 2, 9)
        b += text(x1 + 300, yy + 30, "out = z + r =", 9, MUTED, "end") + vec(x1 + 306, yy + 20, ex["out"][pi], 136, 20, "yellow", 2, 9)
    avg = ex["r_mix_photo_average"]
    b += text(x1, y3 + 14 + 96 + 6, "photo average:  " + " + ".join(f"{w:.2f} e{c}" for c, w in enumerate(avg) if w > 0.005) + "    (then the MLP, zeroed here, and + m before the row transformer)", 9.5, INK, "start", font=MONO)
    # ---- story
    ys = 496
    b += rect(0, ys, W, 72, "yellow", rx=8, sw=1.5)
    b += text(10, ys + 14, "what happened", 10.5, P["yellow"][1], "start", weight="bold")
    b += lines(10, ys + 32, ex["story"], 10, INK, "start", lead=14)
    if number == 3:
        b += rect(0, 336, 322, 130, "grey", rx=8, sw=1.5, fill="#F9FAFB")
        b += text(10, 350, "the test photo's own label never enters", 10, INK, "start", weight="bold")
        b += lines(10, 368, ["the stage is forward(tokens, y_context): y_test is not an",
                             "argument, so nothing a test photo carries depends on its label.",
                             "A test photo is not a source either: no patch of any photo",
                             "attends to a test photo's patches.", "",
                             "context rows: their labels are inputs, and the loss is taken",
                             "on the test rows only, so a context photo reading its own",
                             "label is allowed and expected."], 9, INK, "start", lead=12)
    titles = {1: "Example 1 from the real code: a test photo whose two nearest context photos both carry label 1 reads back exactly e1 on every patch",
              2: "Example 2 from the real code: a test photo between a label-1 and a label-0 photo reads back a mixed vector, about half e0 and half e1 over the photo",
              3: "Example 3 from the real code: a context photo with label 0 retrieves itself and reads back mostly e0; the test photo's label is never an input"}
    svg_file(ex["name"], titles[number], W, H, b)


# =============================================================================================== preview sheet
def preview():
    def card(n, t, w, h):
        return f'<figure><img src="{n}.svg" alt="{esc(t)}" width="{w}" height="{h}"><figcaption><code>{n}.svg</code><span>{esc(t)}</span></figcaption></figure>'
    left = card(*FILES[0])
    right = "\n".join(card(*f) for f in FILES[1:])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image in-context read: explainer</title>
<style>
  body {{ margin: 0; padding: 24px; font-family: {FONT}; color: {INK}; background: #fff; width: 1834px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p {{ margin: 0 0 14px; color: {MUTED}; font-size: 13px; }}
  .cols {{ display: grid; grid-template-columns: 900px 900px; gap: 18px; align-items: start; }}
  .col {{ display: flex; flex-direction: column; gap: 18px; }}
  figure {{ margin: 0; border: 1px solid #E5E7EB; border-radius: 10px; overflow: hidden; background: #fff; }}
  figure img {{ display: block; background: repeating-conic-gradient(#F3F4F6 0 25%, #fff 0 50%) 0 0 / 16px 16px; }}
  figcaption {{ padding: 8px 10px; border-top: 1px solid #E5E7EB; font-size: 12px; }}
  figcaption code {{ display: block; font-weight: bold; color: {P["blue"][1]}; }}
  figcaption span {{ color: {MUTED}; }}
</style></head><body>
<h1>The image in-context read, in detail</h1>
<p>{len(FILES)} SVG files in <code>docs/figures/image-icl/</code>: the full data flow of <code>LabelledRead.forward</code> and three worked examples whose numbers come from running the module (see <code>examples.json</code>).</p>
<div class="cols"><div class="col">{left}</div><div class="col">
{right}
</div></div>
</body></html>
"""
    with open(os.path.join(OUT, "preview.html"), "w") as f:
        f.write(page)
    return 24 * 2 + 80 + 3 * (600 + 56 + 18)


def preview_png(height):
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        print("Chrome not found; preview.png not rendered")
        return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={1834 + 48},{height}",
                    f"--screenshot={os.path.join(OUT, 'preview.png')}", "file://" + os.path.join(OUT, "preview.html")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    pc = parameter_counts()
    full_dataflow(pc)
    exs = examples()
    for n, ex in enumerate(exs, 1):
        example_panel(ex, n)
    with open(os.path.join(OUT, "examples.json"), "w") as f:
        json.dump({"module": SMALL, "weights": "Wq = Wk = Wv = Wo = identity, biases 0; LayerNorms weight 1, bias 0; E = one-hot rows; MLP second linear zeroed",
                   "parameter_counts_full_size": pc, "examples": exs}, f, indent=1)
    height = preview()
    preview_png(height)
    print(len(FILES), "figures + examples.json + preview.html written to", OUT)
