"""Draws docs/figures/flops-vs-baseline.{png,pdf,svg} from the counts of scripts/flops.py: python3 docs/figures/make_flops_figure.py docs/figures/flops.json docs/figures/flops-vs-baseline"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [r for r in json.load(open(sys.argv[1])) if r.get("ours_model") and r.get("ref_tabpfn")]
out = sys.argv[2]
x = np.array([r["context"] for r in rows]); ours = np.array([r["ours_total"] for r in rows]) / 1e12; ref = np.array([r["ref_total"] for r in rows]) / 1e12
od = np.array([r["ours_dino4"] for r in rows]) / 1e12; om = np.array([r["ours_model"] for r in rows]) / 1e12
rd = np.array([r["ref_dino12"] for r in rows]) / 1e12; rt = np.array([r["ref_tabpfn"] for r in rows]) / 1e12
BLUE, ORANGE, INK, MUTED = "#2563EB", "#EA580C", "#1F2937", "#6B7280"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.fill_between(x, ours, ref, color=ORANGE, alpha=0.07, lw=0)
ax.plot(x, ref, "s-", color=ORANGE, lw=2.4, ms=7, label="MulTaBench baseline: all 12 DINOv2 blocks, PCA, TabPFN v2 with 8 estimators")
ax.plot(x, rd, "s--", color=ORANGE, lw=1, ms=4, alpha=0.6, label="   of which DINOv2")
ax.plot(x, ours, "o-", color=BLUE, lw=2.4, ms=7, label="ours: DINOv2 blocks 1 to 4, one pass of our model")
ax.plot(x, od, "o--", color=BLUE, lw=1, ms=4, alpha=0.6, label="   of which DINOv2")
for xi, a, b in zip(x, ours, ref):
    ax.annotate(f"{b / a:.1f}x", (xi, np.sqrt(a * b)), xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=10, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(x[0] / 1.6, x[-1] * 1.35)
ax.set_xticks(x); ax.set_xticklabels([f"{v:,}" for v in x]); ax.minorticks_off()
ax.set_xlabel("context rows (plus 10 % test rows)"); ax.set_ylabel("TFLOPs per table, photos included")
fig.text(0.02, 0.965, "Compute to predict one table with a photo per row", fontsize=12.5, color=INK, va="top")
fig.text(0.02, 0.915, "forward FLOPs counted by torch (matrix products and convolutions); 10 columns, 3 classes; label = baseline over ours", fontsize=9, color=MUTED, va="top")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=0.2, which="major"); ax.legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
print("saved", out)
for xi, a, b, c, d, e, f in zip(x, ours, ref, od, om, rd, rt):
    print(f"{xi:>5}: ours {a:.2f} TF (dino {c:.2f} + model {d:.2f})  baseline {b:.2f} TF (dino {e:.2f} + tabpfn {f:.2f})  ratio {b / a:.1f}")
