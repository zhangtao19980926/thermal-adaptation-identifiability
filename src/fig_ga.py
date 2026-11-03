# -*- coding: utf-8 -*-
"""Graphical abstract for Building & Environment submission.
Spec: 1328 x 531 px at 300 dpi (4.427 x 1.77 in), Arial, statement text >= 12 pt, no title/logo.
Graphic-led: a fixed window vs a dual-time-constant memory kernel vs a climate-conditioned window."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
import os

plt.rcParams["font.family"] = "Arial"
OUT = r"D:\thermal-adaptation-series\p1\outputs\figures"

BLUE="#2471a3"; LBLUE="#d6e6f2"; GREEN="#1e8449"; LGREEN="#d8efe0"
ORANGE="#e67e22"; LOR="#fbe6d2"; RED="#c0392b"; LRED="#f7dcd9"
GREY="#7f8c8d"; LGREY="#eceff1"; DGREY="#34495e"

W_in, H_in, DPI = 4.427, 1.77, 300
fig = plt.figure(figsize=(W_in, H_in), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.1, rad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=%s" % rad,
                                fc=fc, ec=ec, lw=lw, mutation_aspect=W_in / H_in))

def txt(x, y, s, size=12, color="black", weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, transform=ax.transAxes, linespacing=1.1)

def arrow(x0, x1, y, color=DGREY):
    ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.010, head_width=0.045, head_length=0.02,
                            length_includes_head=True, fc=color, ec=color, transform=ax.transAxes))

yt, yb = 0.965, 0.34

# ---- Left: fixed window ----
box(0.015, yb, 0.22, yt - yb, LGREY, GREY)
txt(0.125, 0.905, "Fixed", 14, DGREY, "bold")
txt(0.125, 0.72, "7\u201330 d", 13, "black")
txt(0.125, 0.60, "\u03bb = 0.8", 13, "black")
txt(0.125, 0.46, "one size", 12, GREY, style="italic")
txt(0.125, 0.385, "fits all", 12, GREY, style="italic")
arrow(0.242, 0.30, 0.65)

# ---- Middle: dual memory kernel with decay curves ----
box(0.31, yb, 0.32, yt - yb, LBLUE, BLUE, lw=1.3)
txt(0.47, 0.905, "Memory kernel", 13, BLUE, "bold")
inset = fig.add_axes([0.345, 0.50, 0.25, 0.315]); inset.set_facecolor("none")
tt = np.linspace(0, 10, 300)
inset.plot(tt, np.exp(-tt / 0.55), color=ORANGE, lw=2.0)
inset.plot(tt, np.exp(-tt / 4.2), color=BLUE, lw=2.0)
inset.set_xlim(0, 10); inset.set_ylim(0, 1.05)
inset.set_xticks([]); inset.set_yticks([])
for s in ("top", "right"):
    inset.spines[s].set_visible(False)
inset.spines["left"].set_color("#4d4d4d"); inset.spines["bottom"].set_color("#4d4d4d")
inset.text(1.15, 0.26, "fast \u2248 26 min", color=ORANGE, fontsize=11, fontweight="bold")
inset.text(3.05, 0.60, "slow \u2248 21 d", color=BLUE, fontsize=11, fontweight="bold")
txt(0.47, 0.435, "needs seasonal", 12, DGREY, style="italic")
txt(0.47, 0.368, "excitation", 12, DGREY, style="italic")
arrow(0.638, 0.65, 0.65)

# ---- Right: climate-conditioned ----
box(0.71, yb, 0.275, yt - yb, LGREEN, GREEN, lw=1.3)
txt(0.8475, 0.91, "By climate", 12.5, GREEN, "bold")
txt(0.8475, 0.845, "& operation", 12.5, GREEN, "bold")
rows = [
    ("NV: 15\u201321 d", LGREEN, GREEN, 0.735),
    ("AC / cold: 30 d", LBLUE, BLUE, 0.645),
    ("Heated: none", LOR, ORANGE, 0.555),
    ("Tropical: \u2014", LRED, RED, 0.465),
]
for s, fc, ec, y in rows:
    box(0.725, y - 0.04, 0.245, 0.078, fc, ec, lw=0.9, rad=0.012)
    txt(0.8475, y, s, 12, ec, "bold")
txt(0.8475, 0.382, "\u2014  not identifiable", 10.5, GREY, style="italic")

# ---- bottom take-home band ----
box(0.015, 0.03, 0.97, 0.27, "#f4f7fa", DGREY, lw=1.0)
txt(0.5, 0.165, "One fixed window does not fit every climate.", 13.5, DGREY, "bold")

for ext in ("png", "pdf"):
    fp = os.path.join(OUT, "GA_graphical_abstract." + ext)
    fig.savefig(fp, dpi=DPI if ext == "png" else None)
print("saved", fp)
from PIL import Image
im = Image.open(os.path.join(OUT, "GA_graphical_abstract.png"))
print("PNG size px:", im.size)
