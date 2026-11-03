# -*- coding: utf-8 -*-
"""Figure 1 (redesigned for print column width, 7 in): a unified, card-based conceptual figure.
(a) dual-time-constant memory kernel signal flow with orthogonal connectors (spans 3 cols);
(b) four public datasets as compact horizontal cards (spans 2 cols);
(c) five-step evidence pipeline (1 col).
All text is real Unicode (no mathtext); every glyph is vector / editable after SVG->OOXML.
Long descriptions are wrapped to fit the 7-in full-column width; detail lives in the caption.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import apply_style, save, export_panels, C

apply_style()
fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[0.96, 1.04], hspace=0.18, wspace=0.22)


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, rs=1.4, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%.2f" % rs,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def _tint(hx, t=0.86):
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return "#%02x%02x%02x" % (int(r + (255 - r) * t), int(g + (255 - g) * t),
                              int(b + (255 - b) * t))


def txt(ax, x, y, s, fs, tc="#1a1a1a", ha="center", va="center", bold=False, ital=False,
        style=None, z=3):
    ax.text(x, y, s, fontsize=fs, color=tc, ha=ha, va=va,
            fontweight="bold" if bold else "normal", fontstyle="italic" if ital else "normal",
            zorder=z)


def harrow(ax, x1, x2, y, color="#5a6570", lw=1.6, ms=13):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0))


def bus(ax, xs, ys, color="#8a949e", lw=1.4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="round")


# ================= (a) signal flow, spans all three columns =================
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
txt(ax, 0.4, 99.5, "(a) Dual-time-constant thermal-adaptation memory kernel",
    11.5, C["dgrey"], ha="left", va="top", bold=True)

ym = 52
# excitation card
rbox(ax, 1, 39, 14, 26, "#eaf2f8", C["blue"])
txt(ax, 8, 58, "Outdoor", 8.6, C["blue"], bold=True)
txt(ax, 8, 50, "ambient T(t)", 7.7, C["blue"], bold=True)

# three parallel filter cards (x 18..40)
fx, fw = 18, 22
fh, fgap = 20, 3.5
fys = [65.5, 42, 18.5]          # top / middle / bottom (h=20 -> centres 75.5 / 52 / 28.5)
fcentres = [75.5, 52, 28.5]
filters = [
    ("FAST filter", "\u03c4_fast 20\u201360 min", "#fdedec", C["red"]),
    ("INSTANT", "b0 \u00b7 no memory", "#f4f6f7", C["grey"]),
    ("SLOW filter", "\u03c4_s 2\u20134 weeks", "#e9f7ef", C["green"]),
]
for fy, (head, tau, fc, ec) in zip(fys, filters):
    rbox(ax, fx, fy, fw, fh, fc, ec)
    txt(ax, fx + fw / 2, fy + 14.6, head, 8.0, ec, bold=True)
    txt(ax, fx + fw / 2, fy + 7.0, tau, 7.6, "#1a1a1a")

# summing junction
rbox(ax, 44, 39, 11, 26, "#fef9e7", C["gold"])
txt(ax, 49.5, 57, "\u03a3", 12.5, C["gold"], bold=True)
txt(ax, 49.5, 48, "weighted", 7.2, C["dgrey"])

# ordinal logistic
rbox(ax, 59, 39, 16, 26, "#f5eef8", C["purple"])
txt(ax, 67, 57, "ordinal", 8.2, C["purple"], bold=True)
txt(ax, 67, 51, "logistic", 7.8, "#1a1a1a")
txt(ax, 67, 45, "+ Tn, i", 7.8, "#1a1a1a")

# observed vote
rbox(ax, 80, 39, 15, 26, "#eaf2f8", C["blue"])
txt(ax, 87.5, 55, "TSV / vote", 8.6, C["blue"], bold=True)
txt(ax, 87.5, 47, "preference", 7.6, C["blue"])

# orthogonal split / merge buses (no diagonal lines)
bus(ax, [15, 17], [ym, ym])
bus(ax, [17, 17], [fcentres[2], fcentres[0]])
for yc in fcentres:
    bus(ax, [17, fx], [yc, yc])
for yc in fcentres:
    bus(ax, [fx + fw, 42], [yc, yc])
bus(ax, [42, 42], [fcentres[2], fcentres[0]])
bus(ax, [42, 44], [ym, ym])
harrow(ax, 55, 59, ym)
harrow(ax, 75, 80, ym)

# identifiability note bar (two wrapped lines)
rbox(ax, 1, 3, 98, 11.5, "#f7f9fb", "#c7d0d9", lw=1.0, rs=1.6, z=1.5)
txt(ax, 3, 8.8, "Identifiability \u2014 does the excitation carry enough spectral power in each\n"
               "band to recover that band\u2019s time constant?",
    7.6, C["dgrey"], ha="left", ital=True)

# ================= (b) four datasets, spans two columns =================
axb = fig.add_subplot(gs[1, 0:2]); axb.axis("off"); axb.set_xlim(0, 100); axb.set_ylim(0, 100)
txt(axb, 0.4, 99.5, "(b) Four public datasets", 11.5, C["dgrey"],
    ha="left", va="top", bold=True)

datasets = [
    ("DORN \u2014 tropical wearable (Singapore)",
     "20 occ.; minute-scale skin/air T + votes;\n~2 \u00b0C seasonal range \u2192 fast-\u03c4, weak slow band",
     C["red"]),
    ("In-Gauge \u2014 classrooms, Melbourne",
     "16 classrooms, 169 d; 5-min indoor/outdoor;\nstrong seasonal band \u2192 slow-\u03c4, counterfactual",
     C["teal"]),
    ("Chinese Thermal Comfort Dataset (CTCD)",
     "23,461 votes; five climate zones, blocked folds;\n1/7/15/30-d outdoor memory \u2192 window",
     C["orange"]),
    ("ASHRAE Global Comfort Database II",
     "72,503 votes; temperate/tropical/arid/continental;\ninstant + monthly outdoor \u2192 global validation",
     C["blue"]),
]
dtop, dh, dgap = 89, 17, 4.7
for i, (name, desc, col) in enumerate(datasets):
    y = dtop - dh - i * (dh + dgap)
    rbox(axb, 1, y, 98, dh, _tint(col), col, lw=1.2, rs=1.3)
    axb.add_patch(FancyBboxPatch((1, y), 2.2, dh, boxstyle="round,pad=0,rounding_size=1.3",
                                 facecolor=col, edgecolor=col, linewidth=0, zorder=2.5))
    txt(axb, 5.5, y + 12.6, name, 8.4, col, ha="left", bold=True)
    txt(axb, 5.5, y + 5.4, desc, 6.9, "#333333", ha="left")

# ================= (c) five-step evidence pipeline, one column =================
axc = fig.add_subplot(gs[1, 2]); axc.axis("off"); axc.set_xlim(0, 100); axc.set_ylim(0, 100)
txt(axc, 0.4, 99.5, "(c) Evidence\npipeline", 10.5, C["dgrey"],
    ha="left", va="top", bold=True)

steps = [
    ("1", "Identifiability", "N & days needed\nto recover \u03c4_s", "Fig 2a,b", C["blue"]),
    ("2", "Slow injection", "strong: 20\u201323 d;\nweak band escapes", "Fig 2c\u20133", C["teal"]),
    ("3", "Fast τ anchor", "\u03c4_fast 26 min\n(warm 22 / cool 31)", "Fig 4", C["red"]),
    ("4", "Standards window", "optimum \u2248 15 d,\nclimate / mode dependent", "Fig 5", C["orange"]),
    ("5", "Counterfactual", "graded, excitation-\nadaptive memory rule", "Fig 6", C["green"]),
]
stop, sh, sgap = 87, 15.2, 1.2
for i, (num, head, sub, figno, col) in enumerate(steps):
    y = stop - sh - i * (sh + sgap)
    rbox(axc, 1, y, 98, sh, _tint(col), col, lw=1.2, rs=1.2)
    axc.add_patch(Circle((7.5, y + sh / 2), 3.1, facecolor=col, edgecolor=col, zorder=3.2))
    txt(axc, 7.5, y + sh / 2 + 0.1, num, 8.2, "white", bold=True, z=4)
    txt(axc, 14, y + 12.0, head, 6.8, "#1a1a1a", ha="left", bold=True)
    txt(axc, 14, y + 4.6, sub, 5.9, "#333333", ha="left")
    txt(axc, 96.8, y + 12.0, figno, 6.0, col, ha="right", bold=True)
    if i < len(steps) - 1:
        axc.annotate("", xy=(7.5, y - sgap + 0.2), xytext=(7.5, y - 0.2), zorder=3.4,
                     arrowprops=dict(arrowstyle="-|>", color="#8a949e", lw=1.3,
                                     mutation_scale=9, shrinkA=0, shrinkB=0))

export_panels(fig, [(ax, "r0c0_span3"), (axb, "r1c0_span2"), (axc, "r1c2")], "Fig1_concept")
save(fig, "Fig1_concept")
