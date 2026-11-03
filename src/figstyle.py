# -*- coding: utf-8 -*-
"""Shared publication style for Building & Environment composite figures.
Every figure is a compact multi-panel composite; export PNG (300 dpi) + vector PDF/SVG
(SVG is later inserted into PowerPoint and ungrouped into editable drawing objects)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import font_manager

FIGDIR = r"D:\thermal-adaptation-series\p1\outputs\figures"
os.makedirs(FIGDIR, exist_ok=True)

# colour-blind-safe palette
C = dict(blue="#2471a3", red="#c0392b", green="#1e8449", purple="#7d3c98",
         orange="#e67e22", grey="#7f8c8d", dgrey="#34495e", lblue="#5dade2",
         lred="#ec7063", lgreen="#58d68d", gold="#b7950b", teal="#117a8b")
SEQ = ["#2471a3", "#5dade2", "#e67e22", "#c0392b", "#7d3c98", "#1e8449", "#7f8c8d"]

def apply_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#4d4d4d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.dpi": 120, "savefig.dpi": 300,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
    })

def panel_label(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", color="#1a1a1a")

def export_panels(fig, panels, name):
    """panels: list of (ax, panel_id[, extra artists]). Export each axes region as a
    standalone svg+png so every panel becomes an independent, movable, editable object.

    Crop uses a REGULAR GRID rather than ax.get_tightbbox(): with manual gridspecs the
    per-axes tight bbox can bleed into a neighbour (stealing its ylabel/legend) or clip a
    twin-axis label / edge legend. Rules:
      * an edge facing another panel stops at the midpoint of that gap (no overlap);
      * an outer (figure-edge) side wraps the panel's full content bbox (title, axis
        labels, legend, annotations and any coincident twin axis) plus a small margin."""
    outdir = os.path.join(FIGDIR, "panels")
    os.makedirs(outdir, exist_ok=True)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pxim = fig.dpi_scale_trans.inverted()          # display pixels -> inches
    figw, figh = fig.get_size_inches()
    prim = [it[0] for it in panels]
    extra_map = {id(it[0]): (it[2] if len(it) > 2 else []) for it in panels}

    def twin_axes_of(ax):
        p = ax.get_position()
        out = []
        for o in fig.axes:
            if o is ax or o in prim:
                continue
            q = o.get_position()
            if (abs(q.x0 - p.x0) < 1e-3 and abs(q.y0 - p.y0) < 1e-3 and
                    abs(q.width - p.width) < 1e-3 and abs(q.height - p.height) < 1e-3):
                out.append(o)
        return out

    def content_bbox_in(ax, more_artists):
        bs = []
        for a in [ax] + twin_axes_of(ax):
            try:
                bs.append(a.get_tightbbox(rend))
            except Exception:
                pass
            lg = a.get_legend()
            if lg is not None:
                try:
                    bs.append(lg.get_window_extent(rend))
                except Exception:
                    pass
            for t in list(a.texts):
                try:
                    bs.append(t.get_window_extent(rend))
                except Exception:
                    pass
        for ar in more_artists:
            try:
                bs.append(ar.get_window_extent(rend) if hasattr(ar, "get_window_extent")
                          else ar.get_tightbbox(rend))
            except Exception:
                pass
        if not bs:
            return None
        return Bbox.union(bs).transformed(pxim)     # inches

    def gap_mids(ax):
        p = ax.get_position()
        lefts, rights, bots, tops = [], [], [], []
        for o in prim:
            if o is ax:
                continue
            q = o.get_position()
            row = not (q.y1 < p.y0 - 1e-6 or q.y0 > p.y1 + 1e-6)
            col = not (q.x1 < p.x0 - 1e-6 or q.x0 > p.x1 + 1e-6)
            if row and q.x1 <= p.x0 + 1e-6:
                lefts.append(q.x1)
            if row and q.x0 >= p.x1 - 1e-6:
                rights.append(q.x0)
            if col and q.y1 <= p.y0 + 1e-6:
                bots.append(q.y1)
            if col and q.y0 >= p.y1 - 1e-6:
                tops.append(q.y0)
        L = ((max(lefts) + p.x0) / 2 if lefts else None)
        R = ((p.x1 + min(rights)) / 2 if rights else None)
        B = ((max(bots) + p.y0) / 2 if bots else None)
        T = ((p.y1 + min(tops)) / 2 if tops else None)
        return L, R, B, T

    PAD_X, PAD_Y = 0.012, 0.018
    for item in panels:
        ax, pid = item[0], item[1]
        p = ax.get_position()
        Lf, Rf, Bf, Tf = gap_mids(ax)
        cb = content_bbox_in(ax, extra_map[id(ax)])
        x0 = Lf * figw if Lf is not None else (cb.x0 - PAD_X * figw if cb else p.x0 * figw)
        x1 = Rf * figw if Rf is not None else (cb.x1 + PAD_X * figw if cb else p.x1 * figw)
        y0 = Bf * figh if Bf is not None else (cb.y0 - PAD_Y * figh if cb else p.y0 * figh)
        y1 = Tf * figh if Tf is not None else (cb.y1 + PAD_Y * figh if cb else p.y1 * figh)
        x0 = max(0.0, min(x0, p.x0 * figw)); x1 = min(figw, max(x1, p.x1 * figw))
        y0 = max(0.0, min(y0, p.y0 * figh)); y1 = min(figh, max(y1, p.y1 * figh))
        bb = Bbox.from_bounds(x0, y0, x1 - x0, y1 - y0)
        for ext in ("svg", "png"):
            fp = os.path.join(outdir, "%s_%s.%s" % (name, pid, ext))
            fig.savefig(fp, bbox_inches=bb, dpi=600 if ext == "png" else None)
    print("panels saved", name, len(panels), "->", outdir)

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fp = os.path.join(FIGDIR, name + "." + ext)
        fig.savefig(fp, bbox_inches="tight", dpi=600 if ext == "png" else None)
    print("saved", name, "->", FIGDIR)
    plt.close(fig)
