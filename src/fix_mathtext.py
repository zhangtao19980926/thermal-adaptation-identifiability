# -*- coding: utf-8 -*-
"""Replace matplotlib mathtext ($...$) in fig2-6 with Unicode plain text so that, with
svg.fonttype=none, every label is emitted as an editable SVG <text> (not glyph paths).
Also: Fig2 imshow->pcolormesh (vector heatmap); Fig5 dense scatter downsampled for shape count."""
import os

D = r"D:\thermal-adaptation-series\p1\src\p1mk"
FILES = ["fig2_identifiability.py", "fig3_spectrum.py", "fig4_fast_tau.py",
         "fig5_standards.py", "fig6_counterfactual.py"]

# order matters: hatted / subscripted forms before the bare symbol
REPL = [
    (r'$\\hat{\\tau}_s$', 'τ_s'),
    (r'$\\tau_{fast}$', 'τ_fast'),
    (r'$\\tau_s$', 'τ_s'),
    (r'$\\tau$', 'τ'),
    (r'$\\Delta T$', 'ΔT'),
    (r'$\\Delta$', 'Δ'),
    (r'$\\degree$C', '°C'),
    (r'$R^2$', 'R²'),
    (r'$\\rho$', 'ρ'),
    (r'$\\leq$', '≤'),
    (r'$\\rightarrow$', '→'),
    (r'$\\cdot$', '·'),
    (r'$-$', '−'),
    (r'$T$', 'T'),
]

BLOCKS = {
"fig2_identifiability.py": [(
'''    mesh = ax.imshow(Z, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                     aspect="auto", origin="upper")''',
'''    edges = np.arange(4) - 0.5
    mesh = ax.pcolormesh(edges, edges, Z, cmap="RdYlGn",
                         norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.invert_yaxis()''')],
"fig5_standards.py": [(
'''    d = clean[clean.kband == b]
    col = bandcol[b]
    if b == "A:Tropical":
        axd.scatter(d.out_mon, d.Tneutral, s=14, color=col, alpha=.35, edgecolor="none")
        continue
    b1, b0 = np.polyfit(d.out_mon, d.Tneutral, 1)
    axd.scatter(d.out_mon, d.Tneutral, s=16, color=col, alpha=.75, edgecolor="white", lw=.3,
                label="%s slope %.2f (n=%d)" % (b[0], b1, len(d)))''',
'''    d = clean[clean.kband == b]
    col = bandcol[b]
    dplot = d if len(d) <= 180 else d.sample(180, random_state=0)
    if b == "A:Tropical":
        axd.scatter(dplot.out_mon, dplot.Tneutral, s=14, color=col, alpha=.35, edgecolor="none")
        continue
    b1, b0 = np.polyfit(d.out_mon, d.Tneutral, 1)
    axd.scatter(dplot.out_mon, dplot.Tneutral, s=16, color=col, alpha=.75, edgecolor="white", lw=.3,
                label="%s slope %.2f (n=%d)" % (b[0], b1, len(d)))''')],
}

for fn in FILES:
    fp = os.path.join(D, fn)
    with open(fp, "r", encoding="utf-8") as f:
        txt = f.read()
    n = 0
    for old, new in REPL:
        c = txt.count(old)
        if c:
            txt = txt.replace(old, new); n += c
    for old, new in BLOCKS.get(fn, []):
        assert old in txt, "block not found in %s" % fn
        txt = txt.replace(old, new); n += 1
    with open(fp, "w", encoding="utf-8") as f:
        f.write(txt)
    residual = txt.count("$")
    print("%s: replacements=%s  residual_dollar=%s" % (fn, n, residual))
