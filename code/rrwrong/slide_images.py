"""Three 16:9 slide images (figure + English text) for pasting into PowerPoint (owner 2026-09-29).
  python slide_images.py /pscratch/sd/s/shunth/rrwrong/report_60k.npz OUTDIR   (module load python)
slide_images_zh.py is the Chinese version (needs a CJK font; NERSC has none)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
D = np.load(sys.argv[1]); OUT = sys.argv[2].rstrip("/") + "/"
zt, zp, cat, rem = D["zt"], D["zp"], D["cat"], D["removed"]
A = D["ruleA"]; B = D["dchi2"] & ~A; Cc = D["rr"] & ~A & ~D["dchi2"]
INK, INK2, MUTED, GRID, ORANGE = "#0b0b0b", "#3d3c39", "#6b6a66", "#d9d8d4", "#eb6834"
S1, S2, S3, KEPT, TINT = "#2a78d6", "#1baf7a", "#eda100", "#c9c8c2", "#f3f2ef"
cmap = LinearSegmentedColormap.from_list("seq", ["#d7e6f8", "#8fbbee", "#2a78d6", "#184a88", "#0b2547"])
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 14, "axes.unicode_minus": False,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": "white", "axes.facecolor": "white"})
W, H = 13.333, 7.5
ZMAX = float(np.quantile(zt, 0.999)) * 1.05
bins = np.linspace(0, ZMAX, 200); X = np.linspace(0, ZMAX, 100)
N, NC = len(zt), int(cat.sum()); NK = int((~rem).sum()); NCK = int((cat & ~rem).sum())

def zaxes(ax):
    ax.plot(X, X, color=INK2, lw=0.9, alpha=0.6)
    for s in (1, -1): ax.plot(X, X + s * 0.1 * (1 + X), color=INK2, lw=0.9, ls="--", alpha=0.6)
    ax.set_xlim(0, ZMAX); ax.set_ylim(0, ZMAX); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_xlabel("Redrock z", fontsize=14); ax.set_ylabel("model predicted z", fontsize=14)

def density(ax, m):
    Hh, xe, ye = np.histogram2d(zt[m], zp[m], bins=[bins, bins])
    ax.pcolormesh(xe, ye, Hh.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(Hh.max(), 2)), rasterized=True)
    zaxes(ax)
    mc = m & cat
    ax.scatter(zt[mc], zp[mc], s=16, color=ORANGE, edgecolors="white", linewidths=0.4, zorder=3,
               label="orange: |Δz| > 0.1(1+z)")
    ax.legend(loc="lower right", fontsize=12, facecolor="white", edgecolor=GRID)

def title(fig, t, sub):
    fig.text(0.04, 0.93, t, fontsize=28, weight="bold", color=INK, va="center")
    fig.text(0.04, 0.865, sub, fontsize=14, color=MUTED, va="center")

def para(fig, x, y, lines, size=16, gap=0.058):
    """lines: (text, kind) with kind in {'h','b','s','c'}: header, bullet, sub-bullet, caption."""
    for t, k in lines:
        if k == "h": fig.text(x, y, t, fontsize=size + 2, weight="bold", color=INK, va="top"); y -= gap * 1.05
        elif k == "b": fig.text(x, y, "•  " + t, fontsize=size, color=INK, va="top"); y -= gap
        elif k == "s": fig.text(x + 0.025, y, t, fontsize=size - 2, color=INK2, va="top"); y -= gap * 0.85
        elif k == "c": fig.text(x, y, t, fontsize=size - 3, color=MUTED, va="top"); y -= gap * 0.8
        elif k == "gap": y -= gap * 0.5
    return y

def box(fig, x, y, w, h, color=TINT):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                      transform=fig.transFigure, fc=color, ec="none", zorder=0))

# ---------- slide 1: before
fig = plt.figure(figsize=(W, H))
title(fig, "Before filtering: %d orange points" % NC,
      "60,032 validation spectra (never trained on, all ZWARN = 0)  |  model v2, trained on 300k spectra")
ax = fig.add_axes([0.065, 0.075, 0.40, 0.74]); density(ax, np.ones(N, bool))
y = para(fig, 0.51, 0.78, [
    ("What an orange point is", "h"),
    ("model z and Redrock z differ by > 0.1(1+z)", "b"),
    ("%d objects, %.2f%% of the sample" % (NC, 100 * NC / N), "b"),
    ("", "gap"),
    ("Everything else sits on the diagonal", "h"),
    ("precision NMAD = 2.1×10⁻⁵, about 6 km/s", "b"),
    ("", "gap"),
    ("Three structures in the orange points", "h"),
    ("vertical column at Redrock z ≈ 1.5–1.7", "b"),
    ("known Redrock failure: stars, low-z galaxies at z ≈ 1.65", "s"),
    ("broad cloud at predicted z ≈ 1–1.6", "b"),
    ("mostly ELGs with a single emission line misidentified", "s"),
    ("row at predicted z ≈ 0", "b"),
])
fig.text(0.51, 0.07, "blue shade: spectra per cell (log)  |  dashed: ±0.1(1+z)", fontsize=11, color=MUTED)
fig.savefig(OUT + "slide1_before.png", dpi=200); plt.close(fig)

# ---------- slide 2: after
fig = plt.figure(figsize=(W, H))
title(fig, "After filtering: the same spectra, %d orange points left" % NCK,
      "Only the %s spectra (%.1f%%) the filter removes are taken out; model and predictions unchanged" % (f"{N - NK:,}", 100 * (N - NK) / N))
ax = fig.add_axes([0.065, 0.075, 0.40, 0.74]); density(ax, ~rem)
fig.text(0.51, 0.785, "%d → %d" % (NC, NCK), fontsize=54, weight="bold", color=ORANGE, va="top")
fig.text(0.51, 0.655, "Removing %.1f%% of the spectra\nremoves %.1f%% of the orange points" % (100 * (N - NK) / N, 100 * (NC - NCK) / NC), fontsize=19, weight="bold", va="top", linespacing=1.3)
para(fig, 0.51, 0.535, [
    ("%s spectra remain; %.2f%% agree with Redrock" % (f"{NK:,}", 100 - 100 * NCK / NK), "b"),
    ("the column at z ≈ 1.65 is gone", "b"),
], size=16)
box(fig, 0.505, 0.075, 0.455, 0.31)
para(fig, 0.525, 0.365, [
    ("How to read this", "h"),
    ("Fewer orange points because the validation set lost", "b"),
    ("spectra with doubtful Redrock labels, not a better model", "s"),
    ("Many removed orange points are Redrock errors", "b"),
    ("where the model is right (e.g. the z ≈ 1.65 column)", "s"),
], size=15, gap=0.052)
fig.savefig(OUT + "slide2_after.png", dpi=200); plt.close(fig)

# ---------- slide 3: what the filter is
fig = plt.figure(figsize=(W, H))
title(fig, "The filter: a spectrum is removed if ANY rule holds",
      "Each spectrum is credited to the first rule that removes it")
stages = [("rule A", A, S1), ("DELTACHI2 < 25", B, S2), ("P_WRONG > 0.01", Cc, S3)]
ax = fig.add_axes([0.2, 0.63, 0.63, 0.18])
rows = [("orange points (%d)" % NC, cat), ("all spectra (%s)" % f"{N:,}", np.ones(N, bool))]
for yy, (lab, m) in enumerate(rows):
    left = 0.0
    for name, s, col in stages:
        w = 100 * (s & m).sum() / m.sum()
        ax.barh(yy, w, left=left, color=col, height=0.62, edgecolor="white", linewidth=2)
        if w > 5: ax.text(left + w / 2, yy, "%.1f%%" % w, ha="center", va="center", fontsize=13, color=INK)
        left += w
    ax.barh(yy, 100 - left, left=left, color=KEPT, height=0.62, edgecolor="white", linewidth=2)
    ax.text(101.5, yy, "%.1f%% removed" % left, va="center", fontsize=13, weight="bold", color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=14); ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=11)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
cards = [
    (S1, "Rule A: independent evidence Redrock is wrong", ["another observation of the target disagrees by > 0.01(1+z)", "Gaia motion / parallax says star, but z > 0.01", "star-only target placed at z > 0.1"], A),
    (S2, "DELTACHI2 < 25: Redrock itself is unsure", ["best z barely beats the second-best z"], B),
    (S3, "P_WRONG > 0.01: learned error probability", ["trained on DESI visual inspection (Lan+22, Alexander+22)", "inputs: Redrock fit quality + imaging + Gaia", "dark / bright only; Gaia-confirmed stars at rest kept"], Cc),
]
y = 0.535
for col, head, items, s in cards:
    hgt = 0.046 + 0.031 * (len(items) + 1) + 0.008
    box(fig, 0.035, y - hgt, 0.53, hgt)
    fig.text(0.046, y - 0.006, "●", fontsize=17, color=col, va="top")
    fig.text(0.068, y - 0.008, head, fontsize=14, weight="bold", va="top")
    yy = y - 0.05
    for t in items:
        fig.text(0.075, yy, "•  " + t, fontsize=12.5, color=INK2, va="top"); yy -= 0.031
    fig.text(0.075, yy, "→ removes %.1f%% of spectra and %d orange points" % (100 * s.mean(), (s & cat).sum()), fontsize=12.5, weight="bold", color=INK, va="top")
    y -= hgt + 0.012
ax2 = fig.add_axes([0.63, 0.06, 0.33, 0.36])
for name, s, col in stages:
    m = s & cat
    ax2.scatter(zt[m], zp[m], s=10, color=col, edgecolors="white", linewidths=0.3, label="%s (%d)" % (name, m.sum()))
m = cat & ~rem
ax2.scatter(zt[m], zp[m], s=18, color=ORANGE, edgecolors=INK, linewidths=0.5, zorder=4, label="left after filter (%d)" % m.sum())
zaxes(ax2); ax2.tick_params(labelsize=11); ax2.xaxis.label.set_size(12); ax2.yaxis.label.set_size(12)
ax2.legend(loc="lower left", bbox_to_anchor=(-0.28, 1.01), ncol=2, fontsize=10.5, frameon=False, handletextpad=0.2, columnspacing=0.8)
fig.text(0.625, 0.535, "Which rule removes each orange point", fontsize=14, weight="bold", va="top")
fig.savefig(OUT + "slide3_filter.png", dpi=200); plt.close(fig)
print("saved")
