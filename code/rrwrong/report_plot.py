"""Three report figures from report_60k.npz (report_figs.py).  Run with `module load python`."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
D = np.load("/pscratch/sd/s/shunth/rrwrong/report_60k.npz")
zt, zp, cat, rem = D["zt"], D["zp"], D["cat"], D["removed"]
A = D["ruleA"]; B = D["dchi2"] & ~A; Cc = D["rr"] & ~A & ~D["dchi2"]
OUT = "/pscratch/sd/s/shunth/figs/report_filter_"
SURF, INK, INK2, GRID, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4", "#eb6834"
S1, S2, S3, KEPT = "#2a78d6", "#1baf7a", "#eda100", "#c9c8c2"
cmap = LinearSegmentedColormap.from_list("seq", ["#d7e6f8", "#8fbbee", "#2a78d6", "#184a88", "#0b2547"])
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF})
ZMAX = float(np.quantile(zt, 0.999)) * 1.05
bins = np.linspace(0, ZMAX, 200); X = np.linspace(0, ZMAX, 100)
nm = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))

def scatter_fig(m, title, sub, path):
    d = (zp - zt) / (1 + zt)
    fig, ax = plt.subplots(figsize=(7, 7.3))
    H, xe, ye = np.histogram2d(zt[m], zp[m], bins=[bins, bins])
    ax.pcolormesh(xe, ye, H.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(H.max(), 2)), rasterized=True)
    ax.plot(X, X, color=INK2, lw=0.8, alpha=0.6)
    for s in (1, -1): ax.plot(X, X + s * 0.1 * (1 + X), color=INK2, lw=0.8, ls="--", alpha=0.6)
    mc = m & cat
    ax.scatter(zt[mc], zp[mc], s=10, color=ORANGE, edgecolors=SURF, linewidths=0.4, zorder=3,
               label="catastrophic: |Δz| > 0.1(1+z)")
    ax.legend(loc="lower right", fontsize=9, facecolor=SURF, edgecolor=GRID)
    ax.set_xlim(0, ZMAX); ax.set_ylim(0, ZMAX); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_xlabel("Redrock z"); ax.set_ylabel("model predicted z")
    ax.text(0.03, 0.97, "objects      %s\ncatastrophic %d  (%.2f%%)\nNMAD         %.1e" % (f"{m.sum():,}", mc.sum(), 100 * cat[m].mean(), nm(d[m])),
            transform=ax.transAxes, va="top", family="monospace", fontsize=10, zorder=4,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURF, ec=GRID, lw=0.6))
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, y=0.985)
    fig.text(0.02, 0.935, sub, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=150); plt.close(fig)

base = "Model v2 (seed 42) on 60,032 validation spectra; colour = spectra per cell (log); dashed = ±0.1(1+z)"
scatter_fig(np.ones(len(zt), bool), "1. Before filtering: 727 orange points", base, OUT + "1_before.png")
scatter_fig(~rem, "2. After filtering: 33 orange points",
            "Same 60,032 spectra minus the 3,675 (6.1%) the filter removes; nothing else changed", OUT + "2_after.png")

# figure 3: what the filter removes, all objects vs orange objects, stages credited in order
fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.9], width_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, :])
rows = [("orange points (727)", cat), ("all objects (60,032)", np.ones(len(zt), bool))]
stages = [("rule A: repeat / Gaia / star-only", A, S1), ("DELTACHI2 < 25", B, S2), ("P_WRONG > 0.01 (learned)", Cc, S3)]
for y, (lab, m) in enumerate(rows):
    left = 0.0
    for name, s, col in stages:
        w = 100 * (s & m).sum() / m.sum()
        ax.barh(y, w, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2, label=name if y == 0 else None)
        if w > 4: ax.text(left + w / 2, y, "%.1f%%" % w, ha="center", va="center", fontsize=9, color=INK)
        left += w
    kw = 100 - left
    if y == 1:
        parts = " + ".join("%.1f%%" % (100 * (st & m).sum() / m.sum()) for _, st, _ in stages)
        ax.annotate("removed %.1f%% = %s" % (left, parts), xy=(left, y - 0.28), xytext=(left + 3, y - 0.5),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.barh(y, kw, left=left, color=KEPT, height=0.55, edgecolor=SURF, linewidth=2, label="kept" if y == 0 else None)
    ax.text(left + kw / 2 if kw > 8 else 101, y, "kept %.1f%%" % kw, ha="center" if kw > 8 else "left", va="center", fontsize=9, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 108); ax.set_xlabel("% removed by each stage (each object credited to the first stage that removes it)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
ax.set_title("Removing 6.1% of the objects removes 95.5% of the orange points", loc="left", fontsize=11, pad=34)
ax2 = fig.add_subplot(gs[1, 0])
for name, s, col in stages:
    m = s & cat
    ax2.scatter(zt[m], zp[m], s=12, color=col, edgecolors=SURF, linewidths=0.4, label="%s (%d)" % (name, m.sum()))
m = cat & ~rem
ax2.scatter(zt[m], zp[m], s=16, color=ORANGE, edgecolors=INK, linewidths=0.5, zorder=4, label="left after filter (%d)" % m.sum())
ax2.plot(X, X, color=INK2, lw=0.8, alpha=0.6); ax2.set_xlim(0, ZMAX); ax2.set_ylim(0, ZMAX); ax2.set_aspect("equal")
ax2.grid(color=GRID, lw=0.5, alpha=0.6); ax2.set_xlabel("Redrock z"); ax2.set_ylabel("model predicted z")
ax2.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, facecolor=SURF, edgecolor=GRID)
ax2.set_title("the 727 orange points, by the stage that removes them", loc="left", fontsize=10)
ax3 = fig.add_subplot(gs[1, 1]); ax3.axis("off")
txt = ("What the filter removes (delete if ANY holds)\n\n"
       "rule A - independent evidence Redrock is wrong\n"
       "  - another observation of the same target\n"
       "    disagrees by > 0.01(1+z)\n"
       "  - Gaia motion/parallax says star, but z > 0.01\n"
       "  - star-only target placed at z > 0.1\n\n"
       "DELTACHI2 < 25\n"
       "  - Redrock's best z barely beats its 2nd best\n\n"
       "P_WRONG > 0.01 (learned, dark/bright only)\n"
       "  - model trained on DESI visual inspection\n"
       "    predicts Redrock z is wrong; inputs are\n"
       "    Redrock fit quality + imaging + Gaia\n"
       "  - Gaia-confirmed stars at rest are exempt")
ax3.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=9.5, color=INK)
fig.suptitle("3. What the filter is, and which orange points it removes", x=0.02, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT + "3_what.png", dpi=150); plt.close(fig)
print("saved")
