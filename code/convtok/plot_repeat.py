"""Model z vs Redrock z on the 200k test; red = objects where Redrock disagrees with ITSELF (two confident ZWARN=0
answers from independent coadds, |dz|/(1+z) > 0.1).  Owner 2026-10-11.
usage: python plot_repeat.py TEST_NPZ REPEAT_NPZ OUT_PNG [model label]"""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d, r = np.load(sys.argv[1]), np.load(sys.argv[2])
out = sys.argv[3]; lab = sys.argv[4] if len(sys.argv) > 4 else 'model'
zt, zp = d['zt'].astype(float), d['zp'].astype(float)
assert np.allclose(zt, r['z'], atol=1e-6)
c, has = r['conflict'], r['has_rep']
cat = np.abs(zp - zt) / (1 + zt) > 0.1
n = len(zt)
fig, ax = plt.subplots(figsize=(8.2, 8))
ax.scatter(zt[~c], zp[~c], s=1.2, c='0.45', alpha=0.45, lw=0, rasterized=True, label='Redrock self-consistent or no repeat')
ax.scatter(zt[c], zp[c], s=16, c='#d62728', alpha=0.95, lw=0, label='Redrock disagrees with itself')
g = np.linspace(0, 4.1, 50)
ax.plot(g, g, 'k-', lw=0.6)
for s in (-1, 1):
    ax.plot(g, g + s * 0.1 * (1 + g), 'k--', lw=0.5)
ax.set_xlim(-0.05, 4.05); ax.set_ylim(-0.05, 4.05); ax.set_aspect('equal')
ax.set_xlabel('Redrock z (label)'); ax.set_ylabel('%s z' % lab)
txt = ('Redrock vs Redrock conflicts: %d = %.2f%% of all %s test spectra\n'
       'with an independent confident repeat: %.1f%% of all; conflict rate among them %.2f%%\n'
       'among %s catastrophics (%d): %d conflicts = %.1f%%\n'
       'among %s agreements (%d): %d conflicts = %.2f%%'
       % (c.sum(), 100 * c.mean(), format(n, ','), 100 * has.mean(), 100 * c[has].mean(),
          lab, cat.sum(), (c & cat).sum(), 100 * (c & cat).sum() / max(cat.sum(), 1),
          lab, (~cat).sum(), (c & ~cat).sum(), 100 * (c & ~cat).sum() / max((~cat).sum(), 1)))
zo = r['z_other']; m = c & cat
txt += ('\n\nonly objects WITH a repeat (%s):\n'
        '  %s catastrophic %d -> %d of them Redrock self-conflicts (%.0f%%)\n'
        '  %s agrees with label %d -> %d self-conflicts (%.2f%%)\n'
        '  in %d of the %d, %s z = the OTHER Redrock z (within 0.01)'
        % (format(int(has.sum()), ','), lab, (has & cat).sum(), m.sum(), 100 * m.sum() / max((has & cat).sum(), 1),
           lab, (has & ~cat).sum(), (c & ~cat).sum(), 100 * (c & ~cat).sum() / max((has & ~cat).sum(), 1),
           (m & (np.abs(zp - zo) / (1 + zo) < 0.01)).sum(), m.sum(), lab))
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(fc='white', ec='0.7', alpha=0.9))
ax.legend(loc='lower right', fontsize=8.5, markerscale=3)
plt.tight_layout(); plt.savefig(out, dpi=130); print(out); print(txt)
