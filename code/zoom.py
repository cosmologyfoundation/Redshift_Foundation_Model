"""Diagnostic views for the objects that are unreadable as plain flux.

Last round every faint, featureless-looking spectrum was defaulted to ELG, and
that produced most of the errors: two were stars whose Balmer absorption I never
checked for, and one was a quasar whose only line was BROAD and therefore
invisible to the single narrow filter I was using.

So this looks at four widths at once rather than one:
  continuum (121 px)   the 4000 A break and overall shape
  70 km/s              nebular emission lines
  900 km/s             stellar Balmer absorption
  3000 km/s            quasar broad lines
plus a blue zoom, where Ca II H+K, the Balmer series and the break all live.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import uniform_filter1d

R = os.environ.get('C2', '/pscratch/sd/s/shunth/claude_learning_map/cohort2')
d = np.load(os.path.join(R, 'data', 'cohort.npz'), allow_pickle=True)
LAM, FL, IV, IDS = d['lam'], d['flux'], d['ivar'], list(d['id'])
OUT = os.path.join(R, 'zoom'); os.makedirs(OUT, exist_ok=True)


def cont(fl, iv, w):
    good = iv > 0
    n = uniform_filter1d(np.where(good, fl * iv, 0.), w)
    de = uniform_filter1d(np.where(good, iv, 0.), w)
    return np.divide(n, de, out=np.full_like(n, np.nan), where=de > 0)


def sig(fl, iv, kms):
    c = np.nan_to_num(cont(fl, iv, 301))
    res = np.where(iv > 0, fl - c, 0.)
    n = max(3, int(round(kms / 299792.458 * 6000. / 0.8)) | 1)
    a = uniform_filter1d(res * np.where(iv > 0, iv, 0.), n) * n
    b = np.sqrt(np.maximum(uniform_filter1d(np.where(iv > 0, iv, 0.), n) * n,
                           1e-30))
    return a / b


want = sys.argv[1].split(',')
PER = 4
for s in range(0, len(want), PER):
    grp = want[s:s + PER]
    fig, axes = plt.subplots(len(grp), 4, figsize=(19, 2.5 * len(grp)),
                             squeeze=False)
    for r, oid in enumerate(grp):
        i = IDS.index(oid)
        fl, iv = FL[i], IV[i]
        c = cont(fl, iv, 121)
        panels = [('continuum 121 px', LAM, c, None, (3600, 9824)),
                  ('blue zoom 3600-5600', LAM, cont(fl, iv, 41), None,
                   (3600, 5600)),
                  ('sigma 70 km/s (nebular)', LAM, sig(fl, iv, 70.), 3,
                   (3600, 9824)),
                  ('sigma 900 / 3000 km/s (stellar / broad)', LAM,
                   sig(fl, iv, 900.), 3, (3600, 9824))]
        for cidx, (ttl, x, y, thr, xl) in enumerate(panels):
            ax = axes[r][cidx]
            ax.plot(x, y, lw=.55, color='0.15')
            if cidx == 3:
                ax.plot(x, sig(fl, iv, 3000.), lw=.7, color='C1', alpha=.8)
            k = (x > xl[0]) & (x < xl[1]) & np.isfinite(y)
            if k.any():
                a, b = np.nanpercentile(y[k], [1, 99.5])
                pad = .25 * (b - a) + 1e-9
                ax.set_ylim(a - pad, b + pad)
            if thr:
                for t in (thr, -thr):
                    ax.axhline(t, color='C0', lw=.6, ls=':')
                ax.axhline(0, color='0.6', lw=.5)
            ax.set_xlim(*xl); ax.grid(alpha=.3)
            ax.xaxis.set_major_locator(MultipleLocator(
                500 if xl[1] - xl[0] > 3000 else 400))
            ax.tick_params(labelsize=7)
            ax.set_title(f'{oid}  {ttl}', fontsize=8, loc='left')
    plt.tight_layout()
    p = os.path.join(OUT, f'zoom_{s//PER + 1:02d}.png')
    fig.savefig(p, dpi=96); plt.close(fig); print('wrote', p)
