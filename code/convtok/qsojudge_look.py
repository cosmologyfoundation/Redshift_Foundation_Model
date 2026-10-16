"""Render samples of qsojudge verdicts for an eye check.  usage: python qsojudge_look.py JUDGE_NPZ OUT_DIR"""
import os, sys
import numpy as np, h5py, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = '/pscratch/sd/s/shunth/'; FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
LAM = np.arange(7781) * 0.8 + 3600.0
BL = {'Lya': 1215.67, 'NV': 1240.81, 'SiIV': 1399.8, 'CIV': 1549.48, 'CIII]': 1908.73, 'MgII': 2799.49, 'Hb': 4862.68,
      '[OIII]': 5008.24, 'Ha': 6564.61, '[OII]': 3728.48, 'CaK': 3934.78}
d = np.load(sys.argv[1], allow_pickle=True); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
H = [h5py.File(P + f, 'r') for f in FN]; rng = np.random.default_rng(3)
sm = lambda f, w, k=15: np.convolve(f * w, np.ones(k), 'same') / np.maximum(np.convolve(w, np.ones(k), 'same'), 1e-30)
for v, k in [('label wrong', 10), ('model wrong', 10)]:
    sel = np.where(d['is_err'] & (d['verdict'] == v))[0]
    sel = np.sort(rng.choice(sel, min(k, len(sel)), replace=False))
    fig, ax = plt.subplots(len(sel), 1, figsize=(15, 2.2 * len(sel)), squeeze=False)
    for a, j in zip(ax[:, 0], sel):
        h = H[int(d['src'][j])]; r = int(d['row'][j]); f = h['BRZ_FLUX'][r]; w = np.maximum(h['BRZ_IVAR'][r], 0)
        y = sm(f, w); g = np.convolve(w, np.ones(15), 'same') > 0
        lo, hi = np.percentile(y[g], [1, 99.5]); pad = 0.15 * (hi - lo); a.plot(LAM[g], y[g], 'k', lw=0.5)
        for z, col, ls, dy in [(d['z_label'][j], 'r', '--', 0.9), (d['z_model'][j], 'b', ':', 0.2)]:
            for n, l0 in BL.items():
                x = l0 * (1 + z)
                if 3600 < x < 9824:
                    a.axvline(x, color=col, ls=ls, lw=0.8, alpha=0.7); a.text(x, hi + dy * pad, n, color=col, fontsize=6, ha='center')
        fmt = lambda S: ' '.join('%s%.0f' % (n, s) for n, s in zip(d['lines'], S) if np.isfinite(s))
        a.set_xlim(3600, 9824); a.set_ylim(lo - pad, hi + 1.3 * pad)
        a.set_title('TID %d | label z %.3f (%s, dchi2 %.0f) S: %s | model z %.3f S: %s'
                    % (d['tid'][j], d['z_label'][j], d['spectype'][j], d['dchi'][j], fmt(d['S_label'][j]), d['z_model'][j], fmt(d['S_model'][j])), fontsize=7, loc='left')
    fig.suptitle('QSO catastrophics judged "%s" (red = label z, blue = model z)' % v, fontsize=11, x=0.01, ha='left')
    plt.tight_layout(); fn = '%s/qsojudge_%s.png' % (out, v.replace(' ', '_')); plt.savefig(fn, dpi=75); plt.close(); print(fn)
