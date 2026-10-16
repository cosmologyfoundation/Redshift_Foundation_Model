"""Render samples of the rrjudge verdicts so the rule can be checked by eye (memory: look before formalising).
Top: 9-px inverse-variance smoothed flux; lines at the Redrock z (red, dashed) and the model z (blue, dotted).
usage: python rrlook.py JUDGE_NPZ OUT_DIR"""
import os, sys
import numpy as np, h5py, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
L = {'Lya': 1215.67, 'CIV': 1549.48, 'CIII]': 1908.73, 'MgII': 2799.49, '[OII]': 3728.48, 'CaK': 3934.78, 'CaH': 3969.59,
     'Hd': 4102.89, 'Hb': 4862.68, '[OIII]': 5008.24, 'Mgb': 5176.7, 'NaD': 5894.6, 'Ha': 6564.61, '[SII]': 6725.0}
lam = np.arange(7781) * 0.8 + 3600
d = np.load(sys.argv[1], allow_pickle=True); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
H = [h5py.File(P + f, 'r') for f in FN]
rng = np.random.default_rng(1)


def sm(f, w, k=9):
    ker = np.ones(k); return np.convolve(f * w, ker, 'same') / np.maximum(np.convolve(w, ker, 'same'), 1e-30)


pages = []
for c in ['ELG', 'QSO']:
    for v, k in [('undecided', 8), ('Redrock wrong', 8), ('model wrong', 4)]:
        i = np.where(d['is_err'] & (d['cls'] == c) & (d['verdict'] == v))[0]
        if len(i):
            pages.append((c, v, np.sort(rng.choice(i, min(k, len(i)), replace=False))))
for c, v, sel in pages:
    fig, ax = plt.subplots(len(sel), 1, figsize=(15, 2.3 * len(sel)), squeeze=False)
    for a, j in zip(ax[:, 0], sel):
        h = H[int(d['src'][j])]; r = int(d['row'][j])
        f = h['BRZ_FLUX'][r]; w = np.maximum(h['BRZ_IVAR'][r], 0)
        y = sm(f, w); good = np.convolve(w, np.ones(9), 'same') > 0
        lo, hi = np.percentile(y[good], [1, 99.5]); pad = 0.15 * (hi - lo)
        a.plot(lam[good], y[good], 'k', lw=0.5)
        for z, col, ls in [(d['z_rr'][j], 'r', '--'), (d['z_m'][j], 'b', ':')]:
            for n, l0 in L.items():
                x = l0 * (1 + z)
                if 3600 < x < 9824:
                    a.axvline(x, color=col, ls=ls, lw=0.8, alpha=0.7)
                    a.text(x, hi + (0.9 if col == 'r' else 0.2) * pad, n, color=col, fontsize=6, ha='center')
        a.set_ylim(lo - pad, hi + 1.3 * pad); a.set_xlim(3600, 9824)
        a.set_title('TID %d | target %s | Redrock z %.3f (%s, dchi2 %.0f, score %d, best %.1fs) | model z %.3f (score %d, best %.1fs) | %s'
                    % (d['tid'][j], c, d['z_rr'][j], d['spectype'][j], d['dchi'][j], d['rr_score'][j], d['rr_best'][j],
                       d['z_m'][j], d['m_score'][j], d['m_best'][j], v), fontsize=8, loc='left')
    plt.tight_layout(); fn = '%s/look_%s_%s.png' % (out, c, v.replace(' ', '_'))
    plt.savefig(fn, dpi=80); plt.close(); print(fn)
