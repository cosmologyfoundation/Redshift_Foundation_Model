"""Look at ZT5m40's ELG / QSO catastrophics before changing anything else (owner 2026-10-13: 「先看光譜」).

From the 200k test (same stream as every 1M run, so test_ids.npz applies), per class:
  page A  wrong, no >= 6 sigma line at the label z   (the hard subgroup holding ~87% of the errors)
  page B  RIGHT, no >= 6 sigma line at the label z   (same subgroup, model agrees with Redrock -- the contrast)
  page C  wrong, >= 1 line >= 6 sigma at the label z
  QSO also: page D wrong with Redrock SPECTYPE GALAXY vs page E right with SPECTYPE GALAXY
Each panel: 9-px inverse-variance smoothed flux (native grid); lines at the label (Redrock) z red dashed, model z blue
dotted; title: TARGETID, label z / model z, Redrock SPECTYPE and DELTACHI2, n lines >= 6 sigma and best line sigma at
both z, median S/N per pixel.
usage: python look_elgqso.py TEST_NPZ IDS_NPZ OUT_DIR
"""
import os, sys
import numpy as np, h5py, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
import evidence as E
from grids import LAM_OBS, LAM_NATIVE, z_to_bin, CLASSES

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
L = {'Lya': 1215.67, 'CIV': 1549.48, 'CIII]': 1908.73, 'MgII': 2799.49, '[OII]': 3728.48, 'CaK': 3934.78, 'CaH': 3969.59,
     'Hd': 4102.89, 'Hb': 4862.68, '[OIII]': 5008.24, 'Mgb': 5176.7, 'NaD': 5894.6, 'Ha': 6564.61, '[SII]': 6725.0}
ix = {n: i for i, n in enumerate(E.NAMES)}
PRIO = ['QSO', 'LRG', 'ELG', 'BGS', 'STAR']


def rows(I, idx, keys):
    out = {k: [None] * len(idx) for k in keys}
    for s in (0, 1):
        sel = np.where(I['src'][idx] == s)[0]
        if not len(sel):
            continue
        r = I['row'][idx][sel]; o = np.argsort(r)
        with h5py.File(P + FN[s], 'r') as h:
            for k in keys:
                v = h[k][r[o]] if h[k].ndim == 1 else np.stack([h[k][x] for x in r[o]])
                for j, i in enumerate(sel[o]):
                    out[k][i] = v[j].decode() if isinstance(v[j], bytes) else v[j]
    return out


def ev(F, V, z):
    Fi = np.stack([np.interp(LAM_OBS, LAM_NATIVE, f) for f in F]); Vi = np.stack([np.interp(LAM_OBS, LAM_NATIVE, np.maximum(v, 0)) for v in V])
    x = torch.from_numpy(np.stack([Fi * np.sqrt(Vi), np.sqrt(Vi)], 1).astype(np.float32))
    f = E.candidate_features(E.spectrum_maps(x, sky=0.3), torch.from_numpy(np.asarray(z_to_bin(z))).long().view(-1, 1), slog=False)[:, 0].numpy()
    return f[:, ix['n_emis>=6']], f[:, ix['best_emis']]


def sm(f, w, k=9):
    ker = np.ones(k); return np.convolve(f * w, ker, 'same') / np.maximum(np.convolve(w, ker, 'same'), 1e-30)


def page(title, sel, R, zt, zp, info, fn):
    lam = LAM_NATIVE
    fig, ax = plt.subplots(len(sel), 1, figsize=(15, 2.2 * len(sel)), squeeze=False)
    for a, j in zip(ax[:, 0], sel):
        f, w = R['BRZ_FLUX'][j], np.maximum(R['BRZ_IVAR'][j], 0)
        y = sm(f, w); good = np.convolve(w, np.ones(9), 'same') > 0
        lo, hi = np.percentile(y[good], [1, 99.5]); pad = 0.15 * (hi - lo)
        a.plot(lam[good], y[good], 'k', lw=0.5)
        for z, col, ls, dy in [(zt[j], 'r', '--', 0.9), (zp[j], 'b', ':', 0.2)]:
            for n, l0 in L.items():
                xx = l0 * (1 + z)
                if 3600 < xx < 9824:
                    a.axvline(xx, color=col, ls=ls, lw=0.8, alpha=0.7); a.text(xx, hi + dy * pad, n, color=col, fontsize=6, ha='center')
        a.set_xlim(3600, 9824); a.set_ylim(lo - pad, hi + 1.3 * pad)
        a.set_title(info(j), fontsize=8, loc='left')
    fig.suptitle(title, fontsize=11, x=0.01, ha='left'); plt.tight_layout(); plt.savefig(fn, dpi=75); plt.close(); print(fn, flush=True)


def main():
    tnpz, ids, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    d, I = np.load(tnpz), np.load(ids)
    assert np.array_equal(np.float32(d['zt']), np.float32(I['z'])), 'test stream differs from test_ids'
    zt, zp, cls = d['zt'].astype(float), d['zp'].astype(float), d['cls']
    prio = np.full(len(zt), 'none', object)
    for nm in PRIO[::-1]:
        prio[cls[:, CLASSES.index(nm)] > 0] = nm
    bad = np.abs(zp - zt) / (1 + zt) > 0.1
    rng = np.random.default_rng(7)
    for c in ['ELG', 'QSO']:
        e = np.where((prio == c) & bad)[0]; ok = rng.choice(np.where((prio == c) & ~bad)[0], 3000, replace=False)
        pool = np.concatenate([e, ok])
        R = rows(I, pool, ['BRZ_FLUX', 'BRZ_IVAR', 'TARGETID', 'DELTACHI2', 'SPECTYPE', 'SNR_MEDIAN_ABS'])
        R = {k: (np.stack(v) if k.startswith('BRZ') else np.array(v)) for k, v in R.items()}
        n6t, bt = ev(R['BRZ_FLUX'], R['BRZ_IVAR'], zt[pool]); n6p, bp = ev(R['BRZ_FLUX'], R['BRZ_IVAR'], zp[pool])
        isE = np.arange(len(pool)) < len(e)
        zt_, zp_ = zt[pool], zp[pool]
        info = lambda j: ('TID %d | label z %.3f (%s, dchi2 %.0f) n6=%d best %.1fs | model z %.3f n6=%d best %.1fs | S/N/px %.2f'
                          % (R['TARGETID'][j], zt_[j], R['SPECTYPE'][j], R['DELTACHI2'][j], n6t[j], bt[j], zp_[j], n6p[j], bp[j], R['SNR_MEDIAN_ABS'][j]))
        pick = lambda m, k: np.sort(rng.choice(np.where(m)[0], min(k, m.sum()), replace=False)) if m.any() else []
        print('%s: errors %d, of which no >=6 sigma line at label z: %d' % (c, isE.sum(), (isE & (n6t == 0)).sum()), flush=True)
        page('%s - WRONG, no >=6 sigma line at the label z' % c, pick(isE & (n6t == 0), 10), R, zt_, zp_, info, '%s/%s_A_wrong_noline.png' % (out, c))
        page('%s - RIGHT, no >=6 sigma line at the label z (contrast)' % c, pick(~isE & (n6t == 0), 10), R, zt_, zp_, info, '%s/%s_B_right_noline.png' % (out, c))
        page('%s - WRONG, >=1 line >=6 sigma at the label z' % c, pick(isE & (n6t >= 1), 10), R, zt_, zp_, info, '%s/%s_C_wrong_lines.png' % (out, c))
        if c == 'QSO':
            page('QSO - WRONG, Redrock SPECTYPE GALAXY', pick(isE & (R['SPECTYPE'] == 'GALAXY'), 10), R, zt_, zp_, info, '%s/QSO_D_wrong_galaxy.png' % out)
            page('QSO - RIGHT, Redrock SPECTYPE GALAXY (contrast)', pick(~isE & (R['SPECTYPE'] == 'GALAXY'), 10), R, zt_, zp_, info, '%s/QSO_E_right_galaxy.png' % out)
        # numbers for the contrast: what differs between wrong and right inside the no-line subgroup
        for nm, m in [('wrong', isE & (n6t == 0)), ('right', ~isE & (n6t == 0))]:
            print('  %s no-line %-5s n %4d | median S/N/px %.2f  best line at label z %.1fs  DELTACHI2 %.0f  label z %.2f' %
                  (c, nm, m.sum(), np.median(R['SNR_MEDIAN_ABS'][m]), np.median(bt[m]), np.median(R['DELTACHI2'][m]), np.median(zt_[m])), flush=True)


if __name__ == '__main__':
    main()
