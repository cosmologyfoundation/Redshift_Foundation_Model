"""Where do ZT5's ELG and QSO catastrophics concentrate?  Owner 2026-10-11: 「做，ELG QSO 錯誤分組分析」.
Errors are counted against the label (Redrock Z) -- no attribution to the label is made here.

For every ELG / QSO (priority class) object of the 200k test:
  label z bin, Redrock SPECTYPE, DELTACHI2, survey/program, brightness (h5 SNR_MEDIAN_ABS and TSNR2 of the class),
  evidence at the LABEL z (zfm/evidence.py: best emission line sigma, lines >= 6 sigma, [O II] doublet sigma, broad lines >= 5 sigma)
and per stratum: n objects, n catastrophic, rate, share of the class's errors.
For the errors: the line-swap pattern, i.e. (1+z_model)/(1+z_label) matched to a ratio of two rest lines (within 0.4%),
and "model z ~ 0".
usage: python elgqso_strata.py TEST_NPZ IDS_NPZ OUT_PREFIX
"""
import os, sys, json
import numpy as np, h5py, torch

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
import evidence as E
from grids import LAM_OBS, LAM_NATIVE, z_to_bin, CLASSES

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
NAMES = E.NAMES + E.PAIR_NAMES; ix = {n: i for i, n in enumerate(NAMES)}
PRIO = ['QSO', 'LRG', 'ELG', 'BGS', 'STAR']
SWAP = {'Lya': 1215.67, 'CIV': 1549.48, 'CIII]': 1908.73, 'MgII': 2799.49, '[OII]': 3728.48, '4000break': 4000.0,
        'Hb': 4862.68, '[OIII]': 5008.24, 'Ha': 6564.61}


def strv(a):
    return np.array([x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in a])


def read(src, row, keys, flux=False):
    out = {k: [None] * len(row) for k in keys + (['F', 'V'] if flux else [])}
    for s in (0, 1):
        idx = np.where(src == s)[0]
        if not len(idx):
            continue
        o = np.argsort(row[idx]); rr = row[idx][o]; ii = idx[o]
        with h5py.File(P + FN[s], 'r') as h:
            for k in keys:
                v = h[k][rr]
                if v.dtype == object:
                    v = strv(v)
                for j, i in enumerate(ii):
                    out[k][i] = v[j]
            if flux:
                for a in range(0, len(rr), 4096):
                    F = h['BRZ_FLUX'][rr[a:a + 4096]]; V = h['BRZ_IVAR'][rr[a:a + 4096]]
                    for j in range(len(F)):
                        out['F'][ii[a + j]] = F[j]; out['V'][ii[a + j]] = V[j]
    return {k: (np.stack(v) if k in ('F', 'V') else np.array(v)) for k, v in out.items()}


def evidence_at(F, V, z, bs=512):
    res = []
    for a in range(0, len(z), bs):
        Fi = np.stack([np.interp(LAM_OBS, LAM_NATIVE, f) for f in F[a:a + bs]])
        Vi = np.stack([np.interp(LAM_OBS, LAM_NATIVE, np.maximum(v, 0)) for v in V[a:a + bs]])
        x = torch.from_numpy(np.stack([Fi * np.sqrt(Vi), np.sqrt(Vi)], 1).astype(np.float32)).cuda()
        zb = torch.from_numpy(np.asarray(z_to_bin(z[a:a + bs]))).long().cuda().view(-1, 1)
        res.append(E.candidate_features(E.spectrum_maps(x, sky=0.3), zb, slog=False, pairs=True)[:, 0].cpu().numpy())
    return np.concatenate(res)


def table(title, key, m_cls, bad, bins_or_labels, fmt):
    print('\n  by %s' % title)
    print('    %-22s %7s %6s %8s %8s' % ('group', 'n', 'errors', 'rate', 'share'))
    tot = (bad & m_cls).sum()
    for lab, sel in bins_or_labels:
        m = m_cls & sel
        if m.sum() == 0:
            continue
        print('    %-22s %7d %6d %7.2f%% %7.1f%%' % (lab, m.sum(), (bad & m).sum(), 100 * bad[m].mean(), 100 * (bad & m).sum() / max(tot, 1)))


def bins(v, edges, f='%g'):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(((f + '-' + f) % (a, b), (v >= a) & (v < b)))
    return out


def main():
    tnpz, ids, out = sys.argv[1:4]
    d, I = np.load(tnpz), np.load(ids)
    zt, zp, cls = d['zt'].astype(float), d['zp'].astype(float), d['cls']
    prio = np.full(len(zt), 'none', object)
    for nm in PRIO[::-1]:
        prio[cls[:, CLASSES.index(nm)] > 0] = nm
    sel = np.where(np.isin(prio, ['ELG', 'QSO']))[0]
    print('ELG+QSO objects: %d' % len(sel), flush=True)
    R = read(I['src'][sel], I['row'][sel], ['SPECTYPE', 'DELTACHI2', 'SNR_MEDIAN_ABS', 'TSNR2_ELG', 'TSNR2_QSO', 'SURVEY', 'PROGRAM'], flux=True)
    ev = evidence_at(R['F'], R['V'], zt[sel])
    zt, zp, prio = zt[sel], zp[sel], prio[sel]
    bad = np.abs(zp - zt) / (1 + zt) > 0.1
    best, n6 = ev[:, ix['best_emis']], ev[:, ix['n_emis>=6']]
    oii = (ev[:, ix['dbl_b:oii']] + ev[:, ix['dbl_r:oii']]) / np.sqrt(2)
    nb5 = sum((ev[:, ix['broad:' + l]] >= 5) for l in ['Lya', 'NV', 'SiIV', 'CIV', 'CIII]', 'MgII']) + \
        sum((ev[:, ix['narrow:' + l]] >= 5) for l in ['Hb', 'Ha'])
    for c, zedges, tsn in [('ELG', [0, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 4.1], 'TSNR2_ELG'),
                           ('QSO', [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.1], 'TSNR2_QSO')]:
        m = prio == c
        print('\n================ %s: %d objects, %d catastrophic (%.2f%%)' % (c, m.sum(), (bad & m).sum(), 100 * bad[m].mean()))
        table('label z (Redrock Z)', 'z', m, bad, bins(zt, zedges), '%g')
        table('Redrock SPECTYPE', 'st', m, bad, [(s, R['SPECTYPE'] == s) for s in ['GALAXY', 'QSO', 'STAR']], '')
        table('DELTACHI2', 'dchi', m, bad, bins(R['DELTACHI2'], [9, 16, 25, 50, 100, 400, 1e12], '%g'), '')
        q = np.quantile(R['SNR_MEDIAN_ABS'][m], [0, .2, .4, .6, .8, 1]); q[-1] += 1e-9
        table('SNR_MEDIAN_ABS quintile', 'snr', m, bad, bins(R['SNR_MEDIAN_ABS'], q, '%.2f'), '')
        q = np.quantile(R[tsn][m], [0, .2, .4, .6, .8, 1]); q[-1] += 1e-3
        table('%s quintile' % tsn, 'tsnr', m, bad, bins(R[tsn], q, '%.0f'), '')
        table('survey/program', 'sp', m, bad, [('%s/%s' % (s, p), (R['SURVEY'] == s) & (R['PROGRAM'] == p))
                                             for s in ['main', 'sv3'] for p in ['dark', 'bright', 'backup']], '')
        table('best emission line at label z (sigma)', 'best', m, bad, bins(best, [-1e9, 3, 6, 10, 20, 1e9], '%g'), '')
        table('lines >= 6 sigma at label z', 'n6', m, bad, [('0', n6 == 0), ('1', n6 == 1), ('2', n6 == 2), ('>=3', n6 >= 3)], '')
        if c == 'ELG':
            table('[O II] doublet sigma at label z', 'oii', m, bad, bins(oii, [-1e9, 3, 6, 10, 20, 1e9], '%g'), '')
        else:
            table('broad lines >= 5 sigma at label z', 'nb', m, bad, [('0', nb5 == 0), ('1', nb5 == 1), ('2', nb5 == 2), ('>=3', nb5 >= 3)], '')
        e = np.where(m & bad)[0]
        r = (1 + zp[e]) / (1 + zt[e])
        pat = []
        for k in range(len(e)):
            if zp[e[k]] < 0.02:
                pat.append('model z ~ 0'); continue
            best_p, bd = None, 0.004
            for a, la in SWAP.items():
                for b, lb in SWAP.items():
                    if a != b and abs(np.log(r[k]) - np.log(la / lb)) < bd:
                        best_p, bd = '%s read as %s' % (a, b), abs(np.log(r[k]) - np.log(la / lb))
            pat.append(best_p or 'no line ratio match')
        u, n = np.unique(pat, return_counts=True)
        o = np.argsort(-n)
        print('\n  error patterns (label line -> where the model put it), top 12 of %d errors' % len(e))
        for j in o[:12]:
            print('    %-34s %5d  %5.1f%%' % (u[j], n[j], 100 * n[j] / len(e)))
    np.savez(out + '.npz', zt=zt, zp=zp, prio=prio, bad=bad, spectype=R['SPECTYPE'], dchi=R['DELTACHI2'], snr=R['SNR_MEDIAN_ABS'],
             tsnr_elg=R['TSNR2_ELG'], tsnr_qso=R['TSNR2_QSO'], survey=R['SURVEY'], program=R['PROGRAM'], best=best, n6=n6, oii=oii, nb5=nb5,
             tid_row=I['row'][sel], src=I['src'][sel])
    print('\n->', out + '.npz')


if __name__ == '__main__':
    main()
