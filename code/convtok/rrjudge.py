"""Is Redrock really wrong where the model disagrees by more than 0.1?  Owner 2026-10-10: the standard is NOT clean-2;
it is how ELG / QSO redshifts are originally decided, plus what the blind-reading work learned.

For every catastrophic object both redshifts are read from the spectrum with the SAME fixed instruments
(zfm/evidence.py: no parameters, no labels, no Redrock):

  learned (memory: feature significance = confidence, refusal needs corroboration, [O III] veto, visit the peak,
  ELG needs the continuum too, alias errors are symmetric in line COUNT but not in line PARTNERS)
    n6        emission lines >= 6 sigma at z (narrow 70 km/s filter for galaxy lines, broad 3000 km/s for QSO lines)
    break     4000 A break >= 6 sigma at z (a break galaxy)
    unexpl    >= 8 sigma narrow peaks that z leaves unexplained
    contra    absorption lines that come out as >= 3 sigma emission at z
    vetoes    fixed atomic physics, each needs a strong line to fire:
                [O III] 4960/5008 = 1/2.98   (5008 >= 8 sigma and the ratio off by > 5 sigma)
                [O II] 3727/3729 in [0.35, 1.5] (doublet >= 8 sigma and a member short by > 5 sigma)
                H alpha / H beta >= 2.86       (short by > 5 sigma)
    score(z) = n6 + [break] - unexpl - contra - 2 x vetoes
    verdict   Redrock wrong  : score(model) - score(Redrock) >= 2 and score(model) >= 2
              model wrong    : score(Redrock) - score(model) >= 2 and score(Redrock) >= 2
              undecided      : otherwise (evidence weak or symmetric)
  DESI's own criteria (reported next to the verdict, never inside the model):
    ELG  Raichoor et al. 2023: log10(FOII_SNR) > 0.9 - 0.2 log10(DELTACHI2).  FOII_SNR here = the [O II] doublet's
         combined matched-filter significance at the Redrock z (a stand-in for DESI's Gaussian doublet fit).
    QSO  QuasarNet / MgII afterburner idea: broad lines >= 5 sigma at each z (Lya, NV, SiIV, CIV, CIII], MgII on the
         3000 km/s map; plus H beta / H alpha).
    all  Redrock's own DELTACHI2 (ZWARN=0 means >= 9; < 25 was where the label-noise "square" lived).
Calibration: the same rule on CONTROLS -- objects where model and Redrock agree (|dz| < 0.003), each paired with a decoy
z built from a real error's ratio (1+z_model)/(1+z_Redrock) of the same class.  "Redrock wrong" on a control is a
false accusation; its rate is the rule's error rate.
usage: python rrjudge.py RESULT_NPZ IDS_NPZ OUT_PREFIX
"""
import os, sys
import numpy as np, h5py, torch

sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
import evidence as E
from grids import LAM_OBS, LAM_NATIVE, z_to_bin, z_in_grid

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
NAMES = E.NAMES + E.PAIR_NAMES
ix = {n: i for i, n in enumerate(NAMES)}
PRIO = ['QSO', 'LRG', 'ELG', 'BGS', 'STAR']                                   # priority order, as data.prio_class


def prio(cls, classes):
    out = np.full(len(cls), 'none', object)
    for name in PRIO[::-1]:
        out[cls[:, classes.index(name)] > 0] = name
    return out


def read_rows(src, row, fields):
    """rows of the two h5 files, in the given order"""
    out = {k: [None] * len(row) for k in fields}
    for s in (0, 1):
        sel = np.where(src == s)[0]
        if not len(sel):
            continue
        o = np.argsort(row[sel]); rr = row[sel][o]
        with h5py.File(P + FN[s], 'r') as h:
            for k in fields:
                v = h[k][rr] if h[k].ndim == 1 else np.stack([h[k][r] for r in rr])
                for j, i in enumerate(sel[o]):
                    out[k][i] = v[j]
    return {k: np.stack(v) if np.ndim(v[0]) else np.array(v) for k, v in out.items()}


def features(fl, iv, z2, bs=256):
    """fl, iv (B, 7781) native; z2 (B, 2) -> raw features (B, 2, F)"""
    out = []
    for a in range(0, len(fl), bs):
        F_ = np.stack([np.interp(LAM_OBS, LAM_NATIVE, f) for f in fl[a:a + bs]])
        V_ = np.stack([np.interp(LAM_OBS, LAM_NATIVE, np.maximum(v, 0)) for v in iv[a:a + bs]])
        x = torch.from_numpy(np.stack([F_ * np.sqrt(V_), np.sqrt(V_)], 1).astype(np.float32))
        zb = torch.from_numpy(np.asarray(z_to_bin(z2[a:a + bs].ravel())).reshape(-1, 2)).long()
        out.append(E.candidate_features(E.spectrum_maps(x, sky=0.3), zb, slog=False, pairs=True).numpy())
    return np.concatenate(out)


def score(f):
    """f (..., F) -> score and its parts"""
    g = lambda n: f[..., ix[n]]
    v_oiii = (g('pairok:oiii') > 0) & (g('narrow:[OIII]b') >= 8) & (np.abs(g('pair:oiii')) > 5)
    v_oii = (g('dblok:oii') > 0) & (np.maximum(g('dbl_b:oii'), g('dbl_r:oii')) >= 8) & (g('dbl_short:oii') < -5)
    v_hab = (g('pairok:ha_hb') > 0) & (g('pair:ha_hb') < -5)
    brk = (g('break_valid') > 0) & (g('break') >= 6)
    n6, un, con = g('n_emis>=6'), g('n_unexpl>=8'), g('n_abs<=-3')
    veto = v_oiii.astype(int) + v_oii + v_hab
    s = n6 + brk - un - con - 2 * veto
    nb5 = sum((g('broad:' + l) >= 5) for l in ['Lya', 'NV', 'SiIV', 'CIV', 'CIII]', 'MgII']) + \
        sum((g('narrow:' + l) >= 5) for l in ['Hb', 'Ha'])
    foii = (g('dbl_b:oii') + g('dbl_r:oii')) / np.sqrt(2)
    return dict(score=s, n6=n6, brk=brk, unexpl=un, contra=con, veto=veto, v_oiii=v_oiii, v_oii=v_oii, v_hab=v_hab,
                nbroad5=nb5, foii=foii, best=g('best_emis'))


def verdict(sr, sm):
    d = sm - sr
    return np.where((d >= 2) & (sm >= 2), 'Redrock wrong', np.where((d <= -2) & (sr >= 2), 'model wrong', 'undecided'))


def main():
    res, ids, out = sys.argv[1:4]
    d, I = np.load(res), np.load(ids)
    zt, zp, cls = d['zt'].astype(np.float64), d['zp'].astype(np.float64), d['cls']
    from grids import CLASSES
    pc = prio(cls, list(CLASSES))
    dz = (zp - zt) / (1 + zt)
    err = np.where(np.abs(dz) > 0.1)[0]
    rng = np.random.default_rng(0)
    ctrl, dec = [], []
    for c in PRIO:
        ok = np.where((np.abs(dz) < 0.003) & (pc == c))[0]
        e = err[pc[err] == c]
        if not len(e):
            continue
        for i in rng.choice(ok, min(800, len(ok)), replace=False):
            j = rng.choice(e)
            zd = (1 + zt[i]) * (1 + zp[j]) / (1 + zt[j]) - 1
            if z_in_grid(np.array([zd]))[0]:
                ctrl.append(i); dec.append(zd)
    ctrl, dec = np.array(ctrl), np.array(dec)
    allr = np.concatenate([err, ctrl])
    print('errors %d, controls %d' % (len(err), len(ctrl)), flush=True)
    R = read_rows(I['src'][allr], I['row'][allr], ['BRZ_FLUX', 'BRZ_IVAR', 'TARGETID', 'DELTACHI2', 'SPECTYPE'])
    z2 = np.stack([zt[allr], np.concatenate([zp[err], dec])], 1)
    f = features(R['BRZ_FLUX'], R['BRZ_IVAR'], z2)
    S_rr, S_m = score(f[:, 0]), score(f[:, 1])
    v = verdict(S_rr['score'], S_m['score'])
    dchi = R['DELTACHI2']
    st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in R['SPECTYPE']])
    elg_ok = np.log10(np.maximum(S_rr['foii'], 1e-3)) > 0.9 - 0.2 * np.log10(np.maximum(dchi, 1e-3))
    ne = len(err)
    isE = np.arange(len(allr)) < ne
    print('\nCONTROLS (model = Redrock; decoy z from a real error ratio): how often the rule accuses Redrock falsely')
    for c in PRIO:
        m = ~isE & (pc[allr] == c)
        if m.any():
            print('  %-4s n %4d | Redrock wrong %5.1f%%  model(decoy) wrong %5.1f%%  undecided %5.1f%%'
                  % (c, m.sum(), *[100 * (v[m] == k).mean() for k in ['Redrock wrong', 'model wrong', 'undecided']]))
    print('\nCATASTROPHIC ERRORS (|dz|/(1+z) > 0.1): who is right')
    for c in PRIO + ['all']:
        m = isE & ((pc[allr] == c) if c != 'all' else True)
        if not m.any():
            continue
        cnt = {k: int((v[m] == k).sum()) for k in ['Redrock wrong', 'model wrong', 'undecided']}
        print('  %-4s n %4d | Redrock wrong %4d (%5.1f%%)  model wrong %4d (%5.1f%%)  undecided %4d (%5.1f%%) | '
              'DELTACHI2<25 %5.1f%% | Redrock SPECTYPE %s'
              % (c, m.sum(), cnt['Redrock wrong'], 100 * cnt['Redrock wrong'] / m.sum(), cnt['model wrong'],
                 100 * cnt['model wrong'] / m.sum(), cnt['undecided'], 100 * cnt['undecided'] / m.sum(),
                 100 * (dchi[m] < 25).mean(), dict(zip(*np.unique(st[m], return_counts=True)))))
    m = isE & (pc[allr] == 'ELG')
    print('\nELG errors x DESI ELG criterion (Raichoor+23, FOII_SNR at the Redrock z):')
    for k in ['Redrock wrong', 'model wrong', 'undecided']:
        mm = m & (v == k)
        print('  %-13s n %4d | DESI criterion says Redrock z reliable: %5.1f%%  | median [OII] sigma at Redrock z %.1f, at model z %.1f'
              % (k, mm.sum(), 100 * elg_ok[mm].mean() if mm.any() else 0, np.median(S_rr['foii'][mm]) if mm.any() else 0,
                 np.median(S_m['foii'][mm]) if mm.any() else 0))
    m = isE & (pc[allr] == 'QSO')
    print('\nQSO errors: broad lines >= 5 sigma (QuasarNet-style) at each z:')
    for k in ['Redrock wrong', 'model wrong', 'undecided']:
        mm = m & (v == k)
        if mm.any():
            print('  %-13s n %4d | median broad lines at Redrock z %.0f, at model z %.0f | model z > 2: %4.1f%%  Redrock z > 2: %4.1f%%'
                  % (k, mm.sum(), np.median(S_rr['nbroad5'][mm]), np.median(S_m['nbroad5'][mm]),
                     100 * (zp[allr][mm] > 2).mean(), 100 * (zt[allr][mm] > 2).mean()))
    np.savez(out + '.npz', idx=allr, is_err=isE, cls=pc[allr], z_rr=zt[allr], z_m=z2[:, 1], verdict=v, tid=R['TARGETID'],
             src=I['src'][allr], row=I['row'][allr], dchi=dchi, spectype=st, elg_desi_ok=elg_ok,
             **{'rr_' + k: np.asarray(x) for k, x in S_rr.items()}, **{'m_' + k: np.asarray(x) for k, x in S_m.items()})
    print('->', out + '.npz')


if __name__ == '__main__':
    main()
