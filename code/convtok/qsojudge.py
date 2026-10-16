"""QSO broad-line judge (owner 2026-10-13: 「先做 QSO 寬線判定」).  Which side do the broad lines support?

For each QSO-target catastrophic of ZT5m40 (200k test) and for controls, at the label z (Redrock) and at the model z:
  broad lines  Lya 1215.67, NV 1240.81, SiIV+OIV] 1399.8, CIV 1549.48, CIII] 1908.73, MgII 2799.49, Hb 4862.68, Ha 6564.61
  each line: weighted least squares on the native pixels within +-15,000 km/s of the expected centre,
             flux = a + b x + A G(lambda),  G Gaussian of FWHM 4,000 km/s, centre scanned over -1,500..+1,500 km/s
             (QSO lines are shifted, CIV is blueshifted), S = A / sigma_A at the best shift.
             counted only if >= 70% of the +-6,000 km/s core is measured.
  n_det(z)    lines with S >= 4
  verdict     label wrong (evidence for the model z): n_det(model) >= 2 and n_det(label) == 0
              model wrong (evidence for the label z): n_det(label) >= 2 and n_det(model) == 0
              otherwise: no broad-line evidence either way -> counts as a model error (owner rule 2026-10-11)
Two lines are required because one broad line alone does not fix z (MgII vs Hb vs CIII] ...).
Controls: QSO targets where model and label agree (|dz| < 0.003), each with a decoy "model z" built from a real error's
(1+z_model)/(1+z_label).  "label wrong" on a control is a false accusation; its rate is the rule's error rate.
usage: python qsojudge.py TEST_NPZ IDS_NPZ OUT_PREFIX
"""
import os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from grids import CLASSES, z_in_grid

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
LAM = np.arange(7781) * 0.8 + 3600.0
C = 299792.458
LINES = [('Lya', 1215.67), ('NV', 1240.81), ('SiIV', 1399.8), ('CIV', 1549.48), ('CIII]', 1908.73),
         ('MgII', 2799.49), ('Hb', 4862.68), ('Ha', 6564.61)]
SIG_V = 4000 / 2.3548
SHIFTS = np.array([-1500, -750, 0, 750, 1500])
PRIO = ['QSO', 'LRG', 'ELG', 'BGS', 'STAR']


def line_S(f, w, lam0):
    """best S over the shift grid, or nan if not covered"""
    lo, hi = lam0 * (1 - 15000 / C), lam0 * (1 + 15000 / C)
    if hi < LAM[0] or lo > LAM[-1]:
        return np.nan
    m = (LAM >= lo) & (LAM <= hi)
    core = (LAM >= lam0 * (1 - 6000 / C)) & (LAM <= lam0 * (1 + 6000 / C))
    if core.sum() < 20 or (w[core] > 0).mean() < 0.7:
        return np.nan
    x, y, ww = LAM[m], f[m], w[m]
    ok = ww > 0
    x, y, ww = x[ok], y[ok], ww[ok]
    xn = (x - lam0) / lam0
    best = -np.inf
    for dv in SHIFTS:
        c = lam0 * (1 + dv / C); s = c * SIG_V / C
        G = np.exp(-0.5 * ((x - c) / s) ** 2)
        X = np.stack([np.ones_like(x), xn, G], 1)
        A = X.T @ (X * ww[:, None])
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        beta = cov @ (X.T @ (ww * y))
        if cov[2, 2] > 0:
            best = max(best, beta[2] / np.sqrt(cov[2, 2]))
    return best if np.isfinite(best) else np.nan


def judge(f, w, z):
    S = np.array([line_S(f, w, l * (1 + z)) for _, l in LINES])
    return S


def main():
    tnpz, ids, out = sys.argv[1:4]
    d, I = np.load(tnpz), np.load(ids)
    zt, zp, cls = d['zt'].astype(float), d['zp'].astype(float), d['cls']
    prio = np.full(len(zt), 'none', object)
    for nm in PRIO[::-1]:
        prio[cls[:, CLASSES.index(nm)] > 0] = nm
    dz = np.abs(zp - zt) / (1 + zt)
    err = np.where((prio == 'QSO') & (dz > 0.1))[0]
    rng = np.random.default_rng(0)
    okq = rng.choice(np.where((prio == 'QSO') & (dz < 0.003))[0], 1500, replace=False)
    ctrl, dec = [], []
    for i in okq:
        j = rng.choice(err); zd = (1 + zt[i]) * (1 + zp[j]) / (1 + zt[j]) - 1
        if z_in_grid(np.array([zd]))[0]:
            ctrl.append(i); dec.append(zd)
    ctrl, dec = np.array(ctrl), np.array(dec)
    allr = np.concatenate([err, ctrl]); zb = np.concatenate([zp[err], dec]); isE = np.arange(len(allr)) < len(err)
    print('QSO errors %d, controls %d' % (len(err), len(ctrl)), flush=True)
    F = np.zeros((len(allr), 7781), np.float32); W = np.zeros_like(F)
    tid = np.zeros(len(allr), np.int64); st = np.empty(len(allr), object); dchi = np.zeros(len(allr))
    for s in (0, 1):
        sel = np.where(I['src'][allr] == s)[0]
        r = I['row'][allr][sel]; o = np.argsort(r)
        with h5py.File(P + FN[s], 'r') as h:
            for j, k in zip(sel[o], r[o]):
                F[j] = h['BRZ_FLUX'][k]; W[j] = np.maximum(h['BRZ_IVAR'][k], 0)
                tid[j] = h['TARGETID'][k]; x = h['SPECTYPE'][k]; st[j] = x.decode() if isinstance(x, bytes) else str(x)
                dchi[j] = h['DELTACHI2'][k]
    St = np.stack([judge(F[i], W[i], zt[allr][i]) for i in range(len(allr))])
    Sm = np.stack([judge(F[i], W[i], zb[i]) for i in range(len(allr))])
    nt, nm_ = (np.nan_to_num(St, nan=-9) >= 4).sum(1), (np.nan_to_num(Sm, nan=-9) >= 4).sum(1)
    v = np.where((nm_ >= 2) & (nt == 0), 'label wrong', np.where((nt >= 2) & (nm_ == 0), 'model wrong', 'no evidence'))
    for nm, m in [('CONTROLS (decoy as model z)', ~isE), ('QSO CATASTROPHICS', isE)]:
        k = {x: int((v[m] == x).sum()) for x in ['label wrong', 'model wrong', 'no evidence']}
        print('%-28s n %4d | label wrong %4d (%5.1f%%) | model wrong %4d (%5.1f%%) | no evidence %4d (%5.1f%%)'
              % (nm, m.sum(), k['label wrong'], 100 * k['label wrong'] / m.sum(), k['model wrong'], 100 * k['model wrong'] / m.sum(),
                 k['no evidence'], 100 * k['no evidence'] / m.sum()))
    m = isE
    print('\nerrors by Redrock SPECTYPE:')
    for t in ['GALAXY', 'QSO', 'STAR']:
        mm = m & (st == t)
        if mm.any():
            print('  %-6s n %4d | label wrong %4d  model wrong %4d  no evidence %4d' % (t, mm.sum(), (v[mm] == 'label wrong').sum(),
                  (v[mm] == 'model wrong').sum(), (v[mm] == 'no evidence').sum()))
    print('\nbroad lines >= 4 sigma, errors: at label z median %.0f (>=1: %.1f%%), at model z median %.0f (>=1: %.1f%%)'
          % (np.median(nt[m]), 100 * (nt[m] >= 1).mean(), np.median(nm_[m]), 100 * (nm_[m] >= 1).mean()))
    n_q = int((prio == 'QSO').sum())
    lw = int((m & (v == 'label wrong')).sum())
    print('\nQSO catastrophic as counted: %d / %d = %.3f%%;  with the %d evidence-backed label errors removed: %d = %.3f%%'
          % (m.sum(), n_q, 100 * m.sum() / n_q, lw, m.sum() - lw, 100 * (m.sum() - lw) / n_q))
    np.savez(out + '.npz', idx=allr, is_err=isE, z_label=zt[allr], z_model=zb, verdict=v, S_label=St, S_model=Sm,
             tid=tid, spectype=st.astype(str), dchi=dchi, src=I['src'][allr], row=I['row'][allr], lines=np.array([l[0] for l in LINES]))
    print('->', out + '.npz')


if __name__ == '__main__':
    main()
