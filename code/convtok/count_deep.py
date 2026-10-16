"""How many never-trained spectra have a much DEEPER independent observation of the same object?
(owner 2026-10-13: 「先數深淺配對有多少」)  Idea: shallow spectrum = the row in our h5 (what the model and Redrock both see),
truth = Redrock on a deeper, independent coadd (other SURVEY/PROGRAM, so no shared exposures), ZWARN == 0, fiberstatus 0.

Depth = TSNR2 of the object's class template (QSO -> TSNR2_QSO, ELG -> TSNR2_ELG, LRG -> TSNR2_LRG, BGS / STAR -> TSNR2_BGS),
ratio = TSNR2(deep coadd) / TSNR2(our coadd).  Counted at ratio >= 1.5, 2, 3, 4.
Pools (never trained on): A = the 200k test (ids npz); B = every usable row of the 4,400 main + 234 sv3 validation blocks
(test + early-stop pools, priority class known).
usage: python count_deep.py IDS_NPZ
"""
import sys
import numpy as np, h5py, fitsio
sys.path.insert(0, __import__('os').path.expanduser('~/claude_learning_map/zfm'))
from data import survey_fractions, prio_class, PRIO, BLK, P, FILES
from grids import z_in_grid

ZALL = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'
TS = {'QSO': 'TSNR2_QSO', 'ELG': 'TSNR2_ELG', 'LRG': 'TSNR2_LRG', 'BGS': 'TSNR2_BGS', 'STAR': 'TSNR2_BGS'}
s_ = lambda a: np.array([x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in a])


def pool_rows(ids):
    """(src, row) for pool A and pool B"""
    I = np.load(ids)
    A = (I['src'], I['row'])
    fr, rows = survey_fractions()
    src, row = [], []
    for s, (name, nb) in enumerate([('main', 4400), ('sv3', max(1, int(round(4400 * fr['sv3'] / fr['main']))))]):
        vb = np.sort(np.random.default_rng(777).permutation(rows[name] // BLK)[:nb])
        r = np.concatenate([np.arange(b * BLK, min((b + 1) * BLK, rows[name])) for b in vb])
        src.append(np.full(len(r), s)); row.append(r)
    return A, (np.concatenate(src), np.concatenate(row))


def info(src, row):
    n = len(row); out = {k: np.empty(n, object) for k in ['tid', 'surv', 'prog', 'cls', 'ts', 'ok']}
    for s, (fn, tcol) in enumerate([FILES['main'], FILES['sv3']]):
        m = np.where(src == s)[0]
        if not len(m):
            continue
        o = np.argsort(row[m]); r = row[m][o]; ii = m[o]
        with h5py.File(P + fn, 'r') as h:
            tid = h['TARGETID'][:][r]
            surv = np.full(len(r), 'main' if s == 0 else 'sv3')          # each file holds one survey
            prog = np.empty(len(r), object)                                  # PROGRAM read block by block (strings)
            for b in np.unique(r // BLK):
                mb = (r // BLK) == b
                prog[mb] = s_(h['PROGRAM'][b * BLK:(b + 1) * BLK])[r[mb] - b * BLK]
            pc = np.array(PRIO + ['none'])[prio_class(h[tcol][:][r])]
            ok = (h['COADD_FIBERSTATUS'][:][r] == 0) & z_in_grid(h['Z'][:][r]) & (pc != 'none')
            ts = np.zeros(len(r))
            for c, col in TS.items():
                mm = pc == c
                ts[mm] = h[col][:][r[mm]]
        for k, v in [('tid', tid), ('surv', surv), ('prog', prog), ('cls', pc), ('ts', ts), ('ok', ok)]:
            out[k][ii] = v
    return {k: (v.astype(np.int64) if k == 'tid' else v.astype(float) if k == 'ts' else v.astype(bool) if k == 'ok' else v.astype(str))
            for k, v in out.items()}


def main():
    A, B = pool_rows(sys.argv[1])
    pools = {'A: 200k test': info(*A), 'B: all validation blocks': info(*B)}
    want = np.unique(np.concatenate([p['tid'][p['ok']] for p in pools.values()]))
    f = fitsio.FITS(ZALL)[1]; N = f.get_nrows(); keep = []
    for a in range(0, N, 2_000_000):
        t = f.read_column('TARGETID', rows=np.arange(a, min(a + 2_000_000, N)))
        keep.append(a + np.where(np.isin(t, want))[0])
    keep = np.concatenate(keep)
    Z = f.read(rows=keep, columns=['TARGETID', 'SURVEY', 'PROGRAM', 'ZWARN', 'COADD_FIBERSTATUS'] + sorted(set(TS.values())))
    zs, zp = s_(Z['SURVEY']), s_(Z['PROGRAM'])
    by = {}
    for j in range(len(Z)):
        by.setdefault(int(Z['TARGETID'][j]), []).append(j)
    print('zall rows for these objects: %d' % len(Z))
    for pn, p in pools.items():
        m = np.where(p['ok'])[0]
        best = np.zeros(len(p['tid'])); inv = np.zeros(len(p['tid']))
        for i in m:
            for j in by.get(int(p['tid'][i]), []):
                if (zs[j] == p['surv'][i] and zp[j] == p['prog'][i]) or Z['COADD_FIBERSTATUS'][j] != 0:
                    continue
                o = Z[TS[p['cls'][i]]][j]
                if p['ts'][i] > 0 and Z['ZWARN'][j] == 0:
                    best[i] = max(best[i], o / p['ts'][i])
                if o > 0:                              # OUR (primary, ZWARN=0) coadd is the deep one, the other is shallow
                    inv[i] = max(inv[i], p['ts'][i] / o)
        print('\n%s: %d usable spectra' % (pn, len(m)))
        print('  %-5s %8s | deeper independent coadd with TSNR2 ratio >= 1.5 / 2 / 3 / 4' % ('class', 'n'))
        for c in PRIO + ['all']:
            mm = m if c == 'all' else m[p['cls'][m] == c]
            print('  %-5s %8d | %6d  %6d  %6d  %6d | ours deeper by >= 1.5 / 2 / 3 / 4: %6d  %6d  %6d  %6d'
                  % (c, len(mm), *[(best[mm] >= k).sum() for k in (1.5, 2, 3, 4)], *[(inv[mm] >= k).sum() for k in (1.5, 2, 3, 4)]))


if __name__ == '__main__':
    main()
