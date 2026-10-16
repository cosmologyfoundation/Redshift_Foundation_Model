"""Does Redrock disagree with ITSELF?  Owner 2026-10-11: 「我要怎麼確認 redrock 也不一定是對的」 -> 先做第 1 項.

For every test object, the label is Redrock's Z from one coadd: (SURVEY, PROGRAM) of its row in our h5.  DESI DR1
(zall-pix-iron) also holds Redrock's answer for the SAME TARGETID from other healpix coadds, one per (SURVEY, PROGRAM).
A coadd only contains exposures of its own survey/program, so two different (SURVEY, PROGRAM) coadds share no
exposure: two independent Redrock measurements of one object.

  repeat     another coadd with COADD_FIBERSTATUS == 0 and ZWARN == 0 (Redrock itself called it confident)
  conflict   such a repeat whose Z differs from our label by |dz|/(1+z) > 0.1
             -> at least one of two CONFIDENT Redrock answers is wrong.  No model, no template, no physics involved.
usage: python repeat_z.py IDS_NPZ OUT_NPZ
"""
import sys
import numpy as np, h5py, fitsio

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']
ZALL = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'
s_ = lambda a: np.char.strip(np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a).ravel()], dtype='U16'))


def main():
    ids, out = sys.argv[1:3]
    I = np.load(ids)
    src, row = I['src'], I['row']
    n = len(row)
    tid, surv, prog, z = np.zeros(n, np.int64), np.empty(n, object), np.empty(n, object), np.zeros(n)
    for s in (0, 1):
        m = src == s
        idx = np.where(m)[0]; o = np.argsort(row[idx]); rr = row[idx][o]          # h5 fancy index must increase
        with h5py.File(P + FN[s], 'r') as h:
            for k, dst in [('TARGETID', tid), ('SURVEY', surv), ('PROGRAM', prog), ('Z', z)]:
                v = h[k][rr]
                dst[idx[o]] = s_(v) if v.dtype == object else v
    assert np.allclose(z, I['z']), 'label Z mismatch'
    surv, prog = s_(surv), s_(prog)
    print('test objects %d, unique TARGETID %d' % (n, len(np.unique(tid))), flush=True)
    f = fitsio.FITS(ZALL)[1]
    N = f.get_nrows(); want = set(tid.tolist())
    keep = []
    for a in range(0, N, 2_000_000):
        t = f.read_column('TARGETID', rows=np.arange(a, min(a + 2_000_000, N)))
        keep.append(a + np.where(np.isin(t, tid))[0])
    keep = np.concatenate(keep)
    Zc = f.read(rows=keep, columns=['TARGETID', 'SURVEY', 'PROGRAM', 'Z', 'ZWARN', 'COADD_FIBERSTATUS', 'DELTACHI2', 'SPECTYPE'])
    print('zall rows for these TARGETIDs: %d' % len(Zc), flush=True)
    pos = {t: i for i, t in enumerate(tid)}
    has_rep = np.zeros(n, bool); conflict = np.zeros(n, bool); n_rep = np.zeros(n, np.int32)
    z_other = np.full(n, np.nan); same_found = np.zeros(n, bool)
    zs, zp, zw, zf = s_(Zc['SURVEY']), s_(Zc['PROGRAM']), Zc['ZWARN'], Zc['COADD_FIBERSTATUS']
    for j in range(len(Zc)):
        i = pos[int(Zc['TARGETID'][j])]
        if zs[j] == surv[i] and zp[j] == prog[i]:
            same_found[i] = True                      # our own coadd: must reproduce the label
            continue
        if zw[j] != 0 or zf[j] != 0:
            continue
        has_rep[i] = True; n_rep[i] += 1
        dz = abs(Zc['Z'][j] - z[i]) / (1 + z[i])
        if dz > 0.1 and (not conflict[i] or dz > abs(z_other[i] - z[i]) / (1 + z[i])):
            conflict[i] = True; z_other[i] = Zc['Z'][j]
        elif not conflict[i] and np.isnan(z_other[i]):
            z_other[i] = Zc['Z'][j]
    print('own coadd found in zall: %.2f%%' % (100 * same_found.mean()))
    print('with an independent confident (ZWARN=0) repeat: %d (%.2f%%)' % (has_rep.sum(), 100 * has_rep.mean()))
    print('Redrock vs Redrock conflict (|dz|/(1+z) > 0.1): %d = %.3f%% of all, %.2f%% of those with a repeat'
          % (conflict.sum(), 100 * conflict.mean(), 100 * conflict[has_rep].mean()))
    np.savez(out, tid=tid, survey=surv, program=prog, z=z, has_rep=has_rep, n_rep=n_rep, conflict=conflict, z_other=z_other)
    print('->', out)


if __name__ == '__main__':
    main()
