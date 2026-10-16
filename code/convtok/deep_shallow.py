"""Model vs Redrock on the SAME shallow spectrum, scored against a deeper observation (owner 2026-10-13: 「試試看」).

Objects: every usable row of the never-trained validation blocks (main 4,400 + sv3 234), all 5 priority classes.
  deep   = our h5 row: DESI's primary coadd of the object (ZWARN = 0).  Truth = its Redrock Z.
  shallow= another healpix coadd of the same TARGETID from a different SURVEY/PROGRAM (no shared exposures),
           COADD_FIBERSTATUS 0, with TSNR2(class) at most 1/2 of the deep one; the shallowest such coadd is used.
Both contestants see only the shallow coadd:
  Redrock: its own answer for that coadd from zall-pix-iron (Z, ZWARN) -- not re-run.
  model:   ZT5m40 (best checkpoint) on the shallow coadd, B/R/Z combined onto the 7,781-px grid by speclib.coadd_cameras
           (the same coaddition the h5 files were built with; checked below on deep coadds re-read from the files).
Catastrophic = |z - z_truth| / (1 + z_truth) > 0.1.  Reported per class and depth ratio, and paired
(both right / only Redrock right / only model right / both wrong).  Redrock's ZWARN on the shallow coadd is shown too:
Redrock can flag a doubtful answer, the model cannot, so the table also splits on it.
usage: python deep_shallow.py CKPT OUT_PREFIX [MIN_RATIO]
"""
import os, sys, time
import numpy as np, h5py, fitsio, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map'))
import speclib
from count_deep import pool_rows, info, TS, s_, ZALL
from data import PRIO

P = '/pscratch/sd/s/shunth/'
FN = ['dr1_main_primary_tgt_zwarn0_brz.h5', 'dr1_sv3_primary_tgt_zwarn0_brz.h5']


def pairs(ids, min_ratio):
    _, B = pool_rows(ids)
    p = info(*B)
    m = np.where(p['ok'])[0]
    tid = p['tid'][m]
    f = fitsio.FITS(ZALL)[1]; N = f.get_nrows(); keep = []
    for a in range(0, N, 2_000_000):
        t = f.read_column('TARGETID', rows=np.arange(a, min(a + 2_000_000, N)))
        keep.append(a + np.where(np.isin(t, tid))[0])
    keep = np.concatenate(keep)
    Z = f.read(rows=keep, columns=['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'Z', 'ZWARN', 'SPECTYPE', 'COADD_FIBERSTATUS'] + sorted(set(TS.values())))
    zs, zp = s_(Z['SURVEY']), s_(Z['PROGRAM'])
    by = {}
    for j in range(len(Z)):
        by.setdefault(int(Z['TARGETID'][j]), []).append(j)
    out = []
    for i in m:
        best, bj = 0, -1
        for j in by.get(int(p['tid'][i]), []):
            if (zs[j] == p['surv'][i] and zp[j] == p['prog'][i]) or Z['COADD_FIBERSTATUS'][j] != 0:
                continue
            o = Z[TS[p['cls'][i]]][j]
            if o > 0 and p['ts'][i] / o > best:
                best, bj = p['ts'][i] / o, j
        if bj >= 0 and best >= min_ratio:
            out.append((i, bj, best))
    src, row = B
    I = np.array([o[0] for o in out]); J = np.array([o[1] for o in out])
    return dict(src=src[I], row=row[I], tid=p['tid'][I], cls=p['cls'][I], ratio=np.array([o[2] for o in out]),
                s_survey=zs[J], s_program=zp[J], s_hpix=Z['HEALPIX'][J].astype(int), z_rr_s=Z['Z'][J].astype(float),
                zwarn_s=Z['ZWARN'][J].astype(int), stype_s=s_(Z['SPECTYPE'][J]))


def read_file(args):
    path, tids = args
    try:
        ft = fitsio.read(path, ext='FIBERMAP', columns=['TARGETID'])['TARGETID']
        pos = {int(t): k for k, t in enumerate(ft)}
        rows = [pos.get(int(t), -1) for t in tids]
        ok = [k for k, r in enumerate(rows) if r >= 0]
        if not ok:
            return path, None
        _, fl, iv, _, got = speclib.coadd_cameras(path, rows=np.array([rows[k] for k in ok]))
        return path, (np.array(ok), fl.astype(np.float32), iv.astype(np.float32))
    except Exception as e:
        return path, str(e)


def main():
    ck, out = sys.argv[1:3]; min_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    t0 = time.time()
    D = pairs(os.path.expandvars('$SCRATCH/zt1m_test_ids.npz'), min_ratio)
    n = len(D['tid'])
    print('pairs (shallow TSNR2 <= deep / %.1f): %d | %s  [%.0fs]' % (min_ratio, n, {c: int((D['cls'] == c).sum()) for c in PRIO}, time.time() - t0), flush=True)
    # shallow spectra, file by file
    files = {}
    for k in range(n):
        files.setdefault(speclib.coadd_path(D['s_survey'][k], D['s_program'][k], int(D['s_hpix'][k])), []).append(k)
    FL = np.zeros((n, 7781), np.float32); IV = np.zeros((n, 7781), np.float32); have = np.zeros(n, bool); fails = 0
    with Pool(32) as pool:
        for path, res in pool.imap_unordered(read_file, [(pth, [D['tid'][k] for k in ks]) for pth, ks in files.items()], chunksize=4):
            if res is None or isinstance(res, str):
                fails += 1; continue
            ok, fl, iv = res; ks = np.array(files[path])[ok]
            FL[ks], IV[ks], have[ks] = fl, iv, True
    print('shallow coadds read: %d / %d from %d files (%d files failed)  [%.0fs]' % (have.sum(), n, len(files), fails, time.time() - t0), flush=True)
    # deep spectra from the h5 + truth
    FD = np.zeros((n, 7781), np.float32); VD = np.zeros_like(FD); zt = np.zeros(n)
    for s in (0, 1):
        sel = np.where(D['src'] == s)[0]; o = np.argsort(D['row'][sel]); r = D['row'][sel][o]
        with h5py.File(P + FN[s], 'r') as h:
            zt[sel[o]] = h['Z'][:][r]
            for a in range(0, len(r), 2048):
                FD[sel[o][a:a + 2048]] = h['BRZ_FLUX'][r[a:a + 2048]]; VD[sel[o][a:a + 2048]] = np.maximum(h['BRZ_IVAR'][r[a:a + 2048]], 0)
    # coaddition check: re-read 100 DEEP coadds from the DESI files and compare with the h5 rows
    rng = np.random.default_rng(0); chk = rng.choice(np.where(have)[0], min(100, int(have.sum())), replace=False); dif = []
    for k in chk:
        s, r = int(D['src'][k]), int(D['row'][k])
        with h5py.File(P + FN[s], 'r') as h:
            sv, pg = h['SURVEY'][r], h['PROGRAM'][r]; hp = int(h['HEALPIX'][r])
        sv, pg = (x.decode() if isinstance(x, bytes) else str(x) for x in (sv, pg))
        _, res = read_file((speclib.coadd_path(sv, pg, hp), [D['tid'][k]]))
        if res is None or isinstance(res, str):
            continue
        fl, iv = res[1][0], res[2][0]; w = (iv > 0) & (VD[k] > 0)
        dif.append(np.abs(fl[w] - FD[k][w]).max() / max(np.abs(FD[k][w]).max(), 1e-9))
    print('coaddition check on %d deep coadds: max relative |flux difference| median %.2e, worst %.2e' % (len(dif), np.median(dif), np.max(dif)), flush=True)
    # model
    from ztrans5 import ZTrans5
    from ztrans2 import readz
    dev = 'cuda'
    ckd = torch.load(ck, map_location='cpu', weights_only=False)
    net = ZTrans5(H=32, L=12, lv_n=(5, 5)).to(dev).eval(); net.load_state_dict(ckd['model'])

    def predict(F, V):
        z = np.zeros(len(F))
        with torch.no_grad():
            for a in range(0, len(F), 64):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z[a:a + 64] = readz(net(torch.from_numpy(F[a:a + 64]).to(dev), torch.from_numpy(V[a:a + 64]).to(dev)))
        return z
    zm_s = predict(FL, IV); zm_d = predict(FD, VD)
    print('model predictions done  [%.0fs]' % (time.time() - t0), flush=True)
    cat = lambda z: np.abs(z - zt) / (1 + zt) > 0.1
    rr_bad, m_bad, md_bad = cat(D['z_rr_s']), cat(zm_s), cat(zm_d)
    u = have
    print('\nsanity: model on the DEEP spectrum (h5) vs deep truth: catastrophic %.3f%% (n %d)' % (100 * md_bad[u].mean(), u.sum()))
    print('\n%-5s %6s | %-24s | %-14s | %-9s %-9s %-9s %-9s' % ('class', 'n', 'catastrophic on SHALLOW', 'Redrock ZWARN', 'both ok', 'RR only', 'model only', 'both bad'))
    print('%-5s %6s | %-11s %-12s | %-14s | %-9s %-9s %-9s %-9s' % ('', '', 'Redrock', 'model', '!=0 share', '', 'right', 'right', ''))
    for c in PRIO + ['all']:
        mm = u & ((D['cls'] == c) if c != 'all' else True)
        if not mm.any():
            continue
        a, b = rr_bad[mm], m_bad[mm]
        print('%-5s %6d | %6.2f%% (%3d) %6.2f%% (%3d) | %6.1f%%        | %-9d %-9d %-9d %-9d'
              % (c, mm.sum(), 100 * a.mean(), a.sum(), 100 * b.mean(), b.sum(), 100 * (D['zwarn_s'][mm] != 0).mean(),
                 (~a & ~b).sum(), (~a & b).sum(), (a & ~b).sum(), (a & b).sum()))
    print('\nonly where Redrock did NOT flag the shallow answer (ZWARN == 0):')
    for c in PRIO + ['all']:
        mm = u & (D['zwarn_s'] == 0) & ((D['cls'] == c) if c != 'all' else True)
        if mm.any():
            print('  %-5s n %6d | Redrock %6.2f%% (%3d)  model %6.2f%% (%3d)' % (c, mm.sum(), 100 * rr_bad[mm].mean(), rr_bad[mm].sum(), 100 * m_bad[mm].mean(), m_bad[mm].sum()))
    print('\nby depth ratio (all classes):')
    for lo, hi in [(2, 3), (3, 5), (5, 1e9)]:
        mm = u & (D['ratio'] >= lo) & (D['ratio'] < hi)
        if mm.any():
            print('  ratio %g-%g  n %6d | Redrock %6.2f%%  model %6.2f%%' % (lo, hi if hi < 1e9 else np.inf, mm.sum(), 100 * rr_bad[mm].mean(), 100 * m_bad[mm].mean()))
    np.savez(out + '.npz', **{k: v for k, v in D.items()}, have=have, z_truth=zt, z_model_shallow=zm_s, z_model_deep=zm_d)
    print('->', out + '.npz  [%.0fs]' % (time.time() - t0))


if __name__ == '__main__':
    main()
