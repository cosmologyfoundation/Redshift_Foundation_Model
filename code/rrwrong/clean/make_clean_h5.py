"""Write a copy of a training h5 with likely-wrong labels deleted (owner, 2026-09-27).

Rule (owner's choice), from zfm_clean_<name>.npz (clean_labels.py): delete a row if ANY of
  another independent ZWARN=0 coadd of the same target disagrees by > 0.01(1+z)
  Gaia star-like (catalogue PM/parallax vs an error from G) and Z > 0.01
  star-only target and Z > 0.1
  DELTACHI2 < 25
The SOURCE FILE IS ONLY EVER OPENED READ-ONLY.  Kept rows stay in source order; SRC_ROW is each
row's index in the source.  Every dataset, dtype, chunk shape and attribute is copied.

Speed: the three spectrum arrays (BRZ_FLUX/IVAR/MASK, ~97% of the bytes) are compressed IN THE
WORKERS and written with write_direct_chunk, so the single writer only moves bytes.  They use
DEFLATE level 1 instead of the source's LZF, because zlib is in the standard library and LZF's
compressor is not exposed to Python; the stored values are bit-identical (verified below).
The first version compressed LZF in the one writer and ran at ~650 rows/s (8 h for main).
The copy is written to <out>.tmp and renamed only after verify() passes.
"""
import os, sys, time, zlib, numpy as np, h5py
from multiprocessing import Pool
P = '/pscratch/sd/s/shunth/'
SRC = {'main': 'dr1_main_primary_tgt_zwarn0_brz.h5', 'sv3': 'dr1_sv3_primary_tgt_zwarn0_brz.h5'}
DST = {'main': 'dr1_main_primary_tgt_zwarn0_clean_brz.h5', 'sv3': 'dr1_sv3_primary_tgt_zwarn0_clean_brz.h5'}
RULE = ('delete if: repeat coadd disagrees > 0.01(1+z) | Gaia star-like & Z > 0.01 | '
        'star-only target & Z > 0.1 | DELTACHI2 < 25')
BIG = ('BRZ_FLUX', 'BRZ_IVAR', 'BRZ_MASK')
CH = 64                      # rows per chunk, as in the source
GROUP = 32                   # chunks per worker task (2,048 output rows)
t0 = time.time()
log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
G = {}

def keep_mask(name):
    c = np.load(P + 'zfm_clean_%s.npz' % name)
    assert c['found'].all(), 'unmatched rows'
    bad = ((c['rep_dis'] > 0) | (c['gaia_star'] & (c['z'] > 0.01)) | (c['star_only'] & (c['z'] > 0.1))
           | (c['dchi2'] < 25))
    return ~bad

def init(name, kept):
    G['f'] = h5py.File(P + SRC[name], 'r')
    G['kept'] = kept

def task(c0):
    kept, f = G['kept'], G['f']
    r0, r1 = c0 * CH, min((c0 + GROUP) * CH, len(kept))
    src = kept[r0:r1]
    a, b = int(src[0]), int(src[-1]) + 1
    out = []
    for k in BIG:
        blk = f[k][a:b][src - a]
        for j in range(0, len(blk), CH):
            ch = blk[j:j + CH]
            if len(ch) < CH:                               # edge chunk: stored full-size
                ch = np.concatenate([ch, np.zeros((CH - len(ch),) + ch.shape[1:], ch.dtype)])
            out.append((k, r0 + j, zlib.compress(np.ascontiguousarray(ch).tobytes(), 1)))
    return out

def read_col(args):
    name, k = args
    with h5py.File(P + SRC[name], 'r') as f:
        return k, f[k][:]

def main(name):
    keep = keep_mask(name)
    kept = np.flatnonzero(keep).astype(np.int64)
    n_src, n_out = len(keep), len(kept)
    log('%s: %d rows -> keep %d, delete %d (%.3f%%)' % (name, n_src, n_out, n_src - n_out, 100 * (1 - keep.mean())))
    tmp = P + DST[name] + '.tmp'
    with h5py.File(P + SRC[name], 'r') as src:
        keys = list(src.keys())
        small = [k for k in keys if k not in BIG]
        with h5py.File(tmp, 'w') as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs['CLEAN_SOURCE'] = P + SRC[name]
            dst.attrs['CLEAN_RULE'] = RULE
            dst.attrs['CLEAN_ROWS_SOURCE'] = n_src
            dst.attrs['CLEAN_ROWS_KEPT'] = n_out
            dst.attrs['CLEAN_FLAGS_FILE'] = P + 'zfm_clean_%s.npz' % name
            dst.attrs['CLEAN_UTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            dst.attrs['CLEAN_NOTE'] = ('owner 2026-09-27: any Redrock field may delete training rows; validation '
                                       'is still drawn from the ORIGINAL file (map via SRC_ROW; DESIGN 6d-2). '
                                       'BRZ_* use deflate-1 here (LZF in the source); values identical.')
            for k in keys:
                d = src[k]
                comp = ('gzip', 1) if k in BIG else (d.compression, d.compression_opts)
                dst.create_dataset(k, shape=(n_out,) + d.shape[1:], dtype=d.dtype, chunks=d.chunks,
                                   compression=comp[0], compression_opts=comp[1])
                for ak, av in d.attrs.items():
                    dst[k].attrs[ak] = av
                if k in BIG:
                    assert d.chunks == (CH, d.shape[1]), (k, d.chunks)
            dst.create_dataset('SRC_ROW', data=kept, chunks=(8192,))
            dst['SRC_ROW'].attrs['MEANING'] = 'row index of this spectrum in ' + P + SRC[name]
    # catalogue columns: read whole, index, write (small)
    with Pool(32) as pool:
        cols = dict(pool.map(read_col, [(name, k) for k in small]))
    with h5py.File(tmp, 'r+') as dst:
        for k in small:
            dst[k][:] = cols[k][kept]
    del cols
    log('%s: %d catalogue columns written' % (name, len(small)))
    starts = list(range(0, -(-n_out // CH), GROUP))
    done = nb = 0
    with h5py.File(tmp, 'r+') as dst, Pool(120, initializer=init, initargs=(name, kept)) as pool:
        ids = {k: dst[k].id for k in BIG}
        for res in pool.imap_unordered(task, starts, chunksize=1):
            for k, off, data in res:
                ids[k].write_direct_chunk((off, 0), data)
                nb += len(data)
            done += 1
            if done % max(1, len(starts) // 20) == 0:
                el = time.time() - t0
                log('%s: %d/%d tasks, %.1f GB compressed written, %.0f rows/s' % (name, done, len(starts), nb / 1e9,
                    done * GROUP * CH / el))
    verify(name, tmp, keep, kept)
    os.replace(tmp, P + DST[name])
    log('%s: DONE -> %s (%d rows, %.1f GB)' % (name, P + DST[name], n_out, os.path.getsize(P + DST[name]) / 1e9))

def verify(name, tmp, keep, kept):
    rng = np.random.default_rng(0)
    with h5py.File(P + SRC[name], 'r') as s, h5py.File(tmp, 'r') as d:
        assert set(s.keys()) | {'SRC_ROW'} == set(d.keys())
        assert np.array_equal(d['SRC_ROW'][:], kept), 'SRC_ROW mismatch'
        assert np.array_equal(d['TARGETID'][:], s['TARGETID'][:][keep]), 'TARGETID order mismatch'
        assert (d['DELTACHI2'][:] >= 25).all(), 'a DELTACHI2 < 25 row survived'
        idx = np.sort(np.concatenate([rng.choice(len(kept), 500, replace=False), [0, len(kept) - 1]]))
        for i in idx:
            for k in s.keys():
                a, b = s[k][kept[i]], d[k][i]
                ok = np.array_equal(a, b) or (np.asarray(a).dtype.kind == 'f' and np.array_equal(a, b, equal_nan=True))
                assert ok, ('value mismatch', k, int(i))
        nk = len(list(s.keys()))
    log('%s: verify passed -- SRC_ROW, TARGETID order, DELTACHI2 >= 25, and %d rows (incl. first/last) identical '
        'in all %d datasets' % (name, len(idx), nk))

if __name__ == '__main__':
    for name in sys.argv[1:]:
        main(name)
