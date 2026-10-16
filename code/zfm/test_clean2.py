"""Checks for the v2 clean copies (clean=2): no deleted row (v1 rule OR P_WRONG > 0.01) and no validation
row is ever drawn for training; values match the original rows; clean validation = original validation
blocks minus v2-deleted rows, disjoint from training; v1 and the default path still open their own files."""
import numpy as np, h5py, time
from torch.utils.data import DataLoader
import data as Dm, sys, os
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/rrwrong'))
from make_clean2_h5 import rr_delete
from data import Spectra, P, FILES, FILES_CLEAN, FILES_CLEAN2, BLK
t0 = time.time()
def bad_v2(k):
    c = np.load(P + 'zfm_clean_%s.npz' % k); r = np.load(P + 'zfm_rrwrong_%s.npz' % k)
    return ((c['rep_dis'] > 0) | (c['gaia_star'] & (c['z'] > 0.01)) | (c['star_only'] & (c['z'] > 0.1))
            | (c['dchi2'] < 25) | rr_delete(k, r['p_wrong']))
ds = Spectra(40000, seed=5, clean=2); ds._emit_src = True
got = {'main': [], 'sv3': []}; X = {}
for x, name, sr in DataLoader(ds, batch_size=None, num_workers=16):
    got[name].append(sr); X[(name, sr)] = x
for k in FILES:
    sr = np.array(got[k]); bad = bad_v2(k)
    assert not np.isin(sr // BLK, ds.val_src_blocks[k]).any(), 'VALIDATION ROW DRAWN from ' + k
    assert not bad[sr].any(), 'DELETED ROW DRAWN from ' + k
    with h5py.File(P + FILES_CLEAN2[k], 'r') as h:
        assert int(h.attrs['CLEAN_VERSION']) == 2
        assert np.array_equal(np.flatnonzero(~bad), h['SRC_ROW'][:]), 'SRC_ROW != rows the v2 rule keeps: ' + k
    with h5py.File(P + FILES[k][0], 'r') as f:
        for r in np.random.default_rng(1).choice(sr, min(50, len(sr)), replace=False):
            fl = f['BRZ_FLUX'][r].astype(np.float32); iv = np.maximum(f['BRZ_IVAR'][r].astype(np.float32), 0)
            F_ = np.interp(Dm.LAM_OBS, Dm.LAM_NATIVE, fl); V_ = np.interp(Dm.LAM_OBS, Dm.LAM_NATIVE, iv)
            x0 = np.stack([F_ * np.sqrt(V_), np.sqrt(V_)]).astype(np.float32)
            assert np.array_equal(x0, X[(k, int(r))].numpy()), ('value mismatch', k, int(r))
    print('%s: %d drawn, 0 validation rows, 0 deleted rows, SRC_ROW == v2 keep set, 50 spectra identical' % (k, len(sr)))
res = {}
for clean in (0, 2):
    v = Spectra(1 << 20, seed=0, val=True, clean=clean); v._emit_src = True
    res[clean] = set((n, r) for _, n, r in DataLoader(v, batch_size=None, num_workers=8))
a, b = res[0], res[2]
bad = {k: bad_v2(k) for k in FILES}
assert b <= a and all(not bad[n][r] for n, r in b) and all(bad[n][r] for n, r in a - b), 'clean-2 validation wrong'
tr = set((n, r) for n, rows in got.items() for r in rows)
assert not (tr & b), 'train/validation overlap'
print('validation: original %d, clean2 %d (removed %.2f%%), = original minus v2-deleted, disjoint from training' %
      (len(a), len(b), 100 * (1 - len(b) / len(a))))
assert Spectra(10, clean=1).files_clean is FILES_CLEAN and Spectra(10, clean=True).files_clean is FILES_CLEAN
assert Spectra(10).files_clean is None
print('clean=True/1 -> v1 files, default -> originals; done in %.0fs' % (time.time() - t0))
