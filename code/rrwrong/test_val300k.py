"""Checks for train.py --val_blocks_main / --n_val / --clean_val (2026-10-05), before a 1.5M clean-2 run
with a 300k clean-2 validation on 4 GPUs (DDP, --bs 8, global batch 32).

  1  the validation pool at --val_blocks_main V holds >= N_VAL rows, every rank shard >= N_VAL/WORLD
  2  every validation row lies in a validation block and is kept by clean-2
  3  the 4 rank shards are disjoint; the old 400-block clean-2 validation set is a subset of the new one
  4  training draws (4 ranks): never in a validation block, never a clean-2-deleted row, no TARGETID shared
     with validation, ranks disjoint
  5  defaults unchanged: val_blocks 400, unfiltered validation
Usage: python test_val300k.py [V=1300]    (CPU node; reads spectra, ~10 min)
"""
import os, sys, time, tempfile, numpy as np, h5py
import multiprocessing as mp
V = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
N_VAL, WORLD, NTRAIN = 300000, 4, 40000
t0 = time.time()

def collect(T, tag, kw, nw):
    from torch.utils.data import DataLoader
    from data import Spectra
    ds = Spectra(**kw); ds._emit_src = True
    out = [(0 if n == 'main' else 1, r) for _, n, r in DataLoader(ds, batch_size=None, num_workers=nw)]
    np.save(f'{T}/{tag}.npy', np.array(out, dtype=np.int64).reshape(-1, 2))

if __name__ == '__main__':
    T = tempfile.mkdtemp(prefix='val300k_', dir='/pscratch/sd/s/shunth/')
    from data import Spectra, P, FILES, FILES_CLEAN2, BLK
    jobs = [(f'val_r{r}', dict(n_target=1 << 20, seed=0, val=True, rank=r, world=WORLD, val_blocks_main=V, clean=2), 12)
            for r in range(WORLD)]
    jobs += [('val_old400', dict(n_target=1 << 20, seed=0, val=True, val_blocks_main=400, clean=2), 12)]
    jobs += [(f'tr_r{r}', dict(n_target=NTRAIN, seed=42, rank=r, world=WORLD, val_blocks_main=V, clean=2), 8)
             for r in range(WORLD)]
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=collect, args=(T,) + j) for j in jobs]
    for p in ps: p.start()
    for p in ps: p.join(); assert p.exitcode == 0, 'a collector failed'
    L = {j[0]: np.load(f'{T}/{j[0]}.npy') for j in jobs}
    key = lambda a: set(map(tuple, a.tolist()))
    names = list(FILES)                                   # 0 = main, 1 = sv3
    nrows, kept2, vblocks, tid = {}, {}, {}, {}
    probe = Spectra(10, val=True, val_blocks_main=V)
    for i, k in enumerate(names):
        with h5py.File(P + FILES[k][0], 'r') as f:
            nrows[k] = f['TARGETID'].shape[0]; tid[k] = f['TARGETID'][:]
        with h5py.File(P + FILES_CLEAN2[k], 'r') as h:
            m = np.zeros(nrows[k], bool); m[h['SRC_ROW'][:]] = True; kept2[k] = m
        vblocks[k] = set(np.random.default_rng(777).permutation(nrows[k] // BLK)[:probe.val_blocks[k]].tolist())
    print('V=%d -> validation blocks: main %d, sv3 %d' % (V, probe.val_blocks['main'], probe.val_blocks['sv3']))
    # 1-3 validation
    shards = [key(L[f'val_r{r}']) for r in range(WORLD)]
    sizes = [len(s) for s in shards]
    for a in range(WORLD):
        for b in range(a + 1, WORLD):
            assert not (shards[a] & shards[b]), 'validation shards %d and %d overlap' % (a, b)
    val = set().union(*shards)
    print('validation pool %d rows (main %d, sv3 %d); per-rank %s' % (len(val), sum(1 for f, _ in val if f == 0),
          sum(1 for f, _ in val if f == 1), sizes))
    assert len(val) >= N_VAL, 'validation pool %d < %d: raise --val_blocks_main' % (len(val), N_VAL)
    assert min(sizes) >= -(-N_VAL // WORLD), 'a rank shard is below N_VAL/WORLD'
    for f, r in val:
        k = names[f]
        assert r // BLK in vblocks[k], 'validation row outside the validation blocks'
        assert kept2[k][r], 'validation row deleted by clean-2'
    old = key(L['val_old400'])
    assert old <= val, '%d rows of the old 400-block validation set are missing' % len(old - val)
    print('old 400-block clean-2 validation (%d rows) is a subset of the new pool' % len(old))
    # 4 training
    trs = [key(L[f'tr_r{r}']) for r in range(WORLD)]
    for a in range(WORLD):
        for b in range(a + 1, WORLD):
            assert not (trs[a] & trs[b]), 'training ranks %d and %d overlap' % (a, b)
    tr = set().union(*trs)
    vt = {int(tid[names[f]][r]) for f, r in val}
    for f, r in tr:
        k = names[f]
        assert r // BLK not in vblocks[k], 'TRAINING ROW IN A VALIDATION BLOCK'
        assert kept2[k][r], 'training row deleted by clean-2'
        assert int(tid[k][r]) not in vt, 'training TARGETID also in validation'
    print('training draws %d (main %d, sv3 %d): none in validation blocks, none deleted, no shared TARGETID'
          % (len(tr), sum(1 for f, _ in tr if f == 0), sum(1 for f, _ in tr if f == 1)))
    # 5 defaults
    d = Spectra(10, val=True)
    assert d.val_blocks['main'] == 400 and not d.clean, 'defaults changed'
    print('defaults unchanged (val_blocks main 400, unfiltered)')
    import shutil; shutil.rmtree(T)
    print('ALL CHECKS PASSED in %.0f s' % (time.time() - t0))
