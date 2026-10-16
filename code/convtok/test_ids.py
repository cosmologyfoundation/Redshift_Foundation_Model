"""Recover (file, source row, TARGETID) for every spectrum of a ztrans2 test stream, in the exact order
gather_eval concatenated them, so the saved predictions (results/*_test.npz) can be tied to real objects.

Same Spectra arguments and DataLoader settings as ztrans2 (seed 0, val, native, world 4, workers 8, bs 32,
drop_last, stop once a rank has n_test/world); only _item returns ids instead of flux.  The block order and the
class-mix queues do not depend on _item, so the stream is the same.  Checked: the Redrock Z of every recovered
row must equal the npz 'zt' exactly.
usage: srun -n4 python test_ids.py RESULT_NPZ TEST_BLOCKS N_TEST OUT_NPZ   (one rank per task, SLURM_PROCID)
       python test_ids.py RESULT_NPZ TEST_BLOCKS N_TEST OUT_NPZ merge
"""
import os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from data import Spectra

WORLD, BS, NW = 4, 32, 8


class Ids(Spectra):
    def _item(self, fl_i, iv_i, t_i, z_i, name, sk_i):
        return torch.tensor(z_i, dtype=torch.float64), torch.tensor(0 if name == 'main' else 1), torch.tensor(sk_i), torch.tensor(t_i)


def one(args):
    rank, tb, n_test = args
    ds = Ids(1 << 22, seed=0, val=True, rank=rank, world=WORLD, clean=0, native=True, val_blocks_main=tb)
    Z, S, R, T, n = [], [], [], [], 0
    for z, s, r, t in DataLoader(ds, batch_size=BS, num_workers=NW, drop_last=True):
        if n >= n_test // WORLD:
            break
        Z.append(z.numpy()); S.append(s.numpy()); R.append(r.numpy()); T.append(t.numpy()); n += len(z)
    return [np.concatenate(v) for v in (Z, S, R, T)]


if __name__ == '__main__':
    res, tb, n_test, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if len(sys.argv) < 6:
        r = int(os.environ['SLURM_PROCID'])
        np.savez(out + '.part%d.npz' % r, *one((r, tb, n_test))); sys.exit()
    parts = [[np.load(out + '.part%d.npz' % r)['arr_%d' % i] for i in range(4)] for r in range(WORLD)]
    z, src, row, tbits = [np.concatenate([p_[i] for p_ in parts]) for i in range(4)]
    d = np.load(res)
    assert len(z) == len(d['zt']), (len(z), len(d['zt']))
    same = np.float32(z) == np.float32(d['zt'])
    print('recovered %d rows; Redrock Z identical to the npz for %d (%.4f%%)' % (len(z), same.sum(), 100 * same.mean()))
    assert same.all()
    np.savez(out, src=src, row=row, z=z, tbits=tbits)
    print('->', out)
