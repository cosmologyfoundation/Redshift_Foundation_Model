"""ZT5 on SV3 only (owner 2026-10-11: 「用 sv3 測 ZT5 各類錯誤率」).

Rows: every SV3 row in the SV3 TEST blocks -- the first round(2200 x frac_sv3/frac_main) = 117 blocks of the fixed
validation permutation (seed 777), i.e. exactly the SV3 part of the pool the 200k test was drawn from.  Never trained on
(training excludes the 234 SV3 validation blocks), never used for early stopping (that used SV3 blocks 117-234).
No class mix: SV3's own proportions.  Same row filters as data.py: COADD_FIBERSTATUS 0, Redrock Z on the grid, not a
target read by eye.  Labels: Redrock Z of the SV3 coadd (original ZWARN=0 file, no cleaning).
Also counted: how many of these TARGETIDs also sit in the MAIN file's TRAINING blocks (same object, other exposures).
usage: python eval_sv3.py CKPT OUT_PREFIX
"""
import json, os, sys
import numpy as np, h5py, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/claude_learning_map/zfm'))
from data import survey_fractions, prio_class, PRIO, BLK, SEEN, P, FILES
from grids import z_in_grid
from ztrans5 import ZTrans5
from ztrans2 import readz, nmad

TEST_MAIN = 2200


def main():
    ck, out = sys.argv[1:3]
    fr, rows = survey_fractions()
    nb_test = max(1, int(round(TEST_MAIN * fr['sv3'] / fr['main'])))
    nb_val_all = max(1, int(round(4400 * fr['sv3'] / fr['main'])))
    vb = np.random.default_rng(777).permutation(rows['sv3'] // BLK)[:nb_test]
    seen = set(json.load(open(SEEN))['targetids'])
    fn, tcol = FILES['sv3']
    h = h5py.File(P + fn, 'r')
    R = np.concatenate([np.arange(b * BLK, min((b + 1) * BLK, rows['sv3'])) for b in np.sort(vb)])
    z = h['Z'][:][R]; fs = h['COADD_FIBERSTATUS'][:][R]; tid = h['TARGETID'][:][R]; tg = h[tcol][:][R]
    prog = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in h['PROGRAM'][:][R]])
    keep = (fs == 0) & z_in_grid(z) & np.array([int(t) not in seen for t in tid])
    R, z, tid, tg, prog = R[keep], z[keep], tid[keep], tg[keep], prog[keep]
    pc = np.array(PRIO + ['none'])[prio_class(tg)]
    print('SV3 test blocks %d (of %d SV3 validation blocks): %d rows after filters' % (nb_test, nb_val_all, len(R)), flush=True)
    # same object in MAIN training blocks?
    with h5py.File(P + FILES['main'][0], 'r') as hm:
        mt = hm['TARGETID'][:]
    vbm = np.random.default_rng(777).permutation(rows['main'] // BLK)[:4400]
    in_train = np.ones(rows['main'], bool)
    for b in vbm:
        in_train[b * BLK:(b + 1) * BLK] = False
    leak = np.isin(tid, mt[in_train])
    print('  same TARGETID also in MAIN training blocks: %d (%.1f%%)' % (leak.sum(), 100 * leak.mean()), flush=True)

    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    net = ZTrans5(H=32, L=12, lv_n=(5, 5)).to(dev).eval()
    net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    zp = np.zeros(len(R))
    o_ = np.argsort(R); bs = 64
    with torch.no_grad():
        for a in range(0, len(R), bs):
            ii = o_[a:a + bs]
            fl = torch.from_numpy(h['BRZ_FLUX'][R[ii]].astype(np.float32)).to(dev)
            iv = torch.from_numpy(np.maximum(h['BRZ_IVAR'][R[ii]], 0).astype(np.float32)).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                zp[ii] = readz(net(fl, iv))
            if a % (bs * 100) == 0:
                print('  %d / %d' % (a, len(R)), flush=True)
    d = (zp - z) / (1 + z); bad = np.abs(d) > 0.1
    res = dict(n=len(z), catastrophic=float(bad.mean()), n_catastrophic=int(bad.sum()), nmad=float(nmad(d)),
               n_main_train_overlap=int(leak.sum()), per_class={}, per_program={})
    print('\nZT5 on SV3 test blocks (natural SV3 proportions)')
    print('  %-6s %7s %6s %9s %10s | %s' % ('class', 'n', 'share', 'catastr.', 'NMAD', 'catastrophic excluding objects also in main training'))
    for c in PRIO + ['none', 'all']:
        m = (pc == c) if c != 'all' else np.ones(len(z), bool)
        if not m.any():
            continue
        r = dict(n=int(m.sum()), n_catastrophic=int(bad[m].sum()), catastrophic=float(bad[m].mean()), nmad=float(nmad(d[m])),
                 catastrophic_no_overlap=float(bad[m & ~leak].mean()) if (m & ~leak).any() else None)
        res['per_class'][c] = r
        print('  %-6s %7d %5.1f%% %6.3f%% (%4d) %10.3e | %s' % (c, r['n'], 100 * m.mean(), 100 * r['catastrophic'], r['n_catastrophic'],
              r['nmad'], '%.3f%% of %d' % (100 * r['catastrophic_no_overlap'], (m & ~leak).sum()) if r['catastrophic_no_overlap'] is not None else '-'))
    for p in np.unique(prog):
        m = prog == p
        res['per_program'][p] = dict(n=int(m.sum()), catastrophic=float(bad[m].mean()))
        print('  program %-7s n %6d catastrophic %.3f%%' % (p, m.sum(), 100 * bad[m].mean()))
    json.dump(res, open(out + '.json', 'w'), indent=1)
    np.savez(out + '.npz', row=R, tid=tid, z=z, zp=zp, cls=pc, program=prog, main_train_overlap=leak)
    print('->', out + '.json')


if __name__ == '__main__':
    main()
