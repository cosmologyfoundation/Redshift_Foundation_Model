"""Streaming dataset over both DR1 h5 files, in survey proportion.

Not cached to disk: at 18M spectra a float16 cache is 542 GB.  Random single-row
h5 access is ~130x slower than contiguous (chunks are 64 rows), so this samples
random chunk-aligned BLOCKS and resamples to the log-lambda grid in the worker.
Measured single-threaded: ~1540 spectra/s, so several workers keep up with a
run whose compute is hours.

Labels (option A, chosen 2026-09-08):
  z      Redrock Z.  The only redshift that exists at this scale; DESIGN.md 6c
         records what that costs -- the model inherits Redrock's failure modes,
         and every measured leak is a CONFIDENT line misidentification.
  class  the DESI target bitmask, as a SET.  13.6% of objects carry more than
         one class bit, so this is multi-label and never a softmax.
"""
import collections
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from grids import (LAM_OBS, LAM_NATIVE, CLASSES, BITS, z_to_bin, z_in_grid)

P = '/pscratch/sd/s/shunth/'
FILES = {'main': ('dr1_main_primary_tgt_zwarn0_brz.h5', 'DESI_TARGET'),
         'sv3': ('dr1_sv3_primary_tgt_zwarn0_brz.h5', 'SV3_DESI_TARGET')}
SEEN = P + 'claude_learning_map/seen_targets.json'
BLK = 256
DEEP = P + 'zfm_deep_tail.npz'
PT_EXCL = P + 'zfm_pretrain_exclude_tid.npy'
# Copies with likely-wrong labels deleted (owner 2026-09-27, DESIGN.md 6d-2; clean/make_clean_h5.py).
# Same datasets plus SRC_ROW = the row's index in the ORIGINAL file above.
FILES_CLEAN = {'main': 'dr1_main_primary_tgt_zwarn0_clean_brz.h5',
               'sv3': 'dr1_sv3_primary_tgt_zwarn0_clean_brz.h5'}
# v2 (owner 2026-09-28): v1 rule OR rrwrong P_WRONG > 0.01 (rrwrong/make_clean2_h5.py); strict superset
# of v1's deletions.  clean=True / 1 -> v1, clean=2 -> v2.
FILES_CLEAN2 = {'main': 'dr1_main_primary_tgt_zwarn0_clean2_brz.h5',
                'sv3': 'dr1_sv3_primary_tgt_zwarn0_clean2_brz.h5'}
FILES_CLEAN_BY = {1: FILES_CLEAN, 2: FILES_CLEAN2}

# Class mix (owner 2026-10-09: 「之後抓data都用這個比例抓」 then 「比例等比例分5類就好 各20%」).  Priority class from
# the target bits, QSO(2) > LRG(0) > ELG(1) > BGS_ANY(60) > MWS_ANY(61)=STAR; rows with none of the five bits are not
# drawn.  Every reader keeps its stream at 20% per class: a row whose class is already ahead of its quota is skipped.
# Natural shares for reference (all original ZWARN=0 rows, 19,643,466): QSO 9.90 LRG 15.50 ELG 20.75 BGS 30.97
# STAR 20.08 none 2.80 %, so QSO limits the draw rate (~2x the rows are read per QSO kept).
PRIO = ['QSO', 'LRG', 'ELG', 'BGS', 'STAR']
MIX = {'QSO': 0.2, 'LRG': 0.2, 'ELG': 0.2, 'BGS': 0.2, 'STAR': 0.2, 'none': 0.0}


def prio_class(t):
    """target bitmask array -> priority class index (0..4 = PRIO, 5 = none)"""
    t = np.asarray(t, dtype=np.int64)
    out = np.full(len(t), 5, np.int8)
    for i in range(len(PRIO) - 1, -1, -1):
        out[((t >> BITS[PRIO[i]]) & 1).astype(bool)] = i
    return out


def survey_fractions():
    n = {}
    for k, (fn, _) in FILES.items():
        with h5py.File(P + fn, 'r') as f:
            n[k] = f['BRZ_FLUX'].shape[0]
    t = sum(n.values())
    return {k: v / t for k, v in n.items()}, n


class Spectra(IterableDataset):
    def __init__(self, n_target, seed=0, val=False, val_blocks_main=400, val_from_main=0,
                 rank=0, world=1, zmax=None, deep=None, labelfree=False, clean=False,
                 native=False, mix=True):
        super().__init__()
        # clean, training: draws come from the cleaned copies, and a cleaned row whose SRC_ROW
        # falls in one of the original validation blocks is dropped, so no spectrum is both
        # trained on and validated on.
        # clean, validation (owner 2026-09-28: 「驗證的也要從乾淨的挑」): the SAME original
        # validation blocks, minus the rows the cleaning deleted -- i.e. the rows of those
        # blocks that appear in the cleaned copy's SRC_ROW.  Train and validation stay disjoint.
        # The deep-tail reader addresses ORIGINAL sv3 rows, so it is not combined with clean.
        assert not (clean and deep), 'deep-tail rows index the original sv3 file'
        self.clean = clean
        self.files_clean = FILES_CLEAN_BY[int(clean)] if clean else None
        # labelfree: DESIGN.md 7a pre-training.  No label of any kind enters the
        # selection or the output: the z-in-grid and zmax filters (both read the
        # Redrock Z) are skipped, and z / class come back as zeros.  What stays:
        # ZWARN = 0 (the files themselves -- owner's rule), fiberstatus 0, the
        # fixed validation blocks held out, targets read by eye excluded, and the
        # known-wrong / blind-read test objects excluded by TARGETID.
        assert not (labelfree and (zmax is not None or deep)), 'labelfree takes no label-based option'
        self.labelfree = labelfree
        # native (2026-10-09, FM_DESIGN native input): yield (BRZ_FLUX, BRZ_IVAR) on the 7781-px DESI grid,
        # no interpolation and no S/N conversion; the model does its own normalisation.  BRZ_MASK is not
        # read: on clean-2 every pixel with mask != 0 already has ivar = 0 (mask_audit.py).
        # 2026-10-09: native also serves LABELLED batches for the downstream redshift head (zfm_native.py):
        # (BRZ_FLUX, BRZ_IVAR, z bin, class bits, z) -- same rows, same filters as the log-grid path.
        self.native = native
        self.mix = mix                   # class mix at MIX shares (default on, owner 2026-10-09)
        self.excl = (set(int(t) for t in np.load(PT_EXCL)) if labelfree and os.path.exists(PT_EXCL) else set())
        assert not labelfree or self.excl, 'pre-training exclusion list missing: ' + PT_EXCL
        # deep: DESIGN.md 7b-2.  dict(frac, degrade, teacher) -- with probability
        # frac a training example is REPLACED by one drawn from the SV3 deep tail
        # (deep_tail.py), identically in every arm; degrade=True adds noise to
        # bring it to a main-depth TSNR2 drawn from the same program; teacher=<npz>
        # swaps the label for the teacher's deep-spectrum answer where its
        # evidence is strong.  The three arms differ ONLY in degrade/teacher.
        assert not (val and deep), 'deep draws are for training only'
        self.deep = deep
        # zmax: TRAINING scope restriction on the LABEL z, decided by the owner
        # 2026-09-22 (DESIGN.md 6c).  Never set on the validation stream: the
        # objects above it must still be scored, so its cost stays visible.
        assert not (val and zmax is not None), 'zmax filters training only'
        self.zmax = zmax
        self.n_target = n_target
        self.rank, self.world = rank, world
        self.seed = seed
        self.val = val
        self.frac, self.rows = survey_fractions()
        # Validation must carry the SAME survey proportion as training.
        # An equal BLOCK count per file gives 50/50, and sv3 is 2-4x
        # deeper, so an equal-block validation set is systematically
        # easier than what the model trains on and every loss on it comes
        # out biased low.
        self.val_blocks = {
            'main': val_blocks_main,
            'sv3': max(1, int(round(val_blocks_main
                                    * self.frac['sv3'] / self.frac['main']))),
        }
        self.seen = set(json.load(open(SEEN))['targetids'])
        # val_from_main (2026-10-10, early stopping): a validation stream that SKIPS the first val_from_main main
        # blocks (and sv3 in the same proportion) of the fixed permutation.  With val_blocks_main=1300, val_from_main=400
        # it is disjoint from the default 400-block test pool, and training run with val_blocks_main=1300 never sees it.
        self.val_from = {'main': val_from_main,
                         'sv3': (max(1, int(round(val_from_main * self.frac['sv3'] / self.frac['main']))) if val_from_main else 0)}
        if clean:
            self.rows_clean = {}
            for k, fn in self.files_clean.items():
                with h5py.File(P + fn, 'r') as h:
                    assert 'SRC_ROW' in h and int(h.attrs['CLEAN_ROWS_SOURCE']) == self.rows[k], k
                    self.rows_clean[k] = h['SRC_ROW'].shape[0]
            if val:
                self.kept_orig = {}
                for k, fn in self.files_clean.items():
                    with h5py.File(P + fn, 'r') as h:
                        m = np.zeros(self.rows[k], bool); m[h['SRC_ROW'][:]] = True
                    self.kept_orig[k] = m
            # the original validation blocks, exactly as _blocks(val=True) draws them
            self.val_src_blocks = {k: np.sort(np.random.default_rng(777).permutation(self.rows[k] // BLK)
                                              [:self.val_blocks[k]]) for k in FILES}
        self._emit_src = False           # tests only: also yield (file, source row)

    def _deep_item(self, D, rngd, f):
        k = int(rngd.integers(len(D['rows'])))
        r = int(D['rows'][k])
        prog = str(D['program'][k])
        pool = D['main_tsnr2_' + prog]
        target = float(pool[int(rngd.integers(len(pool)))])     # drawn in EVERY arm
        fl = f['BRZ_FLUX'][r:r + 1][0].astype(np.float64)
        iv = np.maximum(f['BRZ_IVAR'][r:r + 1][0].astype(np.float64), 0)
        tt = int(f[FILES['sv3'][1]][r])
        if self.deep.get('degrade'):
            fac = min(1.0, target / float(D['tsnr2'][k]))
            good = iv > 0
            # noise keyed on (seed, row, draw) so arms A and B add IDENTICAL noise
            g = np.random.default_rng([self.seed, r, k]).standard_normal(len(fl))
            fl = fl + np.where(good, g * np.sqrt((1 - fac) / (fac * np.where(good, iv, 1))), 0)
            iv = iv * fac
        z = float(D['z_lab'][k])
        F_ = np.interp(LAM_OBS, LAM_NATIVE, fl)
        V_ = np.interp(LAM_OBS, LAM_NATIVE, iv)
        x = np.stack([F_ * np.sqrt(V_), np.sqrt(V_)]).astype(np.float32)
        c = np.array([(tt >> BITS[kk]) & 1 for kk in CLASSES], dtype=np.float32)
        return (torch.from_numpy(x), torch.tensor(int(z_to_bin(z))),
                torch.from_numpy(c), torch.tensor(z))

    def _blocks(self, name, rng):
        """Block ids for this split.  Validation takes a FIXED set of blocks,
        the same for every run of every size, so losses are comparable across
        the whole scaling grid -- a defect of the encoder A/B that is not
        repeated here."""
        if self.clean and not self.val:
            # every block of the cleaned copy; validation rows are removed per row in __iter__
            nb = self.rows_clean[name] // BLK
            return rng.permutation(nb)
        nb = self.rows[name] // BLK
        vb = np.random.default_rng(777).permutation(nb)[:self.val_blocks[name]]
        if self.val:
            return vb[self.val_from[name]:]
        keep = np.setdiff1d(np.arange(nb), vb)
        return keep[rng.permutation(len(keep))]

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        # ONE rng for every reader, so every reader builds the SAME block order
        # and the stride below cuts it into disjoint shards.  The first version
        # seeded per worker (seed*1000 + wid): each worker shuffled differently
        # and then took its own stride, so workers overlapped -- about 63% of
        # blocks reached by the union, some twice, some never.  Invisible at
        # 30k out of 18M, but at the 3M and 18M points the DISTINCT data seen
        # is smaller than the nominal D, which is exactly the axis L(D) is
        # fitted on.
        gid, gn = self.rank * nw + wid, self.world * nw
        rng = np.random.default_rng(self.seed * 1000)
        tr_clean = self.clean and not self.val          # training on the cleaned copies
        handles = {k: h5py.File(P + (self.files_clean[k] if tr_clean else v[0]), 'r') for k, v in FILES.items()}
        nrows = self.rows_clean if tr_clean else self.rows
        order = []
        for name in FILES:
            share = self.frac[name] if not self.val else 1.0
            for b in self._blocks(name, rng):
                order.append((name, int(b)))
        rng.shuffle(order)
        order = order[gid::gn]
        made = 0
        cap = -(-self.n_target // gn)      # ceil, so short shards do not starve
        D, rngd = None, None
        # mix may be a dict of class shares (owner 2026-10-11: ELG/QSO fine-tune at 35/35/10/10/10); True = MIX
        mw = self.mix if isinstance(self.mix, dict) else MIX
        cnt = np.zeros(6, np.int64); MIXV = [mw.get(c, 0.0) for c in PRIO + ['none']]
        Q = [collections.deque() for _ in range(6)]; QCAP = 512
        if self.deep:
            D = dict(np.load(DEEP))
            if self.deep.get('teacher'):
                T = np.load(self.deep['teacher'])
                assert np.array_equal(T['rows'], D['rows']), 'teacher file is for another deep list'
                use = T['strong'] & (T['z_teacher'] < 1.6)
                D['z_lab'] = np.where(use, T['z_teacher'], D['z_rr'])
            else:
                D['z_lab'] = D['z_rr']
            # the SAME stream of draws in every arm: seeded by seed and shard only
            rngd = np.random.default_rng([self.seed, 99, gid])
        for name, b in order:
            if made >= cap:
                break
            f = handles[name]
            tcol = FILES[name][1]
            a, e = b * BLK, min((b + 1) * BLK, nrows[name])
            tid = f['TARGETID'][a:e]
            fs = f['COADD_FIBERSTATUS'][a:e]
            z = f['Z'][a:e]
            tg = f[tcol][a:e]
            src_rows = f['SRC_ROW'][a:e] if tr_clean else np.arange(a, e)
            if tr_clean:
                notval = ~np.isin(src_rows // BLK, self.val_src_blocks[name])
            elif self.clean:                               # clean validation: drop deleted rows
                notval = self.kept_orig[name][a:e]
            else:
                notval = np.ones(e - a, bool)
            if self.labelfree:
                keep = ((fs == 0) & notval & np.array([int(t) not in self.seen and int(t) not in self.excl
                                                        for t in tid]))
            else:
                keep = ((fs == 0) & notval & z_in_grid(z)
                        & ((z < self.zmax) if self.zmax is not None else True)
                        & np.array([int(t) not in self.seen for t in tid]))
            if not keep.any():
                continue
            fl = f['BRZ_FLUX'][a:e][keep].astype(np.float32)
            iv = np.maximum(f['BRZ_IVAR'][a:e][keep].astype(np.float32), 0)
            zz, tt = z[keep], tg[keep]
            pcls = prio_class(tt)
            sk = src_rows[keep]
            for i in range(len(fl)):
                if made >= cap:
                    break
                if D is not None and rngd.random() < self.deep['frac']:
                    made += 1
                    yield self._deep_item(D, rngd, handles['sv3'])
                    continue
                raw = (fl[i].copy(), iv[i].copy(), int(tt[i]), float(zz[i]), name, int(sk[i]))
                if not self.mix:
                    made += 1
                    yield self._item(*raw)
                    continue
                # class mix (2026-10-09 fix): rows wait in a per-class queue and leave in quota order.  The first
                # version skipped a row whose class was ahead of quota; rows are class-clustered within blocks
                # (programs), so ~96% of rows were thrown away and every reader exhausted its blocks at ~12.5k of
                # its 15.6k cap (1M run: 199k per rank instead of 250k).
                pc = int(pcls[i])
                if MIXV[pc] > 0 and len(Q[pc]) < QCAP:
                    Q[pc].append(raw)
            if self.mix:
                while made < cap:
                    c_ = max((k for k in range(6) if MIXV[k] > 0), key=lambda k: MIXV[k] * (made + 1) - cnt[k])
                    if not Q[c_]:
                        break
                    cnt[c_] += 1
                    made += 1
                    yield self._item(*Q[c_].popleft())

    def _item(self, fl_i, iv_i, t_i, z_i, name, sk_i):
        """one output tuple, exactly as the stream always yielded it"""
        if self.native:
            fi, vi = torch.from_numpy(np.ascontiguousarray(fl_i)), torch.from_numpy(np.ascontiguousarray(iv_i))
            if self.labelfree:
                return (fi, vi)
            c = np.array([(t_i >> BITS[k]) & 1 for k in CLASSES], dtype=np.float32)
            return (fi, vi, torch.tensor(int(z_to_bin(z_i))), torch.from_numpy(c), torch.tensor(z_i))
        F_ = np.interp(LAM_OBS, LAM_NATIVE, fl_i)
        V_ = np.interp(LAM_OBS, LAM_NATIVE, iv_i)
        # NOT clipped here: the +-50 clip lives in ZFM.forward, in front of the encoder only (clip_audit.py)
        x = np.stack([F_ * np.sqrt(V_), np.sqrt(V_)]).astype(np.float32)
        if self._emit_src:
            return (torch.from_numpy(x), name, sk_i)
        if self.labelfree:
            return (torch.from_numpy(x), torch.tensor(0), torch.zeros(len(CLASSES)), torch.tensor(0.0))
        c = np.array([(t_i >> BITS[k]) & 1 for k in CLASSES], dtype=np.float32)
        return (torch.from_numpy(x), torch.tensor(int(z_to_bin(z_i))), torch.from_numpy(c), torch.tensor(z_i))

if __name__ == '__main__':
    fr, rows = survey_fractions()
    print('rows      ', rows)
    print('fractions ', {k: round(v, 6) for k, v in fr.items()})
    for split in (False, True):
        ds = Spectra(512, val=split)
        it = iter(ds)
        xs = [next(it) for _ in range(64)]
        x, b, c, z = (torch.stack([e[i] for e in xs]) for i in range(4))
        print('%-5s x %s  zbin %s  class %s  z %.3f-%.3f  multi-bit %d/64'
              % ('val' if split else 'train', tuple(x.shape), tuple(b.shape),
                 tuple(c.shape), z.min(), z.max(), int((c.sum(1) > 1).sum())))
        print('      finite %s  class counts %s'
              % (bool(torch.isfinite(x).all()), c.sum(0).tolist()))
