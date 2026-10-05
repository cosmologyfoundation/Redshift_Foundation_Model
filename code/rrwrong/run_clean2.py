#!/usr/bin/env python
"""One command for the clean-2 filter on the two ZWARN = 0 training files, written as two separate files.

  input                                         rows        output
  dr1_main_primary_tgt_zwarn0_brz.h5      18,659,804        dr1_main_primary_tgt_zwarn0_clean2_brz.h5
  dr1_sv3_primary_tgt_zwarn0_brz.h5          992,821        dr1_sv3_primary_tgt_zwarn0_clean2_brz.h5

A row is deleted if ANY of (DESIGN.md 6d-2, 6d-3):
  rule A     another independent ZWARN=0 coadd disagrees by > 0.01(1+z) | Gaia star-like & Z > 0.01 |
             star-only target & Z > 0.1
  DCHI2      DELTACHI2 < 25
  P_WRONG    rrwrong_model_v1 P_WRONG > 0.01, PROGRAM dark/bright only, except a Redrock STAR at |Z| < 0.005
             that Gaia also calls star-like (backup is exempt: VI has no backup-condition spectra)

Stages, each multiprocessing on one CPU node; a stage is skipped when its outputs exist (--force reruns):
  1  flags     clean/clean_labels.py      Pool(64) over zall-pix-iron chunks   -> zfm_clean_{main,sv3}.npz
  2  P_WRONG   rrwrong/score_training.py  Pool(64), OMP_NUM_THREADS=2/worker   -> zfm_rrwrong_{main,sv3}.npz
  3  write     rrwrong/make_clean2_h5.py  Pool(120) workers compress chunks    -> the two clean2 h5 files
               (sv3 then main; written to <out>.tmp and renamed only after verify passes)
The source files are only opened read-only.

Run on a CPU node, never a login node:
  salloc --no-shell -q interactive -C cpu -N 1 -t 4:00:00 -A m5374
  srun --jobid=JOBID --overlap -N1 -n1 -c 128 python run_clean2.py [--force] [--only main|sv3] [--dry]
Timing measured 2026-09-28: stage 1 ~3 min, stage 2 ~4 min, stage 3 sv3 ~3 min + main ~47 min.
"""
import argparse, os, subprocess, sys, time
P = '/pscratch/sd/s/shunth/'
H = os.path.expanduser('~/claude_learning_map/')
NAMES = ('sv3', 'main')
OUT = {k: P + 'dr1_%s_primary_tgt_zwarn0_clean2_brz.h5' % k for k in NAMES}
SRC = {k: P + 'dr1_%s_primary_tgt_zwarn0_brz.h5' % k for k in NAMES}

def stage(n, title, outputs, cmd, cwd, env=None, force=False, dry=False):
    have = all(os.path.exists(o) for o in outputs)
    print('[stage %d] %-8s %s' % (n, title, 'outputs exist -> skip' if have and not force else 'RUN'), flush=True)
    for o in outputs:
        print('           %s %s' % ('ok ' if os.path.exists(o) else '-- ', o))
    if (have and not force) or dry:
        return
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if r.returncode:
        sys.exit('[stage %d] FAILED (exit %d)' % (n, r.returncode))
    print('[stage %d] done in %.0f s' % (n, time.time() - t), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='rerun every stage even if its outputs exist')
    ap.add_argument('--only', choices=NAMES, help='write only this file in stage 3')
    ap.add_argument('--dry', action='store_true', help='report what would run, run nothing')
    a = ap.parse_args()
    for k in NAMES:
        assert os.path.exists(SRC[k]), 'missing input ' + SRC[k]
    names = [a.only] if a.only else list(NAMES)
    if a.force and not a.dry:
        for k in names:                      # stage 3 renames over the old file only after verify; remove a stale tmp
            if os.path.exists(OUT[k] + '.tmp'): os.remove(OUT[k] + '.tmp')
    stage(1, 'flags', [P + 'zfm_clean_%s.npz' % k for k in NAMES],
          [sys.executable, 'clean_labels.py'], H + 'clean', force=a.force, dry=a.dry)
    stage(2, 'P_WRONG', [P + 'zfm_rrwrong_%s.npz' % k for k in NAMES],
          [sys.executable, 'score_training.py'], H + 'rrwrong', env={'OMP_NUM_THREADS': '2'}, force=a.force, dry=a.dry)
    stage(3, 'write', [OUT[k] for k in names],
          [sys.executable, 'make_clean2_h5.py'] + names, H + 'rrwrong', force=a.force, dry=a.dry)
    if not a.dry:
        import h5py
        for k in names:
            with h5py.File(OUT[k], 'r') as f:
                print('%-4s %s: %d of %d rows kept, CLEAN_VERSION %s, file %.1f GB'
                      % (k, os.path.basename(OUT[k]), f['SRC_ROW'].shape[0], int(f.attrs['CLEAN_ROWS_SOURCE']),
                         f.attrs.get('CLEAN_VERSION'), os.path.getsize(OUT[k]) / 1e9))
