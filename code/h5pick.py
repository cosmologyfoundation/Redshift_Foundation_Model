"""Draw non-repeating samples from the two DR1 BRZ h5 files.

No-repeat holds in three senses, each checked rather than assumed:

  within a draw   -- TARGETID is unique inside each file (verified: 0 duplicates
                     in 18,659,804 and 992,821 rows).
  across files    -- the two files share 0 TARGETIDs, although 87% of sv3's
                     healpix are also in main. ZCAT_PRIMARY marks one row per
                     TARGETID across the whole catalogue, so an object observed
                     in both surveys lands in exactly one file.
  across draws    -- the same seen list the cohort builder uses, keyed on
                     TARGETID, so the two paths cannot collide.

The same ZCAT_PRIMARY that makes pooling safe also makes these files unusable
for a BLIND cohort: the *_PRIMARY flags rank by ZWARN, so drawing a blind test
here would let the reference pick the sample. --blind is therefore refused.
"""
import os, sys, json, argparse
import numpy as np
import h5py

P = '/pscratch/sd/s/shunth/'
FILES = {'main': P + 'dr1_main_primary_tgt_zwarn0_brz.h5',
         'sv3': P + 'dr1_sv3_primary_tgt_zwarn0_brz.h5'}
SEEN = P + 'claude_learning_map/seen_targets.json'
BITS = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS': 60, 'STAR': 61}
TGTCOL = {'main': 'DESI_TARGET', 'sv3': 'SV3_DESI_TARGET'}


def load(files, by, need):
    """Read only the columns needed for selection."""
    out = []
    for k in files:
        with h5py.File(FILES[k], 'r') as h:
            d = {'tid': h['TARGETID'][:], 'file': k}
            if by == 'spectype':
                d['cls'] = np.array([s.decode().strip() if isinstance(s, bytes)
                                     else str(s).strip()
                                     for s in h['SPECTYPE'][:]])
            else:
                d['cls'] = h[TGTCOL[k]][:]
            for c in need:
                d[c] = h[c][:]
            d['row'] = np.arange(len(d['tid']))
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='per class')
    ap.add_argument('--classes', default='STAR,BGS,LRG,ELG,QSO')
    ap.add_argument('--files', default='main,sv3')
    ap.add_argument('--by', default='target', choices=('target', 'spectype'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--blind', action='store_true')
    a = ap.parse_args()

    if a.blind:
        raise SystemExit(
            'These files are built with ZCAT_PRIMARY == True and ZWARN == 0. '
            'The *_PRIMARY flags rank by ZWARN, so a blind cohort drawn here '
            'would have its sample chosen by the reference. Draw blind cohorts '
            'with make_cohort3.py from the zcatalog, which dedups on '
            'TSNR2_<class>.')

    files = a.files.split(',')
    classes = a.classes.split(',')
    seen = set(int(x) for x in json.load(open(SEEN))['targetids'])
    print('seen list holds %d TARGETIDs' % len(seen), flush=True)

    parts = load(files, a.by, [])
    rng = np.random.default_rng(a.seed)
    picked, rows = [], []
    for c in classes:
        cand = []
        for d in parts:
            if a.by == 'spectype':
                m = d['cls'] == c
            else:
                m = (d['cls'] & (1 << BITS[c])) != 0
            m &= ~np.isin(d['tid'], list(seen))
            idx = np.flatnonzero(m)
            cand.append((d['file'], d['tid'][idx], idx))
        tot = sum(len(t) for _, t, _ in cand)
        if tot < a.n:
            raise SystemExit('class %s: only %d unseen candidates, need %d'
                             % (c, tot, a.n))
        flat = [(f, int(t), int(i)) for f, ts, ix in cand
                for t, i in zip(ts, ix)]
        take = rng.choice(len(flat), a.n, replace=False)
        for j in take:
            f, t, i = flat[j]
            picked.append(t)
            rows.append(dict(cls=c, file=f, row=i, targetid=t))
        print('  %-5s %d unseen candidates -> drew %d' % (c, tot, a.n),
              flush=True)

    # the three no-repeat guarantees, asserted rather than trusted
    assert len(picked) == len(set(picked)), 'duplicate TARGETID inside the draw'
    assert not (set(picked) & seen), 'draw overlaps the seen list'
    print('\nchecks: %d drawn, %d unique, 0 overlap with the seen list'
          % (len(picked), len(set(picked))))

    if a.dry_run:
        print('--dry-run: nothing written')
        return
    book = json.load(open(SEEN))
    book['targetids'] += picked
    book['sessions'].append(dict(id=len(book['sessions']) + 1, n=len(picked),
                                 note='h5pick %s %s %d/class by %s'
                                      % (a.files, a.classes, a.n, a.by)))
    tmp = SEEN + '.tmp'
    json.dump(book, open(tmp, 'w'), indent=1)
    os.replace(tmp, SEEN)          # seen list first, then the payload
    print('seen list now %d' % len(book['targetids']))
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
