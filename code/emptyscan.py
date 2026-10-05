"""Is 'nothing anywhere above N sigma' a usable predictor of a lost redshift?
Measured on every cohort that has both a cohort.npz and a scored z."""
import glob, json, os
import numpy as np
import speclib as SL

rows = []
for f in sorted(glob.glob('/pscratch/sd/s/shunth/claude_learning_map/*/data/scored.npz')):
    d = np.load(f, allow_pickle=True)
    if 'z' not in d.files or 'zref' not in d.files:
        continue
    cf = f.replace('scored', 'cohort')
    co = np.load(cf, allow_pickle=True)
    coh = f.split('/')[-3]
    cid = [str(x) for x in co['id']]
    for k, oid in enumerate([str(x) for x in d['id']]):
        i = cid.index(oid)
        fl, iv = co['flux'][i], co['ivar'][i]
        s70 = SL.significance(fl, iv, 70.0)
        s3000 = SL.significance(fl, iv, 3000.0)
        step = SL.local_step_significance(fl, iv)
        good = iv > 0
        rows.append(dict(
            cohort=coh, id=oid, tid=int(d['tid'][k]),
            n70=float(np.nanmax(s70)), n3000=float(np.nanmax(s3000)),
            nstep=float(np.nanmax(np.abs(step))),
            medflux=float(np.nanmedian(np.where(good, fl, np.nan))),
            dz=float(abs(d['z'][k] - d['zref'][k])),
            z=float(d['z'][k]), zref=float(d['zref'][k]),
            tclass=str(d['tclass'][k]) if 'tclass' in d.files else '?'))
    print(coh, len(rows))

json.dump(rows, open(os.path.expanduser('~/claude_learning_map/emptyscan.json'), 'w'))
n = len(rows)
best = np.array([max(r['n70'], r['n3000'], r['nstep']) for r in rows])
ok = np.array([r['dz'] < 0.1 for r in rows])
print('\n%d objects with a redshift claim' % n)
print('\n  strongest feature ANYWHERE (max of narrow / broad / |step|)')
print('  %-14s %6s %8s %9s' % ('bin', 'n', 'z right', 'rate'))
edges = [0, 4, 6, 8, 12, 20, 1e9]
for a, b in zip(edges[:-1], edges[1:]):
    m = (best >= a) & (best < b)
    if m.sum():
        print('  %-14s %6d %8d %8.1f%%'
              % ('%g - %g' % (a, b), m.sum(), ok[m].sum(), 100 * ok[m].mean()))
for thr in (4, 5, 6, 8):
    m = best < thr
    print('  below %2d sigma : n=%3d  right %3d  = %5.1f%%   |   above: n=%3d  right %3d = %5.1f%%'
          % (thr, m.sum(), ok[m].sum(), 100 * ok[m].mean() if m.sum() else 0,
             (~m).sum(), ok[~m].sum(), 100 * ok[~m].mean()))
