"""Build a cohort directly from the DR1 BRZ h5 files.

Routing (standing instruction, 2026-09-05): a request that says "sv3" draws from
dr1_sv3_primary_tgt_zwarn0_brz.h5; a request that names no survey draws from
dr1_main_primary_tgt_zwarn0_brz.h5.

How this differs from make_cohort3.py, measured rather than assumed:

  inclusion   IDENTICAL. Both admit an object iff at least one of its
              observations has ZWARN == 0. The h5 filter is
              `ZCAT_PRIMARY == True and ZWARN == 0`, and because ZCAT_PRIMARY
              prefers ZWARN == 0 rows, that is the same set. An earlier version
              of this tooling refused to draw here at all; that refusal was
              over-broad and is now a recorded caveat instead.
  dedup       DIFFERS, but only where there is something to choose: 1.95% of
              main and 27.19% of sv3 have ZCAT_NSPEC > 1. For those, this path
              takes the ZCAT_PRIMARY exposure where make_cohort3 takes the
              highest TSNR2_<class> one. cohort16/OBJ049 is a worked example:
              main row TSNR2_QSO 178.0 against sv3 row 172.1, same target.
  fiberstatus NOT applied by the h5 build (99.994% / 99.815% are zero), so it
              is applied here.

Output layout is identical to make_cohort3.py, so check.py, measure3.py and the
scoring scripts work unchanged.
"""
import os, sys, json, time, hashlib
import numpy as np
import h5py
import speclib as SL

P = '/pscratch/sd/s/shunth/'
FILES = {'main': P + 'dr1_main_primary_tgt_zwarn0_brz.h5',
         'sv3': P + 'dr1_sv3_primary_tgt_zwarn0_brz.h5'}
TGTCOL = {'main': 'DESI_TARGET', 'sv3': 'SV3_DESI_TARGET'}
SEEN = P + 'claude_learning_map/seen_targets.json'
BITS = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS': 60, 'STAR': 61}
TSNR = {'QSO': 'TSNR2_QSO', 'LRG': 'TSNR2_LRG', 'ELG': 'TSNR2_ELG',
        'BGS': 'TSNR2_BGS', 'STAR': 'TSNR2_BGS'}
NSPEC_CHOICE = {'main': 1.95, 'sv3': 27.19}   # % with ZCAT_NSPEC > 1


def _opt(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


ROOT = os.environ.get('C2')
SURVEY = _opt('--survey', 'main')
CLASSES = _opt('--classes', 'STAR,BGS,LRG,ELG,QSO').split(',')
FINAL = _opt('--n', 10, int)
ZMAX = _opt('--zmax', None, float)
# A FIXED seed draws the same POSITIONS every time: removing the previous 50
# rows from 18.6M shifts the index array by almost nothing, so cohort18 landed
# within a median of 2 rows of cohort17 -- the same tiles, not an independent
# sample. The default now varies with the target directory, and the value used
# is written into selection_rules.json so a draw stays reproducible.
SEED = _opt('--seed', None, int)
if SEED is None:
    SEED = int(hashlib.sha256(
        os.path.basename((os.environ.get('C2') or '').rstrip('/')).encode()
    ).hexdigest()[:8], 16)
DRY = '--dry-run' in sys.argv


def main():
    if not ROOT:
        raise SystemExit('C2 is not set. Set it to the cohort directory to create.')
    DATA, IMG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'img')
    if os.path.exists(os.path.join(DATA, 'cohort.npz')):
        raise SystemExit('%s already holds a cohort. A cohort is written once.' % ROOT)
    if SURVEY not in FILES:
        raise SystemExit('--survey must be main or sv3')
    if not DRY:
        os.makedirs(DATA, exist_ok=True)
    t0 = time.time()

    seen = set(int(x) for x in json.load(open(SEEN))['targetids'])
    print('routing: %s -> %s' % (SURVEY, os.path.basename(FILES[SURVEY])), flush=True)
    print('seen list holds %d TARGETIDs' % len(seen), flush=True)
    print('dedup is ZCAT_PRIMARY, not TSNR2; it differs from the zcatalog path '
          'on the %.2f%% of %s with more than one spectrum'
          % (NSPEC_CHOICE[SURVEY], SURVEY), flush=True)

    with h5py.File(FILES[SURVEY], 'r') as h:
        tid = h['TARGETID'][:]
        tgt = h[TGTCOL[SURVEY]][:]
        fs = h['COADD_FIBERSTATUS'][:]
        z = h['Z'][:] if ZMAX is not None else None
        ok = (fs == 0) & (tid > 0) & ~np.isin(tid, list(seen))
        if ZMAX is not None:
            ok &= z < ZMAX
        print('  usable rows after fiberstatus / seen / zmax: %d of %d'
              % (ok.sum(), len(tid)), flush=True)

        rng = np.random.default_rng(SEED)
        sel, cls_of = [], {}
        for c in CLASSES:
            m = ok & ((tgt & (1 << BITS[c])) != 0)
            idx = np.flatnonzero(m)
            if len(idx) < FINAL:
                raise SystemExit('class %s: %d candidates, need %d' % (c, len(idx), FINAL))
            take = idx[rng.choice(len(idx), FINAL, replace=False)]
            sel += take.tolist()
            for i in take:
                cls_of[int(i)] = c
            print('  %-5s %9d candidates -> drew %d' % (c, len(idx), FINAL), flush=True)

        sel = np.array(sorted(set(sel)))
        assert len(sel) == FINAL * len(CLASSES), 'a row was drawn for two classes'
        assert not (set(tid[sel].tolist()) & seen), 'draw overlaps the seen list'

        # h5py fancy indexing requires increasing indices, but the OBJ ids must
        # carry no class ordering.  So EVERY read is done on the sorted index
        # and the shuffle is applied afterwards, to all arrays alike.
        order = rng.permutation(len(sel))
        srt = np.sort(sel)
        pos = {int(v): k for k, v in enumerate(srt)}
        pick = np.array([pos[int(v)] for v in sel[order]])
        dec = lambda a: np.char.strip(np.asarray(a).astype(str))

        FL = h['BRZ_FLUX'][srt, :].astype(np.float32)[pick]
        IV = h['BRZ_IVAR'][srt, :].astype(np.float32)[pick]
        MK = h['BRZ_MASK'][srt, :][pick]
        sv = dec(h['SURVEY'][srt])[pick]
        pg = dec(h['PROGRAM'][srt])[pick]
        hp = h['HEALPIX'][srt][pick]
        tt = h['TARGETID'][srt][pick]
        zz = h['Z'][srt][pick]
        zw = h['ZWARN'][srt][pick]
        st = dec(h['SPECTYPE'][srt])[pick]
        tsn = {c: h[TSNR[c]][srt][pick] for c in set(cls_of.values())}

        sel = sel[order]
        IV = np.where((MK == 0) & np.isfinite(FL) & np.isfinite(IV), IV, 0.0)
        FL = np.nan_to_num(FL)
        cls = np.array([cls_of[int(i)] for i in sel])
        # 13.6% of the 822 objects scored so far carry MORE THAN ONE class bit,
        # and `cls` is only the one this draw happened to pick.  Record the
        # whole mask so the scorer can see when the "right answer" is not
        # unique: over those 822, scoring against the drawn bit gives 90.0%
        # and against any bit the object carries 92.1%.
        allcls = np.array(['+'.join(sorted(c for c, bb in BITS.items()
                                           if (int(tgt[i]) >> bb) & 1))
                           for i in sel])
        ts = np.array([float(tsn[cls[j]][j]) for j in range(len(sel))])
        key = {int(tt[j]): dict(z=float(zz[j]), zwarn=int(zw[j]),
                                spectype=str(st[j])) for j in range(len(sel))}

    # The dry run exercises the whole read and check path and stops only before
    # the first write.  The earlier version returned before the reads, which is
    # exactly why it passed while the real build raised on h5py index ordering.
    if DRY:
        print('--dry-run: selection, reads and all checks passed, nothing written')
        return

    LAM = np.arange(7781) * 0.8 + 3600.0
    good = IV > 0
    with np.errstate(invalid='ignore'):
        snr = np.array([np.nanmedian(np.abs(FL[i][good[i]]) * np.sqrt(IV[i][good[i]]))
                        if good[i].any() else np.nan for i in range(len(sel))])
    ids = ['OBJ%03d' % j for j in range(len(sel))]

    np.savez(os.path.join(DATA, 'cohort.npz'), flux=FL, ivar=IV, lam=LAM,
             tid=tt, cls=cls, allcls=allcls, snr=snr, survey=sv, program=pg,
             healpix=hp,
             id=np.array(ids))
    json.dump([dict(id=ids[j], tid=int(tt[j]), cls=str(cls[j]),
                    allcls=str(allcls[j]), survey=str(sv[j]),
                    program=str(pg[j]), healpix=int(hp[j]), tsnr2=float(ts[j]),
                    snr=float(snr[j])) for j in range(len(sel))],
              open(os.path.join(DATA, 'items.json'), 'w'), indent=1)
    json.dump(dict(zwarn0=True, since='2026-09-02', survey_sel=SURVEY,
                   target_column=TGTCOL[SURVEY], classes=CLASSES,
                   n_per_class=FINAL, redrock_zmax=ZMAX, blind=(ZMAX is None),
                   source=os.path.basename(FILES[SURVEY]),
                   dedup='ZCAT_PRIMARY (baked into the h5 build)',
                   dedup_caveat='inclusion is identical to the TSNR2 path; the '
                                'chosen exposure differs on the %.2f%% of %s '
                                'with ZCAT_NSPEC > 1' % (NSPEC_CHOICE[SURVEY], SURVEY),
                   coadd_fiberstatus0=True, targetid_positive=True,
                   seed=int(SEED)),
              open(os.path.join(DATA, 'selection_rules.json'), 'w'), indent=1)

    kp = os.path.join(DATA, 'key_SEALED.json')
    flagged = {k: v for k, v in key.items() if v['zwarn'] != 0}
    if flagged:
        raise SystemExit('sealed key: %d targets carry ZWARN != 0' % len(flagged))
    json.dump({str(k): v for k, v in key.items()}, open(kp, 'w'), indent=1)
    os.chmod(kp, 0o600)
    print('sealed key written for %d targets (mode 600), all ZWARN==0' % len(key))

    # byte-for-byte the same figure make_cohort3.py draws, so that a cohort
    # from this path is read on the same representation as every earlier one.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    os.makedirs(IMG, exist_ok=True)
    for j in range(len(sel)):
        g = IV[j] > 0
        y = np.where(g, FL[j], np.nan)
        y = uniform_filter1d(np.nan_to_num(y), 5)
        y[~g] = np.nan
        fig, ax = plt.subplots(figsize=(13, 3.4))
        ax.plot(LAM, y, lw=.6, color='0.15')
        f = np.isfinite(y)
        if f.any():
            a, b = np.nanpercentile(y[f], [0.5, 99.6])
            pad = .12 * (b - a) + 1e-9
            ax.set_ylim(a - pad, b + pad)
        ax.set_xlim(3600, 9824)
        ax.set_xlabel('observed wavelength [A]')
        ax.set_ylabel('flux')
        ax.set_title(ids[j], fontsize=11, loc='left')
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG, ids[j] + '.png'), dpi=110)
        plt.close(fig)
    print('wrote cohort.npz, items.json and %d renders' % len(ids))

    book = json.load(open(SEEN))
    book['targetids'] += [int(t) for t in tt]
    book['sessions'].append(dict(id=len(book['sessions']) + 1, n=len(ids),
                                 note='%s h5:%s %s %d/class'
                                      % (os.path.basename(ROOT.rstrip('/')),
                                         SURVEY, '+'.join(CLASSES), FINAL)))
    tmp = SEEN + '.tmp'
    json.dump(book, open(tmp, 'w'), indent=1)
    os.replace(tmp, SEEN)
    print('seen list now %d' % len(book['targetids']))
    print('DONE %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
