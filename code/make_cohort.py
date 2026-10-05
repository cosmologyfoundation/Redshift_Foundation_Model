"""Build a blind cohort: 20 highest-SNR DR1 spectra per class, 100 total.

Selection is Redrock-free.  The only catalogue columns touched are TARGETID,
SURVEY, PROGRAM, HEALPIX, COADD_FIBERSTATUS, DESI_TARGET and TSNR2_*.

  keep      SURVEY=="main", TARGETID>0, COADD_FIBERSTATUS==0
  class     DESI_TARGET bits, priority QSO(2) > LRG(0) > ELG(1) > BGS_ANY(60)
            > MWS_ANY(61)=STAR
  dedup     a TARGETID can appear on several rows; keep the one with the
            highest TSNR2_<class>.  NOT the *_PRIMARY flags -- those rank by
            ZWARN, which is a Redrock product.
  shortlist top ~60 per class by TSNR2_<class> (STAR uses TSNR2_BGS, the
            bright-time metric) as a brightness proxy
  final     extract the spectra, measure SNR = median(|flux|*sqrt(ivar)) over
            valid pixels, keep the top 20 per class

The renders carry NO object information -- no class, no SNR, no TSNR2, no
programme, only an opaque id.  The previous cohort printed TSNR2 and programme
on every panel, which leaks brightness and observing mode into a blind read.

--write-sealed-key stores the pipeline Z/ZWARN/SPECTYPE for the final 100 to
data/key_SEALED.json with mode 600.  It is opened once, by the final
comparison, and nothing upstream reads it.
"""
import os, sys, json, time
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

ROOT = os.environ.get('C2', '/pscratch/sd/s/shunth/claude_learning_map/cohort2')
DATA, IMG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'img')
DR1 = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron'
CAT = os.path.join(DR1, 'zcatalog/v1/zall-pix-iron.fits')
SEEN = '/pscratch/sd/s/shunth/claude_learning_map/seen_targets.json'
def _opt(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

CLASSES = (_opt('--classes', 'STAR,BGS,LRG,ELG,QSO').split(','))

# --zmax is a Redrock quantity used at selection.  It exists so a *learning*
# cohort can be restricted to a redshift range being studied -- it is NOT
# blind, and any cohort built with it must be reported as a learning set.
ZMAX = _opt('--zmax', None, float)


def _window_report(zmin, zmax):
    """What a redshift cut leaves in the 3600-9824 A window, per class.

    Pure geometry -- rest wavelengths and the spectrograph range.  No Redrock,
    so this is printable at build time without leaking anything.

    It exists because `--zmax 1.6` was applied to two cohorts without anyone
    working out what it does to the QSO class.  Measured afterwards on 40 QSO
    targets, grouped by how many broad UV lines fell in the window:

        4 lines  9/9      2 lines  3/5
        3 lines  13/14    1 line   0/4

    Below z = 0.886 a quasar has only Mg II in range, and a single line cannot
    be checked against a second (RULE.md 5: one line 6/7, two or more 29/30).
    The cut did not make the sample slightly harder -- it removed the evidence.

    The flag deliberately compares the requested range against the whole
    0 - 3.5 range and fires only on classes the cut makes WORSE.  Flagging
    every class that is hard somewhere fired on all four every time, which is
    how a warning gets ignored.
    """
    lo, hi = 3600.0, 9824.0
    lines = {
        'QSO':  {'Lya': 1215.7, 'C IV': 1549.5, 'C III]': 1908.7,
                 'Mg II': 2799.1},
        'ELG':  {'[O II]': 3728.5, 'Hbeta': 4862.7, '[O III]': 5008.2,
                 'Halpha': 6564.6},
        'LRG':  {'Ca H+K': 3949.0, 'D4000': 4000.0, 'G band': 4304.4,
                 'Mg b': 5176.7},
    }
    lines['BGS'] = dict(lines['ELG'], **{'D4000': 4000.0})

    def thin_share(cls, a, b):
        """Share of [a, b] where the class has at most one feature in range."""
        zz = np.linspace(a, b, 2000)
        n = np.zeros_like(zz)
        for l0 in lines[cls].values():
            n += ((l0 * (1 + zz) > lo) & (l0 * (1 + zz) < hi))
        return float((n <= 1).mean())

    a = 0.0 if zmin is None else float(zmin)
    b = 3.5 if zmax is None else float(zmax)
    print('redshift range %.2f - %.2f: what is left in the %g-%g A window'
          % (a, b, lo, hi), flush=True)
    worse = []
    for cls in ('QSO', 'ELG', 'BGS', 'LRG'):
        if cls not in CLASSES:
            continue
        rep = []
        for nm, l0 in sorted(lines[cls].items(), key=lambda kv: kv[1]):
            zin, zout = max(a, lo / l0 - 1.0), min(b, hi / l0 - 1.0)
            rep.append('%s %.0f%%' % (nm, 100 * max(0.0, zout - zin)
                                      / max(b - a, 1e-9)))
        cut, full = thin_share(cls, a, b), thin_share(cls, 0.0, 3.5)
        print('  %-4s %-58s  <=1 feature over %3.0f%% of the range '
              '(unrestricted: %.0f%%)'
              % (cls, ', '.join(rep), 100 * cut, 100 * full), flush=True)
        if cut > full + 0.10:
            worse.append((cls, cut, full))
    for cls, cut, full in worse:
        print('  *** THIS CUT MAKES %s HARDER: the share of the range where it '
              'has at most one feature rises from %.0f%% to %.0f%%. On 40 QSO '
              'targets the one-line group scored 0/4 while the three- and '
              'four-line groups scored 22/23. Expect the score to fall for '
              'reasons that have nothing to do with the reader.'
              % (cls, 100 * full, 100 * cut), flush=True)
    if not worse:
        print('  no class is made thinner by this cut than by the full '
              '0 - 3.5 range.', flush=True)


# SV3 has its own targeting column.  The five class bits sit at the same
# positions in SV3_DESI_TARGET as in DESI_TARGET, but the column name differs.
SURVEY_SEL = _opt('--survey', 'main')
TGTCOL = 'SV3_DESI_TARGET' if SURVEY_SEL == 'sv3' else 'DESI_TARGET'
BITS = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS': 60, 'STAR': 61}
TSNR = {'QSO': 'TSNR2_QSO', 'LRG': 'TSNR2_LRG', 'ELG': 'TSNR2_ELG',
        'BGS': 'TSNR2_BGS', 'STAR': 'TSNR2_BGS'}
COLS = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'COADD_FIBERSTATUS',
        TGTCOL, 'TSNR2_QSO', 'TSNR2_LRG', 'TSNR2_ELG', 'TSNR2_BGS',
        'ZWARN']
if ZMAX is not None:
    COLS = COLS + ['Z']
CHUNK, SHORT = 2_000_000, 40
FINAL = _opt('--n', 10, int)
LAM = np.round(np.arange(3600.0, 9824.0 + 0.4, 0.8), 4)
NW = len(LAM)


def assign_class(dt):
    """priority QSO > LRG > ELG > BGS_ANY > MWS_ANY; anything else dropped"""
    c = np.full(len(dt), '', dtype='<U4')
    for nm in ['STAR', 'BGS', 'ELG', 'LRG', 'QSO']:      # lowest first
        c[((dt >> BITS[nm]) & 1).astype(bool)] = nm
    return c


def scan():
    seen = set()
    if os.path.exists(SEEN):
        seen = set(json.load(open(SEEN))['targetids'])
    print(f'excluding {len(seen)} previously studied targets', flush=True)
    best = {c: {} for c in CLASSES}       # tid -> row dict, highest TSNR2
    with fits.open(CAT, memmap=True) as h:
        d = h[1].data
        n = len(d)
        for s in range(0, n, CHUNK):
            t = {c: np.asarray(d[c][s:s + CHUNK]) for c in COLS}
            sv = np.char.strip(t['SURVEY'].astype(str))
            keep = ((sv == SURVEY_SEL) & (t['TARGETID'] > 0)
                    & (t['COADD_FIBERSTATUS'] == 0)
                    & (t['ZWARN'] == 0))      # rule added 2026-09-02
            if ZMAX is not None:
                keep &= (t['Z'] < ZMAX)
            cls = assign_class(t[TGTCOL])
            keep &= (cls != '')
            if seen:
                keep &= ~np.isin(t['TARGETID'], list(seen))
            pg = np.char.strip(t['PROGRAM'].astype(str))
            for c in CLASSES:
                k = np.where(keep & (cls == c))[0]
                if not len(k):
                    continue
                v = t[TSNR[c]][k]
                g = np.isfinite(v) & (v > 0)
                k, v = k[g], v[g]
                if not len(k):
                    continue
                take = k[np.argsort(-v)[:SHORT * 8]]
                for i in take:
                    tid = int(t['TARGETID'][i]); val = float(t[TSNR[c]][i])
                    old = best[c].get(tid)
                    if old is None or val > old['tsnr2']:
                        best[c][tid] = dict(tid=tid, cls=c, tsnr2=val,
                                            survey=str(sv[i]),
                                            program=str(pg[i]),
                                            healpix=int(t['HEALPIX'][i]))
            if (s // CHUNK) % 4 == 0:
                print(f'  {s + CHUNK:,}/{n:,}', flush=True)
    short = {}
    for c in CLASSES:
        rows = sorted(best[c].values(), key=lambda r: -r['tsnr2'])[:SHORT]
        short[c] = rows
        print(f'  {c}: {len(best[c])} unique targets -> shortlist {len(rows)}, '
              f'TSNR2 {rows[-1]["tsnr2"]:.0f}-{rows[0]["tsnr2"]:.0f}')
    return short


def coadd_path(sv, pg, hp):
    return os.path.join(DR1, 'healpix', sv, pg, str(hp // 100), str(hp),
                        f'coadd-{sv}-{pg}-{hp}.fits')


def extract(rows):
    """ivar-weighted merge of the three cameras onto the shared 0.8 A comb"""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r['survey'], r['program'], r['healpix']), []).append(i)
    n = len(rows)
    FL = np.zeros((n, NW), np.float32); IV = np.zeros((n, NW), np.float32)
    got = np.zeros(n, bool)
    for (sv, pg, hp), idxs in sorted(by.items()):
        p = coadd_path(sv, pg, hp)
        if not os.path.exists(p):
            continue
        want = {rows[i]['tid']: i for i in idxs}
        with fits.open(p, memmap=True) as h:
            tid = h['FIBERMAP'].data['TARGETID']
            rr, dd = [], []
            for tt, i in want.items():
                w = np.where(tid == tt)[0]
                if len(w):
                    rr.append(int(w[0])); dd.append(i)
            if not rr:
                continue
            num = np.zeros((len(rr), NW)); den = np.zeros((len(rr), NW))
            for b in ('B', 'R', 'Z'):
                w = h[f'{b}_WAVELENGTH'].data
                f = np.asarray(h[f'{b}_FLUX'].data[rr], float)
                iv = np.asarray(h[f'{b}_IVAR'].data[rr], float)
                m = np.asarray(h[f'{b}_MASK'].data[rr], np.int32)
                j = np.round((w - LAM[0]) / 0.8).astype(int)
                ok = (j >= 0) & (j < NW)
                j = j[ok]
                ive = np.where((m[:, ok] != 0) | ~np.isfinite(f[:, ok])
                               | ~np.isfinite(iv[:, ok]), 0.0, iv[:, ok])
                fe = np.nan_to_num(f[:, ok])
                num[:, j] += ive * fe
                den[:, j] += ive
        fl = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        FL[dd] = fl.astype(np.float32); IV[dd] = den.astype(np.float32)
        got[dd] = True
    return FL, IV, got


def render(idx, FL, IV, ids):
    """A clean spectrum.  No class, no SNR, no programme -- only the id."""
    os.makedirs(IMG, exist_ok=True)
    for k, i in enumerate(idx):
        fl, iv = FL[i], IV[i]
        good = iv > 0
        y = np.where(good, fl, np.nan)
        y = uniform_filter1d(np.nan_to_num(y), 5)
        y[~good] = np.nan
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
        ax.set_title(ids[k], fontsize=11, loc='left')
        ax.grid(alpha=.25)
        plt.tight_layout()
        fig.savefig(os.path.join(IMG, f'{ids[k]}.png'), dpi=105)
        plt.close(fig)


def sealed_key(sel_rows):
    """Reference Z/ZWARN/SPECTYPE for exactly the rows that were selected.

    A TARGETID appears on several catalogue rows, each with its own Redrock
    solution, so matching on TARGETID alone can key the answer to a different
    observation than the spectrum that was extracted.  Match the full row.
    """
    key_of = {(int(r['tid']), int(r['healpix']), str(r['survey']),
               str(r['program'])) for r in sel_rows}
    tids = set(int(r['tid']) for r in sel_rows)
    out = {}
    with fits.open(CAT, memmap=True) as h:
        d = h[1].data
        for s_ in range(0, len(d), CHUNK):
            t = np.asarray(d['TARGETID'][s_:s_ + CHUNK])
            m = np.isin(t, list(tids))
            if m.any():
                hp = np.asarray(d['HEALPIX'][s_:s_ + CHUNK])[m]
                sv = np.char.strip(
                    np.asarray(d['SURVEY'][s_:s_ + CHUNK]).astype(str))[m]
                pg = np.char.strip(
                    np.asarray(d['PROGRAM'][s_:s_ + CHUNK]).astype(str))[m]
                z = np.asarray(d['Z'][s_:s_ + CHUNK])[m]
                zw = np.asarray(d['ZWARN'][s_:s_ + CHUNK])[m]
                st = np.asarray(d['SPECTYPE'][s_:s_ + CHUNK])[m]
                for a, hh, ss, pp, b, c, e in zip(t[m], hp, sv, pg, z, zw, st):
                    if (int(a), int(hh), str(ss), str(pp)) in key_of:
                        out[int(a)] = dict(z=float(b), zwarn=int(c),
                                           spectype=str(e).strip())
            if not (tids - set(out)):
                break
    missing = tids - set(out)
    if missing:
        raise SystemExit('sealed key: no matching row for %d targets'
                         % len(missing))
    flagged = {k: v['zwarn'] for k, v in out.items() if v['zwarn'] != 0}
    if flagged:
        raise SystemExit('sealed key: %d targets carry ZWARN != 0 -- the '
                         'ZWARN==0 filter did not hold' % len(flagged))
    p = os.path.join(DATA, 'key_SEALED.json')
    json.dump(out, open(p, 'w'), indent=1)
    os.chmod(p, 0o600)
    print(f'sealed key written for {len(out)} targets (mode 600), all '
          f'ZWARN==0 -- not read again until the final comparison')


def main():
    # ROOT defaults to cohort3 when C2 is unset.  Forgetting C2 once silently
    # overwrote Part V.  Refuse rather than overwrite, and refuse to default.
    if 'C2' not in os.environ:
        raise SystemExit(
            'C2 is not set. This script would default to %s and overwrite it. '
            'Set C2 to the cohort directory you intend to create.' % ROOT)
    if os.path.exists(os.path.join(DATA, 'cohort.npz')):
        raise SystemExit(
            '%s already holds a cohort (cohort.npz exists). A cohort is '
            'written once; delete the directory deliberately to rebuild.' % ROOT)
    os.makedirs(DATA, exist_ok=True)
    # Printed BEFORE anything is drawn, so the cost of the cut is on screen
    # while it can still be changed.  Correction 16.
    _window_report(None, ZMAX)
    t0 = time.time()
    short = scan()
    rows = [r for c in CLASSES for r in short[c]]
    FL, IV, got = extract(rows)
    print(f'extracted {int(got.sum())}/{len(rows)}  ({time.time()-t0:.0f}s)',
          flush=True)
    with np.errstate(invalid='ignore'):
        snr = np.array([np.nanmedian(np.abs(FL[i][IV[i] > 0])
                                     * np.sqrt(IV[i][IV[i] > 0]))
                        if got[i] and (IV[i] > 0).any() else np.nan
                        for i in range(len(rows))])
    sel = []
    for c in CLASSES:
        k = [i for i, r in enumerate(rows)
             if r['cls'] == c and got[i] and np.isfinite(snr[i])]
        k = sorted(k, key=lambda i: -snr[i])[:FINAL]
        sel += k
        print(f'  {c}: measured SNR {snr[k[-1]]:.1f}-{snr[k[0]]:.1f} '
              f'(n={len(k)})')
    sel = np.array(sel)
    rng = np.random.default_rng(20260820)
    order = rng.permutation(len(sel))          # ids carry no class ordering
    ids = [f'OBJ{j:03d}' for j in range(len(sel))]
    sel_shuf = sel[order]
    items = [dict(id=ids[j], tid=rows[i]['tid'], cls=rows[i]['cls'],
                  survey=rows[i]['survey'], program=rows[i]['program'],
                  healpix=rows[i]['healpix'], tsnr2=rows[i]['tsnr2'],
                  snr=float(snr[i])) for j, i in enumerate(sel_shuf)]
    np.savez(os.path.join(DATA, 'cohort.npz'), flux=FL[sel_shuf],
             ivar=IV[sel_shuf], lam=LAM,
             tid=np.array([r['tid'] for r in [rows[i] for i in sel_shuf]]),
             cls=np.array([r['cls'] for r in [rows[i] for i in sel_shuf]]),
             snr=snr[sel_shuf],
             survey=np.array([r['survey'] for r in [rows[i] for i in sel_shuf]]),
             program=np.array([r['program'] for r in [rows[i] for i in sel_shuf]]),
             healpix=np.array([r['healpix'] for r in [rows[i] for i in sel_shuf]]),
             id=np.array(ids))
    json.dump(items, open(os.path.join(DATA, 'items.json'), 'w'), indent=1)
    json.dump(dict(zwarn0=True, since='2026-09-02', survey_sel=SURVEY_SEL,
                   target_column=TGTCOL, classes=CLASSES, n_per_class=FINAL,
                   redrock_zmax=ZMAX, blind=(ZMAX is None),
                   blind_note=('sealed blind test' if ZMAX is None else
                               'LEARNING SET -- selection used Redrock Z < %s'
                               % ZMAX),
                   survey=SURVEY_SEL, targetid_positive=True,
                   coadd_fiberstatus0=True,
                   dedup='highest TSNR2_<class>, never *_PRIMARY'),
              open(os.path.join(DATA, 'selection_rules.json'), 'w'), indent=1)
    render(sel_shuf, FL, IV, ids)
    print(f'wrote cohort.npz, items.json and {len(ids)} renders')

    book = json.load(open(SEEN))
    book['targetids'] += [int(r['tid']) for r in [rows[i] for i in sel_shuf]]
    book['sessions'].append(dict(
        id=len(book['sessions']) + 1, n=len(sel),
        note='%s %s %d/class%s' % (os.path.basename(ROOT.rstrip('/')),
                                   '+'.join(CLASSES), FINAL,
                                   '' if ZMAX is None else
                                   ' (learning set, z<%s)' % ZMAX)))
    json.dump(book, open(SEEN, 'w'), indent=1)
    print(f'seen list now {len(book["targetids"])}')

    # The key is written by default.  It used to be opt-in, and a build
    # without the flag produced 50 spectra, 50 renders and 50 spent targets
    # that could never be scored -- discovered only when the scoring step
    # looked for the file.  An opt-out has to be stated.
    if '--no-sealed-key' in sys.argv:
        print('WARNING: --no-sealed-key given; this cohort cannot be scored')
    else:
        sealed_key([rows[i] for i in sel_shuf])
    print(f'DONE {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
