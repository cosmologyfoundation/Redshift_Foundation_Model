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
CLASSES = ['STAR', 'BGS', 'LRG', 'ELG', 'QSO']
BITS = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS': 60, 'STAR': 61}
TSNR = {'QSO': 'TSNR2_QSO', 'LRG': 'TSNR2_LRG', 'ELG': 'TSNR2_ELG',
        'BGS': 'TSNR2_BGS', 'STAR': 'TSNR2_BGS'}
COLS = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'COADD_FIBERSTATUS',
        'DESI_TARGET', 'TSNR2_QSO', 'TSNR2_LRG', 'TSNR2_ELG', 'TSNR2_BGS']
CHUNK, SHORT, FINAL = 2_000_000, 60, 20
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
            keep = ((sv == 'main') & (t['TARGETID'] > 0)
                    & (t['COADD_FIBERSTATUS'] == 0))
            cls = assign_class(t['DESI_TARGET'])
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


def sealed_key(tids):
    want = set(int(t) for t in tids)
    out = {}
    with fits.open(CAT, memmap=True) as h:
        d = h[1].data
        for s in range(0, len(d), CHUNK):
            t = np.asarray(d['TARGETID'][s:s + CHUNK])
            m = np.isin(t, list(want))
            if m.any():
                z = np.asarray(d['Z'][s:s + CHUNK])[m]
                zw = np.asarray(d['ZWARN'][s:s + CHUNK])[m]
                st = np.asarray(d['SPECTYPE'][s:s + CHUNK])[m]
                for a, b, c, e in zip(t[m], z, zw, st):
                    out.setdefault(int(a), dict(z=float(b), zwarn=int(c),
                                                spectype=str(e).strip()))
            if not (want - set(out)):
                break
    p = os.path.join(DATA, 'key_SEALED.json')
    json.dump(out, open(p, 'w'), indent=1)
    os.chmod(p, 0o600)
    print(f'sealed key written for {len(out)} targets (mode 600) -- '
          f'not read again until the final comparison')


def main():
    os.makedirs(DATA, exist_ok=True)
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
    render(sel_shuf, FL, IV, ids)
    print(f'wrote cohort.npz, items.json and {len(ids)} renders')

    book = json.load(open(SEEN))
    book['targetids'] += [int(r['tid']) for r in [rows[i] for i in sel_shuf]]
    book['sessions'].append(dict(id=len(book['sessions']) + 1, n=len(sel),
                                 note='cohort2 blind 20/class'))
    json.dump(book, open(SEEN, 'w'), indent=1)
    print(f'seen list now {len(book["targetids"])}')

    if '--write-sealed-key' in sys.argv:
        sealed_key([r['tid'] for r in [rows[i] for i in sel_shuf]])
    print(f'DONE {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
