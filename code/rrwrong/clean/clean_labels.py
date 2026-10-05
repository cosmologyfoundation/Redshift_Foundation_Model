"""Flag training labels that are likely wrong (owner, 2026-09-27; DESIGN.md 6d-2).

Any Redrock field may be used to DELETE training rows.  Per spectrum of the full DR1
zcatalog (zall-pix-iron, 28.4M rows) we compute evidence of a wrong redshift, then map it
onto the rows of the two training files (main primary 18.66M, sv3 primary 0.99M) by
(TARGETID, SURVEY, PROGRAM).  Nothing here is a model input; validation is never filtered.

Per row (saved, continuous where possible, so the deletion rule can be changed later):
  dchi2        Redrock DELTACHI2 (best vs second-best z)
  rep_agree    other independent ZWARN=0 coadds of the same target (another survey or program)
               whose z agrees within 0.01(1+z)
  rep_dis      ... whose z differs by more than 0.01(1+z)  -> at least one of them is wrong
  gaia_star    Gaia proper motion or parallax clearly non-zero (catalogue values; the error is
               approximated from G because zall carries no *_IVAR columns)
  rchi2        CHI2 / NPIXELS;  zerr_rel  ZERR / (1+Z)
Flags (bit):   1 LOW_DCHI2   2 REPEAT_DISAGREE   4 GAIA_STAR_HIGHZ (z > 0.01)
               8 MWS_TARGET_HIGHZ (star-only target, z > 0.1 -- the proven white-dwarf pattern)
               16 RCHI2_TAIL   32 ZERR_TAIL  (top 0.1% within SPECTYPE)
"""
import os, sys, time, numpy as np, h5py, fitsio, pandas as pd
from multiprocessing import Pool
ZALL = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'
P = '/pscratch/sd/s/shunth/'
H5 = {'main': ('dr1_main_primary_tgt_zwarn0_brz.h5', 'DESI_TARGET', 'MWS_TARGET'),
      'sv3': ('dr1_sv3_primary_tgt_zwarn0_brz.h5', 'SV3_DESI_TARGET', 'SV3_MWS_TARGET')}
COLS = ['TARGETID', 'SURVEY', 'PROGRAM', 'Z', 'ZERR', 'ZWARN', 'CHI2', 'NPIXELS', 'SPECTYPE', 'DELTACHI2',
        'COADD_FIBERSTATUS', 'PARALLAX', 'PMRA', 'PMDEC', 'GAIA_PHOT_G_MEAN_MAG']
TOL = 0.01
t0 = time.time()
log = lambda *a: print('[%5.0fs]' % (time.time() - t0), *a, flush=True)

def read_chunk(se):
    s, e = se
    d = fitsio.read(ZALL, ext=1, columns=COLS, rows=np.arange(s, e))
    out = {c: d[c] for c in COLS}
    for c in ('SURVEY', 'PROGRAM', 'SPECTYPE'):
        out[c] = np.char.strip(out[c].astype('U'))
    return out

def gaia_sigma(G):
    """Approximate Gaia DR3 proper-motion error [mas/yr] from G (Lindegren+2021 fig.); parallax ~0.8x."""
    G = np.where(np.isfinite(G) & (G > 0), G, 21.0)
    return np.interp(G, [12, 15, 17, 19, 20, 21], [0.015, 0.02, 0.07, 0.3, 0.6, 1.4])

if __name__ == '__main__':
    nrow = fitsio.FITS(ZALL)[1].get_nrows()
    edges = np.linspace(0, nrow, 257).astype(int)
    with Pool(64) as pool:
        parts = pool.map(read_chunk, list(zip(edges[:-1], edges[1:])))
    Z = {c: np.concatenate([p[c] for p in parts]) for c in COLS}
    del parts
    log('zall read: %d spectra' % nrow)
    df = pd.DataFrame({'tid': Z['TARGETID'].astype(np.int64), 'survey': Z['SURVEY'], 'program': Z['PROGRAM'],
                       'z': Z['Z'].astype(np.float64), 'ok': (Z['ZWARN'] == 0) & (Z['COADD_FIBERSTATUS'] == 0)})
    df['i'] = np.arange(len(df))
    # ---- repeats: other independent coadds of the same target, both ZWARN = 0
    good = df[df.ok & (df.tid > 0)]
    multi = good[good.duplicated('tid', keep=False)]
    pr = multi[['tid', 'z', 'i']].merge(multi[['tid', 'z', 'i']], on='tid', suffixes=('', '_o'))
    pr = pr[pr.i != pr.i_o]
    dz = np.abs(pr.z.values - pr.z_o.values) / (1 + np.minimum(pr.z.values, pr.z_o.values))
    pr = pr.assign(dis=dz > TOL, dz=dz)
    agg = pr.groupby('i').agg(rep_dis=('dis', 'sum'), rep_n=('dis', 'size'), rep_maxdz=('dz', 'max'))
    rep_dis = np.zeros(nrow, np.int16); rep_agree = np.zeros(nrow, np.int16); rep_maxdz = np.zeros(nrow, np.float32)
    rep_dis[agg.index.values] = agg.rep_dis.values
    rep_agree[agg.index.values] = agg.rep_n.values - agg.rep_dis.values
    rep_maxdz[agg.index.values] = agg.rep_maxdz.values
    log('repeats: %d targets with >=2 ZWARN=0 coadds, %d pairs, %d spectra disagree with another' %
        (multi.tid.nunique(), len(pr), int((rep_dis > 0).sum())))
    # ---- Gaia: catalogue proper motion / parallax against an error approximated from G
    sg = gaia_sigma(Z['GAIA_PHOT_G_MEAN_MAG'].astype(float))
    pm = np.hypot(np.nan_to_num(Z['PMRA'].astype(float)), np.nan_to_num(Z['PMDEC'].astype(float)))
    plx = np.nan_to_num(Z['PARALLAX'].astype(float))
    gaia_star = (pm / sg > 8) | (plx / (0.8 * sg) > 8)
    rchi2 = Z['CHI2'] / np.maximum(Z['NPIXELS'], 1)
    zrel = Z['ZERR'] / (1 + Z['Z'])
    tail_r = np.zeros(nrow, bool); tail_z = np.zeros(nrow, bool)
    for st in np.unique(Z['SPECTYPE']):
        m = (Z['SPECTYPE'] == st) & df.ok.values
        if m.sum() > 1000:
            tail_r |= m & (rchi2 > np.quantile(rchi2[m], 0.999))
            tail_z |= m & (zrel > np.quantile(zrel[m], 0.999))
    key = pd.DataFrame({'tid': df.tid, 'survey': df.survey, 'program': df.program, 'i': df.i})
    # ---- map onto the training files
    for name, (fn, tcol, mcol) in H5.items():
        with h5py.File(P + fn, 'r') as f:
            h = pd.DataFrame({'tid': f['TARGETID'][:].astype(np.int64),
                              'survey': np.char.strip(f['SURVEY'][:].astype('U')),
                              'program': np.char.strip(f['PROGRAM'][:].astype('U'))})
            zz = f['Z'][:]; dt = f[tcol][:].astype(np.int64); mw = f[mcol][:].astype(np.int64)
        h['row'] = np.arange(len(h))
        m = h.merge(key, on=['tid', 'survey', 'program'], how='left')
        m = m.drop_duplicates('row').sort_values('row')
        assert len(m) == len(h), name
        found = m.i.notna().values
        ii = m.i.fillna(0).astype(np.int64).values
        star_only = (((dt >> 61) & 1) == 1) & ((dt & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 60))) == 0)
        out = dict(row=h.row.values, found=found,
                   dchi2=Z['DELTACHI2'][ii].astype(np.float32), rep_agree=rep_agree[ii], rep_dis=rep_dis[ii],
                   rep_maxdz=rep_maxdz[ii], gaia_star=gaia_star[ii], rchi2=rchi2[ii].astype(np.float32),
                   zerr_rel=zrel[ii].astype(np.float32), z=zz, star_only=star_only, mws=mw != 0)
        fl = np.zeros(len(h), np.uint16)
        fl |= (out['rep_dis'] > 0).astype(np.uint16) * 2
        fl |= (out['gaia_star'] & (zz > 0.01)).astype(np.uint16) * 4
        fl |= (star_only & (zz > 0.1)).astype(np.uint16) * 8
        fl |= tail_r[ii].astype(np.uint16) * 16
        fl |= tail_z[ii].astype(np.uint16) * 32
        out['flags'] = np.where(found, fl, 0).astype(np.uint16)
        np.savez(P + 'zfm_clean_%s.npz' % name, **out)
        log('%s: %d rows, matched %.4f%%; flags: repeat-disagree %d, gaia-star z>0.01 %d, star-only z>0.1 %d, '
            'rchi2 tail %d, zerr tail %d' % (name, len(h), 100 * found.mean(), int(((fl & 2) > 0).sum()),
            int(((fl & 4) > 0).sum()), int(((fl & 8) > 0).sum()), int(((fl & 16) > 0).sum()), int(((fl & 32) > 0).sum())))
