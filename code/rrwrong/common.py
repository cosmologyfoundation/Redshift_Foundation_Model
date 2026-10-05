"""Shared column lists and FITS->pandas loading for the Redrock-error study."""
import fitsio, numpy as np, pandas as pd
ZC = "/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/"
VI = "/global/cfs/cdirs/desi/public/edr/vac/edr/vi/v1.0/"
RR = ["Z","ZERR","ZWARN","CHI2","COEFF","NPIXELS","SPECTYPE","SUBTYPE","DELTACHI2"]
OBS = ["COADD_FIBERSTATUS","COADD_NUMEXP","COADD_EXPTIME","COADD_NUMNIGHT",
       "TSNR2_ELG","TSNR2_LRG","TSNR2_QSO","TSNR2_BGS","TSNR2_LYA"]
PHOT = ["FLUX_G","FLUX_R","FLUX_Z","FLUX_W1","FLUX_W2","FLUX_IVAR_G","FLUX_IVAR_R",
        "FLUX_IVAR_Z","FLUX_IVAR_W1","FLUX_IVAR_W2","FIBERFLUX_G","FIBERFLUX_R","FIBERFLUX_Z",
        "EBV","MORPHTYPE","SHAPE_R","SERSIC","GAIA_PHOT_G_MEAN_MAG","GAIA_PHOT_BP_MEAN_MAG",
        "GAIA_PHOT_RP_MEAN_MAG","PARALLAX","PMRA","PMDEC","MASKBITS"]

def to_df(a):
    d = {}
    for n in a.dtype.names:
        v = a[n]
        if v.ndim == 2:
            for i in range(v.shape[1]):
                d[f"{n}_{i}"] = v[:, i].astype("f8")
        elif v.dtype.kind == "S":
            d[n] = np.char.strip(v.astype("U"))
        else:
            d[n] = v.astype(v.dtype.newbyteorder("="))
    return pd.DataFrame(d)

def read_match(path, cols, tids, ext="ZCATALOG", chunk=2_000_000):
    """Read only rows whose TARGETID is in tids, in chunks."""
    f = fitsio.FITS(path)[ext]
    n = f.get_nrows(); have = set(f.get_colnames())
    cols = [c for c in cols if c in have]
    tids = np.asarray(sorted(tids))
    parts = []
    for s in range(0, n, chunk):
        t = f.read(columns=["TARGETID"], rows=np.arange(s, min(n, s+chunk)))["TARGETID"]
        idx = np.nonzero(np.isin(t, tids))[0] + s
        if len(idx):
            parts.append(to_df(f.read(columns=cols, rows=idx)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

def load_vi():
    import glob
    vi = []
    for p in sorted(glob.glob(VI + "EDR_VI_*_v1.csv")):
        d = pd.read_csv(p); d["VI_SET"] = p.split("EDR_VI_")[1].split("_v1")[0]; vi.append(d)
    return pd.concat(vi, ignore_index=True)
