#!/usr/bin/env python
"""Flag Redrock redshifts that are wrong or likely wrong; keep the ones that are right.

Model: bagged gradient boosting on Redrock catalogue fields (DELTACHI2, ZERR, CHI2/NPIXELS,
template COEFF, Z, SPECTYPE, ZWARN, TSNR2) + Legacy Survey / Gaia imaging + target bits.
Trained on DESI EDR visual-inspection truth (Lan+22 arXiv:2208.08516, Alexander+22 arXiv:2208.08517),
with Redrock run at SV1 per-night, tile-cumulative and healpix depth (train_final.py).

Validation (see README in this folder):
  VI, leave-one-field-out, main depth, equal keep 81%:  wrong 1.58% (ZWARN0 & DCHI2>25) -> 0.35%
  1,008,385 main repeat pairs, equal keep 88%:         pair disagreement 0.73% -> 0.16%
  Gaia WD at z>0.1 removed:     P>0.01 100%, P>0.05 98.7%   (DCHI2<=25: 30.8%)
  blind-read Redrock errors:    P>0.01 78%,  P>0.05 66%     (DCHI2<=25: 13.6%)

Usage:
  python rrwrong_filter.py IN.fits OUT.parquet [--thr 0.01] [--ext ZCATALOG]
IN is any DR1 zcatalog file (zpix-*, ztile-*, zall-*). OUT has TARGETID, SURVEY, PROGRAM,
(TILEID), Z, SPECTYPE, P_WRONG, KEEP. KEEP = COADD_FIBERSTATUS==0 & P_WRONG<=thr.
"""
import argparse, numpy as np, pandas as pd, fitsio
from common import RR, OBS, PHOT, to_df
from apply_model import p_wrong

TGT = {"main": "DESI_TARGET", "sv1": "SV1_DESI_TARGET", "sv2": "SV2_DESI_TARGET", "sv3": "SV3_DESI_TARGET"}

def run(path, out, thr, ext, chunk=2_000_000):
    h = fitsio.FITS(path)[ext]; have = set(h.get_colnames())
    ids = [c for c in ["TARGETID","SURVEY","PROGRAM","TILEID","NIGHT"] if c in have]
    tcols = [c for c in TGT.values() if c in have]
    cols = ids + [c for c in RR + OBS + PHOT if c in have] + tcols
    miss = [c for c in RR + OBS + PHOT if c not in have]
    if miss: raise SystemExit(f"missing columns: {miss}")
    parts = []
    for s in range(0, h.get_nrows(), chunk):
        d = to_df(h[cols][s:min(h.get_nrows(), s+chunk)])
        for c in ["SPECTYPE","SUBTYPE","SURVEY","PROGRAM","MORPHTYPE"]:
            if c in d: d[c] = d[c].astype(str).str.strip()
        # target bits of the survey the row belongs to; one column if the file has only one
        if "SURVEY" in d and len(tcols) > 1:
            t = np.zeros(len(d), "u8")
            for sv, c in TGT.items():
                if c in d: m = (d.SURVEY==sv).to_numpy(); t[m] = d[c].to_numpy()[m].astype("u8")
            d["TGT"] = t
        else:
            d["TGT"] = d[tcols[0]].astype("u8")
        d["P_WRONG"] = p_wrong(d, "TGT").astype("f4")
        d["KEEP"] = (d.COADD_FIBERSTATUS==0).to_numpy() & (d.P_WRONG<=thr).to_numpy()
        parts.append(d[ids + ["Z","ZWARN","DELTACHI2","SPECTYPE","TGT","P_WRONG","KEEP"]])
        print(f"{min(h.get_nrows(), s+chunk)}/{h.get_nrows()} rows", flush=True)
    r = pd.concat(parts, ignore_index=True); r.to_parquet(out)
    print(f"kept {r.KEEP.mean():.4f} of {len(r)} at P_WRONG<={thr}")
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--ext", default="ZCATALOG")
    a = ap.parse_args(); run(a.inp, a.out, a.thr, a.ext)
