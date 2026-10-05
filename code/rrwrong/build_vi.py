# Join EDR VI truth to Redrock at three depths: sv1 healpix, sv1 tile cumulative, sv1 pernight.
import glob, numpy as np, pandas as pd, sys
from multiprocessing import Pool
from common import *
vi = load_vi()
print("VI rows", len(vi), "unique TARGETID", vi.TARGETID.nunique())
print(vi.groupby("VI_SET").VI_QUALITY.describe().round(2))
dup = vi[vi.TARGETID.duplicated(keep=False)].sort_values("TARGETID")
print("dup rows", len(dup)); print(dup.head(10).to_string())
tids = set(vi.TARGETID)
base = ["TARGETID"] + RR + OBS + PHOT + ["SV1_DESI_TARGET","SV1_BGS_TARGET","SV1_MWS_TARGET","SV1_SCND_TARGET","DESI_TARGET","SURVEY","PROGRAM","NIGHT","TILEID"]
jobs = [(p, "pn") for p in glob.glob(ZC+"ztile-sv1-*-pernight.fits")] + \
       [(p, "tc") for p in glob.glob(ZC+"ztile-sv1-*-cumulative.fits")] + \
       [(p, "px") for p in glob.glob(ZC+"zpix-sv1-*.fits")]
def one(j):
    p, tag = j
    d = read_match(p, base, tids); d["DEPTH"] = tag; d["SRCFILE"] = p.split("/")[-1]; return d
with Pool(len(jobs)) as pool:
    res = pool.map(one, jobs)
rr = pd.concat(res, ignore_index=True)
for c in ["SPECTYPE","SUBTYPE","SURVEY","PROGRAM","MORPHTYPE"]:
    if c in rr: rr[c] = rr[c].astype(str)
print(rr.groupby(["DEPTH","SRCFILE"]).size())
rr.to_parquet("/pscratch/sd/s/shunth/rrwrong/vi_rr_raw.parquet")
vi.to_parquet("/pscratch/sd/s/shunth/rrwrong/vi.parquet")
