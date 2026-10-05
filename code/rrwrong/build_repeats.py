# Main-survey targets observed on >=2 different tiles: independent Redrock runs on disjoint exposures.
import numpy as np, pandas as pd, fitsio
from common import *
F = ZC + "zall-tilecumulative-iron.fits"
h = fitsio.FITS(F)["ZCATALOG"]
k = h.read(columns=["TARGETID","SURVEY","TILEID","COADD_FIBERSTATUS"])
sv = np.char.strip(k["SURVEY"].astype("U"))
ok = (sv=="main") & (k["TARGETID"]>0) & (k["COADD_FIBERSTATUS"]==0)
idx = np.nonzero(ok)[0]
t = pd.DataFrame({"TARGETID": k["TARGETID"][idx].astype("i8"), "TILEID": k["TILEID"][idx].astype("i8"), "row": idx})
t = t.drop_duplicates(["TARGETID","TILEID"])
nt = t.groupby("TARGETID").TILEID.transform("nunique")
t = t[nt >= 2]
print("main rows", ok.sum(), "targets with >=2 tiles", t.TARGETID.nunique(), "rows", len(t))
rows = np.sort(t.row.to_numpy())
cols = ["TARGETID","TILEID","PROGRAM"] + RR + OBS + PHOT + ["DESI_TARGET","BGS_TARGET","MWS_TARGET"]
parts = [to_df(h.read(columns=cols, rows=rows[s:s+2_000_000])) for s in range(0, len(rows), 2_000_000)]
d = pd.concat(parts, ignore_index=True)
for c in ["SPECTYPE","SUBTYPE","PROGRAM","MORPHTYPE"]: d[c] = d[c].astype(str)
d.to_parquet("/pscratch/sd/s/shunth/rrwrong/main_repeats.parquet")
print(d.groupby("PROGRAM").size()); print(d.groupby("TARGETID").size().value_counts().head())
