# Known-wrong sets (Gaia WD z>0.1; my frozen blind reads) and their controls: how many does each filter remove?
import numpy as np, pandas as pd, h5py
from common import *
from apply_model import p_wrong
P = "/pscratch/sd/s/shunth/"
it = np.load(P+"zfm_knownwrong_eps2_items.npz")
H = {"main": "dr1_main_primary_tgt_zwarn0_brz.h5", "sv3": "dr1_sv3_primary_tgt_zwarn0_brz.h5"}
rows = []
for src in ["main","sv3"]:
    m = it["src"]==src
    with h5py.File(P+H[src]) as f:
        r = it["row"][m]; o = np.argsort(r)
        tid = f["TARGETID"][:][r]; sv = f["SURVEY"][:][r]; pg = f["PROGRAM"][:][r]; z = f["Z"][:][r]
    rows.append(pd.DataFrame({"set": it["set"][m], "TARGETID": tid.astype("i8"), "SURVEY": [s.decode().strip() if isinstance(s, bytes) else str(s).strip() for s in sv],
                              "PROGRAM": [s.decode().strip() if isinstance(s, bytes) else str(s).strip() for s in pg], "z_h5": z, "z_rr": it["z_rr"][m]}))
k = pd.concat(rows, ignore_index=True)
cols = ["TARGETID","SURVEY","PROGRAM"] + RR + OBS + PHOT + ["DESI_TARGET","SV3_DESI_TARGET"]
z = read_match(ZC+"zall-pix-iron.fits", cols, set(k.TARGETID))
for c in ["SPECTYPE","SUBTYPE","SURVEY","PROGRAM","MORPHTYPE"]: z[c] = z[c].astype(str).str.strip()
d = k.merge(z, on=["TARGETID","SURVEY","PROGRAM"], how="left")
print("matched", d.Z.notna().sum(), "of", len(d), " z agree with h5:", np.allclose(d.Z, d.z_h5, atol=1e-6))
d["TGT"] = np.where(d.SURVEY=="sv3", d.SV3_DESI_TARGET, d.DESI_TARGET).astype("u8")
d["P_WRONG"] = p_wrong(d, "TGT")
d.to_parquet(P+"rrwrong/known_scored.parquet")
rep = []
for s, g in d.groupby("set"):
    r = {"set": s, "n": len(g), "dchi2<=25": (g.DELTACHI2<=25).mean()}
    for t in [0.01, 0.05, 0.1, 0.5]: r[f"P>{t}"] = (g.P_WRONG>t).mean()
    rep.append(r)
print("fraction REMOVED by each filter (want high on *_wrong, low on controls)")
print(pd.DataFrame(rep).round(3).to_string(index=False))
