"""P_WRONG for every row of the two training h5 files (owner 2026-09-28: rebuild the clean set with rrwrong).

Scores every spectrum of zall-pix-iron (28.4M) with rrwrong_model_v1, then maps onto the h5 rows by
(TARGETID, SURVEY, PROGRAM) exactly as clean/clean_labels.py does. Output zfm_rrwrong_<name>.npz:
row, p_wrong (float32), found.
"""
import time, numpy as np, pandas as pd, h5py, fitsio
from multiprocessing import Pool
from common import ZC, RR, OBS, PHOT, to_df
from apply_model import p_wrong
ZALL = ZC + "zall-pix-iron.fits"
P = "/pscratch/sd/s/shunth/"
H5 = {"main": "dr1_main_primary_tgt_zwarn0_brz.h5", "sv3": "dr1_sv3_primary_tgt_zwarn0_brz.h5"}
TGT = {"main": "DESI_TARGET", "sv1": "SV1_DESI_TARGET", "sv2": "SV2_DESI_TARGET", "sv3": "SV3_DESI_TARGET"}
COLS = ["TARGETID","SURVEY","PROGRAM"] + RR + OBS + PHOT + list(TGT.values())
t0 = time.time(); log = lambda *a: print("[%5.0fs]" % (time.time()-t0), *a, flush=True)

def score(se):
    s, e = se
    d = to_df(fitsio.FITS(ZALL)["ZCATALOG"][COLS][s:e])
    for c in ["SPECTYPE","SUBTYPE","SURVEY","PROGRAM","MORPHTYPE"]: d[c] = d[c].astype(str).str.strip()
    t = np.zeros(len(d), "u8")
    for sv, c in TGT.items():
        m = (d.SURVEY==sv).to_numpy(); t[m] = d[c].to_numpy()[m].astype("u8")
    d["TGT"] = t
    return (d.TARGETID.to_numpy().astype("i8"), d.SURVEY.to_numpy(), d.PROGRAM.to_numpy(),
            p_wrong(d, "TGT").astype("f4"))

if __name__ == "__main__":
    n = fitsio.FITS(ZALL)["ZCATALOG"].get_nrows()
    edges = np.linspace(0, n, 129).astype(int)
    with Pool(64) as pool:
        parts = pool.map(score, list(zip(edges[:-1], edges[1:])))
    key = pd.DataFrame({"tid": np.concatenate([p[0] for p in parts]), "survey": np.concatenate([p[1] for p in parts]),
                        "program": np.concatenate([p[2] for p in parts]), "p": np.concatenate([p[3] for p in parts])})
    log("zall scored: %d rows, P<=0.01 %.4f" % (len(key), (key.p<=0.01).mean()))
    for name, fn in H5.items():
        with h5py.File(P+fn, "r") as f:
            h = pd.DataFrame({"tid": f["TARGETID"][:].astype(np.int64),
                              "survey": np.char.strip(f["SURVEY"][:].astype("U")),
                              "program": np.char.strip(f["PROGRAM"][:].astype("U"))})
        h["row"] = np.arange(len(h))
        m = h.merge(key, on=["tid","survey","program"], how="left").drop_duplicates("row").sort_values("row")
        assert len(m) == len(h), name
        found = m.p.notna().to_numpy()
        np.savez(P+"zfm_rrwrong_%s.npz" % name, row=m.row.to_numpy(), p_wrong=m.p.fillna(1.0).to_numpy().astype("f4"), found=found)
        log("%s: %d rows matched %.4f%%  P<=0.01 %.4f  P<=0.05 %.4f" % (name, len(h), 100*found.mean(),
            (m.p<=0.01).mean(), (m.p<=0.05).mean()))
