"""Clean copy v2 of the training h5 (owner 2026-09-28: rebuild the clean set with rrwrong).

Delete a row if ANY of the v1 rule (clean/make_clean_h5.py: repeat coadd disagrees > 0.01(1+z) |
Gaia star-like & Z > 0.01 | star-only target & Z > 0.1 | DELTACHI2 < 25) OR P_WRONG > 0.01
on PROGRAM dark/bright, except Redrock STAR at |Z| < 0.005 that Gaia also calls star-like.
backup is exempt: VI has no backup-condition spectra and the model deleted 72% of Gaia-confirmed
stars there (2.0% in dark/bright) -- an extrapolation, not a measurement.
(rrwrong_model_v1, zfm_rrwrong_<name>.npz from score_training.py). Deletion is a strict superset of v1:
inside P<=0.01, DELTACHI2<25 rows are still 5-14x more often wrong (rule_check.py), so it stays.
Writer, chunk layout and verification come from v1 (imported); only the mask, names and attrs change.
"""
import os, sys, time, numpy as np, h5py
sys.path.insert(0, os.path.expanduser("~/claude_learning_map/clean"))
import make_clean_h5 as M
P = M.P
THR = 0.01
M.DST = {"main": "dr1_main_primary_tgt_zwarn0_clean2_brz.h5", "sv3": "dr1_sv3_primary_tgt_zwarn0_clean2_brz.h5"}
M.RULE = M.RULE + " | (PROGRAM in dark,bright & not Gaia-corroborated rest-frame STAR & P_WRONG > %g, rrwrong_model_v1)" % THR
_v1_mask = M.keep_mask

def keep_mask(name):
    r = np.load(P + "zfm_rrwrong_%s.npz" % name)
    assert r["found"].all(), "unmatched rows in rrwrong scores"
    return _v1_mask(name) & ~rr_delete(name, r["p_wrong"])

def rr_delete(name, p):
    c = np.load(P + "zfm_clean_%s.npz" % name)
    with h5py.File(P + M.SRC[name], "r") as f:
        pg = np.char.strip(f["PROGRAM"][:].astype("U")); st = np.char.strip(f["SPECTYPE"][:].astype("U"))
    corroborated_star = (st == "STAR") & (np.abs(c["z"]) < 0.005) & c["gaia_star"]
    return np.isin(pg, ["dark", "bright"]) & ~corroborated_star & (p > THR)
M.keep_mask = keep_mask

_v1_verify = M.verify
def verify(name, tmp, keep, kept):
    _v1_verify(name, tmp, keep, kept)
    r = np.load(P + "zfm_rrwrong_%s.npz" % name)
    with h5py.File(tmp, "r+") as d:
        assert not rr_delete(name, r["p_wrong"])[d["SRC_ROW"][:]].any(), "a row the rrwrong rule deletes survived"
        d.attrs["CLEAN_VERSION"] = 2
        d.attrs["CLEAN_RRWRONG_FILE"] = P + "zfm_rrwrong_%s.npz" % name
        d.attrs["CLEAN_RRWRONG_MODEL"] = P + "rrwrong/rrwrong_model_v1.joblib"
        d.attrs["CLEAN_RRWRONG_THR"] = THR
    M.log("%s: v2 check passed -- no kept row is deleted by the rrwrong rule (P_WRONG > %g, dark/bright)" % (name, THR))
M.verify = verify

if __name__ == "__main__":
    for name in sys.argv[1:]:
        M.main(name)
