"""Report figures (owner 2026-09-29): the same 60,032 unfiltered validation objects before and after the
clean-2 filter, and what the filter removes.  Model: v2 seed 42.  Objects are mapped to their h5 rows by
Redrock z inside the fixed validation blocks (100% matched, as in attrib_v2.py)."""
import sys, glob, numpy as np, h5py
sys.path.insert(0, "/global/homes/s/shunth/claude_learning_map/zfm")
from data import FILES, BLK, P
import make_clean2_h5 as C
STEM = "C4_300000_d320_s42_nc_fn_ev_ch_mu0.001_cl2"
R = [np.load(p) for p in sorted(glob.glob("/global/homes/s/shunth/claude_learning_map/zfm/results/pred60k_raw_r[0-9]*.npz"))]
g = lambda k: np.concatenate([r[k] for r in R]); zt = g("zt"); zp = g(STEM + "_i"); cls = g("cls")
fr = {}
for k in FILES:
    with h5py.File(P + FILES[k][0]) as f: fr[k] = f["Z"].shape[0]
vb = {"main": 400, "sv3": max(1, int(round(400 * fr["sv3"] / fr["main"])))}
cols = {k: [] for k in ("z", "rep", "gaia", "staronly", "dchi2", "rr")}
for k in FILES:
    blocks = np.sort(np.random.default_rng(777).permutation(fr[k] // BLK)[:vb[k]])
    rows = np.concatenate([np.arange(b * BLK, (b + 1) * BLK) for b in blocks])
    with h5py.File(P + FILES[k][0]) as f: z = f["Z"][:][rows]
    c = np.load(P + "zfm_clean_%s.npz" % k); p = np.load(P + "zfm_rrwrong_%s.npz" % k)["p_wrong"]
    cols["z"].append(z)
    cols["rep"].append((c["rep_dis"] > 0)[rows])
    cols["gaia"].append((c["gaia_star"] & (c["z"] > 0.01))[rows])
    cols["staronly"].append((c["star_only"] & (c["z"] > 0.1))[rows])
    cols["dchi2"].append((c["dchi2"] < 25)[rows])
    cols["rr"].append(C.rr_delete(k, p)[rows])
cols = {k: np.concatenate(v) for k, v in cols.items()}
cz = cols["z"].astype(np.float32).astype(np.float64); zq = zt.astype(np.float32).astype(np.float64)
o = np.argsort(cz); i = np.clip(np.searchsorted(cz[o], zq), 0, len(cz) - 1); idx = o[i]
assert (np.abs(cz[idx] - zq) < 1e-6).all(), "unmatched objects"
F = {k: v[idx] for k, v in cols.items() if k != "z"}
d = (zp - zt) / (1 + zt); cat = np.abs(d) > 0.1
ruleA = F["rep"] | F["gaia"] | F["staronly"]
removed = ruleA | F["dchi2"] | F["rr"]
np.savez("/pscratch/sd/s/shunth/rrwrong/report_60k.npz", zt=zt, zp=zp, cls=cls, cat=cat, ruleA=ruleA,
         dchi2=F["dchi2"], rr=F["rr"], removed=removed)
s1 = ruleA; s2 = F["dchi2"] & ~s1; s3 = F["rr"] & ~s1 & ~F["dchi2"]
print("objects %d  orange %d" % (len(zt), cat.sum()))
for name, m in (("rule A (repeat / Gaia / star-only)", s1), ("DELTACHI2 < 25", s2), ("P_WRONG > 0.01", s3), ("total removed", removed)):
    print("%-36s objects %5d (%.2f%%)   orange %4d (%.1f%% of orange)" % (name, m.sum(), 100 * m.mean(), (m & cat).sum(), 100 * (m & cat).sum() / cat.sum()))
print("orange left after filter: %d of %d kept objects (%.3f%%)" % ((cat & ~removed).sum(), (~removed).sum(), 100 * cat[~removed].mean()))
