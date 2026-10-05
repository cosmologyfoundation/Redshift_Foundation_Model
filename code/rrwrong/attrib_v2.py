# Where does v2 disagree with Redrock while v1 agrees (and vice versa)?  Map each 60k unfiltered validation
# object back to its h5 row by its Redrock z inside the validation blocks, then read P_WRONG and the v1/v2 rules.
import sys, glob, numpy as np, h5py
sys.path.insert(0, "/global/homes/s/shunth/claude_learning_map/zfm")
from data import FILES, BLK, P
import make_clean2_h5 as C
R = [np.load(p) for p in sorted(glob.glob("/global/homes/s/shunth/claude_learning_map/zfm/results/pred60k_raw_r[0-9]*.npz"))]
g = lambda k: np.concatenate([r[k] for r in R]); zt = g("zt")
arms = sorted(k[:-2] for k in R[0].files if k.endswith("_i"))
cat = {a: np.abs((g(a + "_i") - zt) / (1 + zt)) > 0.1 for a in arms}
# validation rows, same fixed blocks as Spectra(val=True)
fr = {}
for k in FILES:
    with h5py.File(P + FILES[k][0]) as f: fr[k] = f["Z"].shape[0]
tot = sum(fr.values())
vbm = 400; vb = {"main": vbm, "sv3": max(1, int(round(vbm * (fr["sv3"] / tot) / (fr["main"] / tot))))}
cand_z, cand_p, cand_v1, cand_v2 = [], [], [], []
for k in FILES:
    blocks = np.sort(np.random.default_rng(777).permutation(fr[k] // BLK)[:vb[k]])
    rows = np.concatenate([np.arange(b * BLK, (b + 1) * BLK) for b in blocks])
    with h5py.File(P + FILES[k][0]) as f: z = f["Z"][:][rows]
    p = np.load(P + "zfm_rrwrong_%s.npz" % k)["p_wrong"][rows]
    v1 = ~C._v1_mask(k)[rows]; v2 = ~C.keep_mask(k)[rows]
    cand_z.append(z); cand_p.append(p); cand_v1.append(v1); cand_v2.append(v2)
cz, cp, c1, c2 = map(np.concatenate, (cand_z, cand_p, cand_v1, cand_v2))
cz = cz.astype(np.float32).astype(np.float64); zt = zt.astype(np.float32).astype(np.float64); o = np.argsort(cz); i = np.clip(np.searchsorted(cz[o], zt), 0, len(cz) - 1)
hit = np.abs(cz[o][i] - zt) < 1e-6; idx = o[i]
print("objects %d, matched by exact Redrock z %.4f" % (len(zt), hit.mean()))
p, d1, d2 = cp[idx], c1[idx], c2[idx]
print("validation rows: deleted by v1 %.3f, extra deleted by v2 %.3f" % (d1.mean(), (d2 & ~d1).mean()))
for s in ("42", "43"):
    a1 = [a for a in arms if "_s%s_" % s in a and a.endswith("_cl")][0]; a2 = [a for a in arms if "_s%s_" % s in a and a.endswith("_cl2")][0]
    only2 = cat[a2] & ~cat[a1] & hit; only1 = cat[a1] & ~cat[a2] & hit
    for name, m in (("v2 disagrees, v1 agrees", only2), ("v1 disagrees, v2 agrees", only1)):
        print("seed %s  %-24s n %3d  | row deleted by v1 %3d  extra-deleted by v2 %3d  kept by both %3d | P_WRONG median %.3f, >0.5 %d"
              % (s, name, m.sum(), (m & d1).sum(), (m & d2 & ~d1).sum(), (m & ~d2).sum(), np.median(p[m]) if m.any() else np.nan, (m & (p > 0.5)).sum()))
