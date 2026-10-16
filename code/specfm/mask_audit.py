"""How does BRZ_MASK relate to BRZ_IVAR in the clean-2 files?  Random 256-row blocks, both files."""
import h5py, numpy as np, sys
P = "/pscratch/sd/s/shunth/"
F = {"main": "dr1_main_primary_tgt_zwarn0_clean2_brz.h5", "sv3": "dr1_sv3_primary_tgt_zwarn0_clean2_brz.h5"}
NB = {"main": 80, "sv3": 40}
rng = np.random.default_rng(0)
LAM = np.arange(7781) * 0.8 + 3600.0
for name, fn in F.items():
    f = h5py.File(P + fn, "r"); n = f["BRZ_FLUX"].shape[0]
    print("== %s  rows %d  mask dtype %s" % (name, n, f["BRZ_MASK"].dtype))
    starts = rng.choice(n // 256, NB[name], replace=False) * 256
    tot = dict(pix=0, m=0, m_ivpos=0, m0_iv0=0, rows=0, rows_bad=0)
    bits = np.zeros(32, np.int64); bits_ivpos = np.zeros(32, np.int64)
    per_row_bad = []; lam_bad = np.zeros(7781, np.int64)
    for s in starts:
        m = f["BRZ_MASK"][s:s + 256].astype(np.int64); iv = f["BRZ_IVAR"][s:s + 256]
        bad = (m != 0) & (iv > 0)
        tot["pix"] += m.size; tot["m"] += int((m != 0).sum()); tot["m_ivpos"] += int(bad.sum())
        tot["m0_iv0"] += int(((m == 0) & (iv <= 0)).sum()); tot["rows"] += len(m); tot["rows_bad"] += int(bad.any(1).sum())
        per_row_bad += bad.sum(1).tolist(); lam_bad += bad.sum(0)
        for b in range(32):
            hit = (m >> b) & 1
            bits[b] += int(hit.sum()); bits_ivpos[b] += int((hit.astype(bool) & (iv > 0)).sum())
    pr = np.array(per_row_bad)
    print("  sampled rows %d, pixels %d" % (tot["rows"], tot["pix"]))
    print("  mask != 0                 : %.4f%% of pixels" % (100 * tot["m"] / tot["pix"]))
    print("  mask != 0 AND ivar > 0    : %.4f%% of pixels  (%d px)" % (100 * tot["m_ivpos"] / tot["pix"], tot["m_ivpos"]))
    print("  mask == 0 AND ivar <= 0   : %.4f%% of pixels" % (100 * tot["m0_iv0"] / tot["pix"]))
    print("  spectra with >=1 masked-but-ivar>0 pixel: %.2f%%; per such spectrum median %d, p90 %d, max %d px"
          % (100 * tot["rows_bad"] / tot["rows"], np.median(pr[pr > 0]) if (pr > 0).any() else 0,
             np.percentile(pr[pr > 0], 90) if (pr > 0).any() else 0, pr.max()))
    print("  mask bits set (pixel counts; of which ivar > 0):")
    for b in np.nonzero(bits)[0]:
        print("    bit %2d: %10d  (ivar>0: %d)" % (b, bits[b], bits_ivpos[b]))
    if lam_bad.sum():
        top = np.argsort(lam_bad)[::-1][:8]
        print("  wavelengths with most masked-but-ivar>0 pixels:", ", ".join("%.0fA(%d)" % (LAM[i], lam_bad[i]) for i in sorted(top)))
