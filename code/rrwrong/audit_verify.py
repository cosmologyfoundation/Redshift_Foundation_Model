"""Independent audit of redrock_verify_20260928/full_59055544 (FLAG = GOOD/SUSPECT/FILTERED/BAD) on main.
Truth-ish sources, none used by verify_redrock.py:
  known-wrong items (Gaia WD z>0.1; blind-read Redrock wrong / right)     zfm_knownwrong_eps2_items.npz
  main tile-level repeat pairs (different tiles, same program)            rrwrong/main_repeats_scored.parquet
  rrwrong P_WRONG and the clean-2 rule                                     zfm_rrwrong_main.npz, make_clean2_h5
"""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/global/homes/s/shunth/claude_learning_map/rrwrong")
import make_clean2_h5 as C
P = "/pscratch/sd/s/shunth/"; V = P + "redrock_verify_20260928/full_59055544/"
cols = ["SOURCE_ROW", "TARGETID", "PROGRAM", "Z", "SPECTYPE", "DESI_TARGET", "DELTACHI2", "SNR_MEDIAN_ABS", "FLAG", "VERIFY_REASON", "EVIDENCE_TYPE", "PREFILTER_PASS"]
v = pq.read_table(V + "main_verified.parquet", columns=cols).to_pandas()
v = v.sort_values("SOURCE_ROW").reset_index(drop=True)
assert (v.SOURCE_ROW.to_numpy() == np.arange(len(v))).all(), "SOURCE_ROW is not 0..N-1"
for c in ("PROGRAM", "SPECTYPE", "FLAG"): v[c] = v[c].astype(str).str.strip()
p = np.load(P + "zfm_rrwrong_main.npz")["p_wrong"]
keep2 = C.keep_mask("main"); keep1 = C._v1_mask("main")
v["P"] = p; v["KEEP2"] = keep2; v["KEEP1"] = keep1
t = v.DESI_TARGET.to_numpy().astype(np.int64)
bits = {"QSO": 2, "LRG": 0, "ELG": 1, "BGS": 60, "MWS": 61}
nb = sum(((t >> b) & 1) for b in bits.values())
v["CLS"] = np.select([((t >> b) & 1).astype(bool) & (nb == 1) for b in bits.values()], list(bits), "multi/none")
F = ["GOOD", "SUSPECT", "FILTERED", "BAD"]
pd.set_option("display.width", 220)
print("=== 1. flags by single-bit target class (row %)")
print((pd.crosstab(v.CLS, v.FLAG, normalize="index")[F] * 100).round(1).assign(n=v.CLS.value_counts()).to_string())
print("\n=== by SPECTYPE"); print((pd.crosstab(v.SPECTYPE, v.FLAG, normalize="index")[F] * 100).round(1).to_string())
print("\n=== spectrum SNR (median |flux|*sqrt(ivar)) by flag: median / 10th pct")
print(v.groupby("FLAG").SNR_MEDIAN_ABS.describe(percentiles=[.1, .5])[["count", "10%", "50%"]].round(2).to_string())
print("\n=== z distribution by flag (fraction in bins)")
zb = pd.cut(v.Z, [-1, 0.01, 0.5, 1.0, 1.6, 2.1, 5], labels=["<0.01", "0.01-0.5", "0.5-1", "1-1.6", "1.6-2.1", ">2.1"])
print((pd.crosstab(zb, v.FLAG, normalize="columns")[F] * 100).round(1).to_string())
print("\n=== top reasons: FILTERED / SUSPECT / BAD")
for f in ("FILTERED", "SUSPECT", "BAD"):
    print(f, v.VERIFY_REASON[v.FLAG == f].astype(str).value_counts().head(6).to_dict())
print("\n=== 2. known sets (independent): flag of each item")
it = np.load(P + "zfm_knownwrong_eps2_items.npz")
m = it["src"] == "main"
k = pd.DataFrame({"set": it["set"][m], "row": it["row"][m]})
k["FLAG"] = v.FLAG.to_numpy()[k.row]; k["KEEP2"] = keep2[k.row]
tab = pd.crosstab(k.set, k.FLAG).reindex(columns=F, fill_value=0); tab["clean2_kept"] = k.groupby("set").KEEP2.sum(); tab["n"] = k.set.value_counts()
print(tab.to_string())
print("\n=== 3. rrwrong vs flags")
print(v.groupby("FLAG").agg(n=("P", "size"), P_median=("P", "median"), P_gt_0_01=("P", lambda x: (x > 0.01).mean()),
      clean2_kept=("KEEP2", "mean"), clean1_kept=("KEEP1", "mean")).reindex(F).round(4).to_string())
print("GOOD & clean2-deleted: %d (%.2f%% of GOOD)" % (((v.FLAG == "GOOD") & ~v.KEEP2).sum(), 100 * (~v.KEEP2[v.FLAG == "GOOD"]).mean()))
print("\n=== 4. main tile repeat pairs: disagreement of the two tile z's, by the healpix row's selection")
r = pd.read_parquet(P + "rrwrong/main_repeats_scored.parquet", columns=["TARGETID", "PROGRAM", "Z", "SPECTYPE"])
a = r.iloc[0::2].reset_index(drop=True); b = r.iloc[1::2].reset_index(drop=True)
assert (a.TARGETID.to_numpy() == b.TARGETID.to_numpy()).all()
qso = (a.SPECTYPE.str.strip() == "QSO") | (b.SPECTYPE.str.strip() == "QSO")
dis = (np.abs(a.Z - b.Z) / (1 + np.minimum(a.Z, b.Z)) > np.where(qso, 0.01, 0.0033)).to_numpy()
pr = pd.DataFrame({"TARGETID": a.TARGETID.astype("i8"), "PROGRAM": a.PROGRAM.astype(str).str.strip(), "dis": dis})
key = v[["TARGETID", "PROGRAM", "FLAG", "KEEP2", "P", "CLS"]].copy(); key["TARGETID"] = key.TARGETID.astype("i8")
pr = pr.merge(key.drop_duplicates(["TARGETID", "PROGRAM"]), on=["TARGETID", "PROGRAM"], how="inner")
print("pairs matched to a training row:", len(pr), " overall disagreement %.4f" % pr.dis.mean())
out = []
for name, msk in [("all", np.ones(len(pr), bool)), ("GOOD", pr.FLAG == "GOOD"), ("SUSPECT", pr.FLAG == "SUSPECT"),
                  ("FILTERED", pr.FLAG == "FILTERED"), ("BAD", pr.FLAG == "BAD"), ("clean-2 kept", pr.KEEP2),
                  ("GOOD & clean-2 kept", (pr.FLAG == "GOOD") & pr.KEEP2), ("clean-2 kept, not GOOD", pr.KEEP2 & (pr.FLAG != "GOOD"))]:
    out.append((name, int(msk.sum()), msk.mean(), pr.dis[msk].mean(), int(pr.dis[msk].sum())))
print(pd.DataFrame(out, columns=["selection", "pairs", "share", "disagree", "n_dis"]).round(4).to_string(index=False))
kg = (pr.FLAG == "GOOD").mean(); thr = np.quantile(pr.P, kg)
m2 = pr.P <= thr
print("equal-keep test: GOOD keeps %.3f of pairs -> disagree %.4f;  lowest-P_WRONG %.3f -> disagree %.4f"
      % (kg, pr.dis[pr.FLAG == "GOOD"].mean(), m2.mean(), pr.dis[m2].mean()))
print("\n=== 5. GOOD by class: share of the class kept, repeat disagreement inside GOOD")
g = pr[pr.FLAG == "GOOD"].groupby("CLS").dis.agg(["size", "mean"]).round(4)
print(pd.concat([(v.FLAG == "GOOD").groupby(v.CLS).mean().round(3).rename("GOOD_share"), g], axis=1).to_string())
