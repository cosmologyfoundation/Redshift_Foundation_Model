# Independent main-survey check: pairs of tile coadds of the same target. Among pairs where BOTH rows are
# kept, the z-disagreement rate estimates ~2x the per-row error of the kept set. Model vs DELTACHI2 at equal keep.
import numpy as np, pandas as pd
from apply_model import p_wrong
P = "/pscratch/sd/s/shunth/rrwrong/"
d = pd.read_parquet(P+"main_repeats.parquet")
d = d.sort_values(["TARGETID","TILEID"]).groupby("TARGETID").head(2).reset_index(drop=True)
d = d[d.groupby("TARGETID").TARGETID.transform("size")==2].reset_index(drop=True)
d["P_WRONG"] = p_wrong(d)
d.to_parquet(P+"main_repeats_scored.parquet")
a = d.iloc[0::2].reset_index(drop=True); b = d.iloc[1::2].reset_index(drop=True)
assert (a.TARGETID.to_numpy()==b.TARGETID.to_numpy()).all()
qso = (a.SPECTYPE=="QSO")|(b.SPECTYPE=="QSO")
dis = (np.abs(a.Z-b.Z)/(1+np.minimum(a.Z,b.Z)) > np.where(qso, 0.01, 0.0033)).to_numpy()
t = a.DESI_TARGET.to_numpy().astype(np.int64)
cls = np.select([((t>>b)&1).astype(bool) for b in (2,0,1,60,61)], ["QSO","LRG","ELG","BGS","MWS"], "OTHER")
print("pairs", len(a), "overall disagreement", dis.mean().round(4)); print(pd.Series(cls).value_counts().to_dict())
def cmp(mask, name):
    A, B, D = a[mask], b[mask], dis[mask]
    base = lambda s: ((s.ZWARN==0)&(s.DELTACHI2>25)).to_numpy()
    kb = np.concatenate([base(A), base(B)]).mean()
    thr = np.quantile(np.concatenate([A.P_WRONG, B.P_WRONG]), kb)
    for lab, ka, kbb in [("zwarn0", (A.ZWARN==0).to_numpy(), (B.ZWARN==0).to_numpy()),
                         ("dchi2>25", base(A), base(B)),
                         ("model@same", (A.P_WRONG<=thr).to_numpy(), (B.P_WRONG<=thr).to_numpy())]:
        both = ka & kbb
        print(f"  {name:6s} {lab:11s} row-keep {np.concatenate([ka,kbb]).mean():.3f}  pairs kept {both.sum():7d}  disagree {D[both].mean():.4f} ({D[both].sum()})")
    for p in [0.01, 0.05, 0.1]:
        ka, kbb = (A.P_WRONG<=p).to_numpy(), (B.P_WRONG<=p).to_numpy(); both = ka & kbb
        print(f"  {name:6s} P<={p:<8} row-keep {np.concatenate([ka,kbb]).mean():.3f}  pairs kept {both.sum():7d}  disagree {D[both].mean():.4f} ({D[both].sum()})")
cmp(np.ones(len(a), bool), "ALL")
for c in ["QSO","LRG","ELG","BGS","MWS"]:
    if (cls==c).sum() > 500: cmp(cls==c, c)
