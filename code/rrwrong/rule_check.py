# Within rows the model keeps (P<=thr), does DELTACHI2<25 still mark more errors? VI (OOF) and main repeat pairs.
import numpy as np, pandas as pd
P = "/pscratch/sd/s/shunth/rrwrong/"
o = pd.read_parquet(P+"oof_cat.parquet")
for thr in [0.01, 0.05]:
    print(f"--- P<={thr}")
    for dep in ["pn","px"]:
        s = o[(o.DEPTH==dep)&(o.P_WRONG<=thr)]
        lo = s.DELTACHI2<25
        print(f"VI {dep}: kept {len(s)}  dchi2<25 share {lo.mean():.4f}  wrong(dchi2<25) {s.WRONG[lo].mean():.4f} ({s.WRONG[lo].sum()})  wrong(>=25) {s.WRONG[~lo].mean():.4f} ({s.WRONG[~lo].sum()})")
    d = pd.read_parquet(P+"main_repeats_scored.parquet", columns=["TARGETID","Z","SPECTYPE","DELTACHI2","P_WRONG"])
    a = d.iloc[0::2].reset_index(drop=True); b = d.iloc[1::2].reset_index(drop=True)
    qso = (a.SPECTYPE=="QSO")|(b.SPECTYPE=="QSO")
    dis = (np.abs(a.Z-b.Z)/(1+np.minimum(a.Z,b.Z)) > np.where(qso,0.01,0.0033)).to_numpy()
    both = ((a.P_WRONG<=thr)&(b.P_WRONG<=thr)).to_numpy()
    lo = ((a.DELTACHI2<25)|(b.DELTACHI2<25)).to_numpy()
    print(f"repeats: kept pairs {both.sum()}  with a dchi2<25 row {(both&lo).sum()}  disagree {dis[both&lo].mean():.4f}  vs others {dis[both&~lo].mean():.4f}")
