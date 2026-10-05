# Attach VI truth and the wrong/right label; print base rates and baselines per depth.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
P="/pscratch/sd/s/shunth/rrwrong/"
vi = pd.read_parquet(P+"vi.parquet"); rr = pd.read_parquet(P+"vi_rr_raw.parquet")
# one truth per TARGETID: drop targets whose duplicate VI entries disagree at quality>=2.5
g = vi[vi.VI_QUALITY>=2.5].groupby("TARGETID")
spread = g.VI_Z.agg(lambda s: s.max()-s.min())
bad = spread[spread > 0.0033*(1+g.VI_Z.min()[spread.index])].index
print("targets with conflicting good VI:", len(bad))
vi = vi.sort_values("VI_QUALITY", ascending=False).drop_duplicates("TARGETID")
vi = vi[(vi.VI_QUALITY>=2.5) & ~vi.TARGETID.isin(bad)]
d = rr.merge(vi[["TARGETID","VI_Z","VI_QUALITY","VI_SPECTYPE","VI_SET"]], on="TARGETID")
d = d[(d.COADD_FIBERSTATUS==0)]
tol = np.where(d.VI_SPECTYPE.str.strip()=="QSO", 0.01, 0.0033)
d["DV"] = (d.Z-d.VI_Z)/(1+d.VI_Z)
d["WRONG"] = (np.abs(d.DV) > tol).astype(int)
d.to_parquet(P+"vi_labeled.parquet")
for dep, s in d.groupby("DEPTH"):
    print(f"\n=== {dep}: rows {len(s)}  targets {s.TARGETID.nunique()}  wrong {s.WRONG.mean():.4f}")
    z0 = s[s.ZWARN==0]
    print(f"  ZWARN==0: {len(z0)/len(s):.3f} kept, wrong {z0.WRONG.mean():.4f}")
    print(s.groupby("VI_SET").agg(n=("WRONG","size"), wrong=("WRONG","mean"), wrong_z0=("WRONG", lambda w: w[s.loc[w.index,"ZWARN"]==0].mean())).round(4).to_string())
    for name, x in [("log DELTACHI2", np.log10(s.DELTACHI2.clip(1e-3))), ("-ZWARN!=0", -(s.ZWARN!=0).astype(float))]:
        print(f"  AUC(right vs wrong) {name}: {roc_auc_score(1-s.WRONG, x):.3f}")
    for cut in [9,25,40,100]:
        k = s[(s.ZWARN==0)&(s.DELTACHI2>cut)]
        print(f"  ZWARN0 & DCHI2>{cut}: keep {len(k)/len(s):.3f}  wrong {k.WRONG.mean():.4f}  ({k.WRONG.sum()} errors left)")
