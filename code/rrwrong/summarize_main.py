# Keep fraction per target class for the filter vs the current clean rule, on full main zpix.
import numpy as np, pandas as pd
P = "/pscratch/sd/s/shunth/rrwrong/"
d = pd.concat([pd.read_parquet(P+f"zpix_main_{p}_rrwrong.parquet") for p in ["dark","bright"]], ignore_index=True)
d = d[(d.TARGETID>0)]
t = d.TGT.to_numpy().astype(np.int64)
d["cls"] = np.select([((t>>b)&1).astype(bool) for b in (2,0,1,60,61)], ["QSO","LRG","ELG","BGS","MWS"], "OTHER")
z0 = d.ZWARN==0
rows = []
for c, g in d.groupby("cls"):
    g0 = g[g.ZWARN==0]
    r = {"cls": c, "n": len(g), "zwarn0": (g.ZWARN==0).mean(), "zw0&dchi2>25": ((g.ZWARN==0)&(g.DELTACHI2>25)).mean()}
    for thr in [0.01, 0.05, 0.1]: r[f"P<={thr}"] = (g.P_WRONG<=thr).mean()
    r["P<=0.01 among zw0&dchi2>25"] = (g0[g0.DELTACHI2>25].P_WRONG<=0.01).mean()
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("all:", len(d), "P<=0.01", round((d.P_WRONG<=0.01).mean(),4), "P<=0.05", round((d.P_WRONG<=0.05).mean(),4))
