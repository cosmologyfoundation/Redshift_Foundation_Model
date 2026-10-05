import numpy as np, pandas as pd
P="/pscratch/sd/s/shunth/rrwrong/"
d = pd.read_parquet(P+"vi_labeled.parquet")
pd.set_option("display.width",250)
for dep in ["px","pn"]:
    s = d[(d.DEPTH==dep)&(d.ZWARN==0)&(d.DELTACHI2>25)]
    w = s[s.WRONG==1]
    print(f"\n=== {dep} confident errors: {len(w)} of {len(s)}")
    print(pd.crosstab(w.VI_SPECTYPE, w.SPECTYPE))
    adz = np.abs(w.DV)
    print("|dv| quantiles", adz.quantile([.1,.25,.5,.75,.9]).round(4).to_dict())
    print("near-threshold (<0.01):", (adz<0.01).mean().round(3))
    print(w.groupby("VI_SET").size())
    print(w[["VI_SET","VI_SPECTYPE","VI_Z","VI_QUALITY","SPECTYPE","Z","DELTACHI2"]].sample(25,random_state=1).round(4).to_string())
