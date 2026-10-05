# Out-of-fold P(wrong) from catalogue+imaging features, grouped by TARGETID; compare to DELTACHI2.
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from feats import features, CAT
P = "/pscratch/sd/s/shunth/rrwrong/"
d = pd.read_parquet(P+"vi_labeled.parquet").reset_index(drop=True)
# stars: VI star and Redrock |z|<0.005 counts as right (star velocity precision, not a redshift error)
star_ok = (d.VI_SPECTYPE.str.strip()=="STAR") & (d.Z.abs()<0.005)
d.loc[star_ok, "WRONG"] = 0
X = features(d, "SV1_DESI_TARGET"); y = d.WRONG.to_numpy()
cat = [X.columns.get_loc(c) for c in CAT]
oof = np.zeros(len(d))
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, d.TARGETID)):
    m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, categorical_features=cat, early_stopping=True,
            validation_fraction=0.1, random_state=k)
    m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
d["P_WRONG"] = oof
d[["TARGETID","DEPTH","SRCFILE","NIGHT","TILEID","WRONG","P_WRONG","Z","VI_Z","DELTACHI2","ZWARN","SPECTYPE","VI_SPECTYPE","VI_SET"]].to_parquet(P+"oof_cat.parquet")
def report(s, name):
    base = (s.ZWARN==0)&(s.DELTACHI2>25)
    kb = base.mean(); eb = s.WRONG[base].mean()
    thr = np.quantile(s.P_WRONG, kb); km = s.P_WRONG <= thr
    print(f"{name}: n={len(s)} wrong={s.WRONG.mean():.4f}  AUC dchi2={roc_auc_score(s.WRONG,-s.DELTACHI2):.3f} model={roc_auc_score(s.WRONG,s.P_WRONG):.3f}"
          f"  AP dchi2={average_precision_score(s.WRONG,-np.log10(s.DELTACHI2.clip(1e-2))):.3f} model={average_precision_score(s.WRONG,s.P_WRONG):.3f}")
    print(f"   same keep {kb:.3f}: baseline wrong {eb:.4f} ({s.WRONG[base].sum()})  model wrong {s.WRONG[km].mean():.4f} ({s.WRONG[km].sum()})")
    o = np.argsort(s.P_WRONG.to_numpy()); w = s.WRONG.to_numpy()[o]; cum = np.cumsum(w)/np.arange(1,len(w)+1)
    for tgt in [0.01, 0.005, 0.002]:
        ok = np.nonzero(cum <= tgt)[0]; k = (ok.max()+1)/len(w) if len(ok) else 0
        print(f"   keep at wrong<={tgt}: model {k:.3f}", end="")
        dc = s.sort_values("DELTACHI2", ascending=False); c2 = np.cumsum(dc.WRONG.to_numpy())/np.arange(1,len(dc)+1)
        ok2 = np.nonzero(c2 <= tgt)[0]; print(f"  dchi2-only {(ok2.max()+1)/len(dc) if len(ok2) else 0:.3f}")
for dep in ["pn","px"]:
    s = d[d.DEPTH==dep]; report(s, dep)
    for cls, ss in s.groupby("VI_SET"): report(ss, "  "+dep+" "+cls)
m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=cat, random_state=0).fit(X, y)
from sklearn.inspection import permutation_importance
s = d.DEPTH=="pn"
pi = permutation_importance(m, X[s].sample(20000, random_state=0), y[s][:20000] if False else d.WRONG[s].sample(20000, random_state=0), scoring="average_precision", n_repeats=3, random_state=0, n_jobs=16)
print(pd.Series(pi.importances_mean, X.columns).sort_values(ascending=False).head(20).round(4).to_string())
