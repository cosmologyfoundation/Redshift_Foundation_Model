# Leave-one-field-out CV (fields = VI sky clusters) + ablations: no z, no COEFF, catalogue-only vs imaging-only.
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
from feats import features, CAT
P = "/pscratch/sd/s/shunth/rrwrong/"
d = pd.read_parquet(P+"vi_labeled.parquet").reset_index(drop=True)
d.loc[(d.VI_SPECTYPE.str.strip()=="STAR") & (d.Z.abs()<0.005), "WRONG"] = 0
vi = pd.read_parquet(P+"vi.parquet").drop_duplicates("TARGETID")
pos = vi.set_index("TARGETID")[["TARGET_RA","TARGET_DEC"]]
ra = pos.TARGET_RA.reindex(d.TARGETID).to_numpy(); dec = pos.TARGET_DEC.reindex(d.TARGETID).to_numpy()
field = (np.round(ra/6)*1000 + np.round(dec/6)).astype(int)
big = {5999:0, 18009:1, 18010:1, 17009:1, 25000:2, 25001:2}
field = np.array([big.get(int(f), 3) for f in field])
fs, fc = np.unique(field, return_counts=True); print("fields", len(fs), dict(zip(fs, fc)))
X = features(d, "SV1_DESI_TARGET"); y = d.WRONG.to_numpy()
coef = [c for c in X.columns if c.startswith("coef")]
imaging = [c for c in X.columns if c.startswith(("mag_","g_r","r_z","z_w1","w1_w2","fib","sn_","ebv","morph","shape","sersic","gaia","parallax","pm","maskbits","t_"))]
arms = {"all": list(X.columns), "no_z": [c for c in X.columns if c!="z"], "no_coef": [c for c in X.columns if c not in coef],
        "rr_only": [c for c in X.columns if c not in imaging], "dchi2+z+spectype": ["logdchi2","z","spectype"]}
pn = (d.DEPTH=="pn").to_numpy()
base = ((d.ZWARN==0)&(d.DELTACHI2>25)).to_numpy()
kb = base[pn].mean()
print(f"baseline pn keep {kb:.3f} wrong {y[pn&base].mean():.4f}")
for name, cols in arms.items():
    oof = np.zeros(len(d))
    for f in fs:
        te = field==f; tr = ~te
        m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                categorical_features=[c in CAT for c in cols], early_stopping=True, validation_fraction=0.1, random_state=0)
        m.fit(X.loc[tr, cols], y[tr]); oof[te] = m.predict_proba(X.loc[te, cols])[:, 1]
    p = oof[pn]; yy = y[pn]; thr = np.quantile(p, kb); k = p <= thr
    print(f"{name:18s} AUC pn {roc_auc_score(yy,p):.3f}  px {roc_auc_score(y[d.DEPTH=="px"], oof[(d.DEPTH=="px").to_numpy()]):.3f}  pn wrong at baseline keep {yy[k].mean():.4f} ({yy[k].sum()})")
    per = [(f, roc_auc_score(y[pn&(field==f)], oof[pn&(field==f)]), yy[k & (field[pn]==f)].mean()) for f in fs if y[pn&(field==f)].sum()>20]
    print("    per-field AUC / wrong-at-keep:", [(int(a), round(b,3), round(c,4)) for a,b,c in per])
