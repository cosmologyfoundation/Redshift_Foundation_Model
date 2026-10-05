# Fit the final P(Redrock wrong) model on all VI-labelled rows (all depths) and save it.
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from feats import features, CAT
P = "/pscratch/sd/s/shunth/rrwrong/"
d = pd.read_parquet(P+"vi_labeled.parquet").reset_index(drop=True)
d.loc[(d.VI_SPECTYPE.str.strip()=="STAR") & (d.Z.abs()<0.005), "WRONG"] = 0
X = features(d, "SV1_DESI_TARGET"); y = d.WRONG.to_numpy()
models = []
for seed in range(5):   # small bag for stability
    m = HGB(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            categorical_features=[c in CAT for c in X.columns], random_state=seed)
    idx = np.random.default_rng(seed).choice(len(d), len(d), replace=True)
    models.append(m.fit(X.iloc[idx], y[idx]))
joblib.dump({"models": models, "columns": list(X.columns)}, P+"rrwrong_model_v1.joblib")
oof = pd.read_parquet(P+"oof_cat.parquet")
pn = oof[oof.DEPTH=="pn"]
for q in [0.70,0.75,0.80,0.85,0.88,0.90]:
    t = np.quantile(pn.P_WRONG, q); k = pn.P_WRONG <= t
    print(f"keep {q:.2f}: P_WRONG<={t:.4f}  wrong {pn.WRONG[k].mean():.4f}")
print("saved")
