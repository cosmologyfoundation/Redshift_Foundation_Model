"""P(Redrock wrong) for any DR1 zcatalog-like table (pandas DataFrame with Redrock + fibermap columns)."""
import numpy as np, joblib
from feats import features
def p_wrong(d, tgt_col="DESI_TARGET", path="/pscratch/sd/s/shunth/rrwrong/rrwrong_model_v1.joblib"):
    M = joblib.load(path)
    X = features(d, tgt_col)[M["columns"]]
    return np.mean([m.predict_proba(X)[:, 1] for m in M["models"]], axis=0)
