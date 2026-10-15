"""Binary-classification metrics used for every epoch and for the multi-seed report."""
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

METRIC_KEYS = ["AUC", "AUPR", "ACC", "F1", "MCC", "TPR", "TNR", "Precision", "SCC", "PCC"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true : {0,1};  y_prob : sigmoid(logit).
    AUC/AUPR/SCC/PCC are threshold-free; ACC/F1/MCC/TPR/TNR/Precision use `threshold`.
    SCC / PCC = Spearman / Pearson correlation between predicted probability and label.
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pred = (p >= threshold).astype(np.int64)
    out = {k: float("nan") for k in METRIC_KEYS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(np.unique(y)) == 2:
            out["AUC"] = roc_auc_score(y, p)
            out["AUPR"] = average_precision_score(y, p)
            if np.std(p) > 0:
                out["SCC"] = stats.spearmanr(p, y)[0]
                out["PCC"] = stats.pearsonr(p, y)[0]
        out["ACC"] = accuracy_score(y, pred)
        out["F1"] = f1_score(y, pred, zero_division=0)
        out["MCC"] = matthews_corrcoef(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["TPR"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out["TNR"] = tn / (tn + fp) if (tn + fp) else float("nan")
        out["Precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    return {k: float(v) for k, v in out.items()}
