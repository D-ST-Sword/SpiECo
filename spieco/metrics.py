"""Task-specific evaluation; all accuracy/F1 values are fractions, not percentages."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def metrics(dataset, target, prediction, sims_clip=True):
    y = np.asarray(target, np.float64).reshape(-1)
    raw = np.asarray(prediction, np.float64).reshape(-1)
    if y.shape != raw.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(raw).all():
        raise ValueError("Expected aligned, nonempty, finite targets and predictions")
    p = raw.copy()
    if dataset == "sims" and sims_clip:
        y, p = np.clip(y, -1, 1), np.clip(p, -1, 1)
    result = {"mae": float(np.abs(y - p).mean()),
              "corr": float(np.corrcoef(y, p)[0, 1]) if y.std() > 1e-12 and p.std() > 1e-12 else 0.,
              "n_samples": len(y)}
    selected = y != 0 if dataset == "mosi" else np.ones(len(y), bool)
    if not selected.any():
        raise ValueError("No examples eligible for the binary evaluation")
    if dataset in ("urfunny", "mosi", "sims"):
        truth, pred = y[selected] > 0, p[selected] > 0
    else:
        raise ValueError(dataset)
    result.update(accuracy=float(accuracy_score(truth, pred)),
                  weighted_f1=float(f1_score(truth, pred, average="weighted", zero_division=0)),
                  classification_samples=int(selected.sum()))
    z = raw[selected]
    result["selection_classification_loss"] = float((np.logaddexp(0, z) - truth * z).mean())
    if dataset == "mosi":
        result["acc_7"] = float(accuracy_score(np.round(np.clip(y, -3, 3)), np.round(np.clip(p, -3, 3))))
    elif dataset == "sims":
        for count, bounds in ((3, [-.1, .1]), (5, [-.7, -.1, .1, .7])):
            result[f"acc_{count}"] = float(accuracy_score(np.digitize(y, bounds, right=True), np.digitize(p, bounds, right=True)))
    return result
