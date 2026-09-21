"""Explicit v35 metrics. Historical metrics are retained for archive reproduction."""
import numpy as np


def direction_metrics(base, candidate, eps=1e-6):
    """Cosine only for moving/moving windows; report all degenerate cases.

    eps is the L2 norm threshold over all time/channel first differences.
    Zero vectors have no direction; they are never assigned a cosine.
    """
    x = np.diff(base.astype(np.float64), axis=1).reshape(len(base), -1)
    y = np.diff(candidate.astype(np.float64), axis=1).reshape(len(candidate), -1)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    mx, my = nx > eps, ny > eps
    valid = mx & my
    cos = np.full(len(x), np.nan)
    cos[valid] = np.clip(np.sum(x[valid]*y[valid], axis=1)/(nx[valid]*ny[valid]), -1, 1)
    return dict(moving_cosine=float(np.mean(cos[valid])) if valid.any() else None,
                moving_moving=int(valid.sum()), static_static=int((~mx & ~my).sum()),
                static_moving=int((~mx & my).sum()), moving_static=int((mx & ~my).sum()),
                base_static=int((~mx).sum()), increment_rmse=float(np.sqrt(np.mean((x-y)**2)))), cos


def equal_group_weights(group):
    _, inv, counts = np.unique(group, return_inverse=True, return_counts=True)
    w = 1.0/counts[inv]
    return w/w.sum()


def weighted_mmd(x, y, gamma, weights=None):
    """Biased squared RBF MMD using every row and fixed train-only gamma."""
    x = x.reshape(len(x), -1).astype(np.float64)
    y = y.reshape(len(y), -1).astype(np.float64)
    w = np.ones(len(x))/len(x) if weights is None else np.asarray(weights)/np.sum(weights)
    v = w if len(y) == len(x) else np.ones(len(y))/len(y)
    def kernel(a,b):
        sq = np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None,:]-2*a@b.T
        return np.exp(-gamma*np.maximum(sq,0))
    return float(max(0, w@kernel(x,x)@w + v@kernel(y,y)@v - 2*w@kernel(x,y)@v))
