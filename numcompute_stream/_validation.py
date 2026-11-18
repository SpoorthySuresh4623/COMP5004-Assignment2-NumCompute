"""Small, shared validators.  Every public matrix is samples x features."""
import numpy as np


def matrix(X, *, allow_nan=True, allow_empty=False):
    try:
        x = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('X must contain numeric values') from exc
    if x.ndim != 2:
        raise ValueError(f'X must be 2D (samples, features); received shape {x.shape}')
    if x.shape[1] == 0 or (x.shape[0] == 0 and not allow_empty):
        raise ValueError('X needs at least one feature and (unless predicting) one sample')
    if np.isinf(x).any() or (not allow_nan and np.isnan(x).any()):
        raise ValueError('X contains invalid infinity or NaN values')
    return x


def labels(y, n=None):
    a = np.asarray(y)
    if a.ndim != 1 or (n is not None and len(a) != n) or len(a) == 0:
        raise ValueError('y must be a nonempty 1D array matching the samples in X')
    if a.dtype.kind in 'fc' and not np.isfinite(a).all():
        raise ValueError('y must not contain NaN or infinity')
    if a.dtype.kind not in 'biufUS':
        raise ValueError('y labels must be numeric or strings')
    return a


def pair(y_true, y_pred, *, scores=False):
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError('Inputs must be nonempty matching 1D arrays')
    if a.dtype.kind in 'fc' and not np.isfinite(a).all():
        raise ValueError('True labels cannot contain NaN or infinity')
    if scores and (b.dtype.kind not in 'biuf' or not np.isfinite(b).all()):
        raise ValueError('Scores must be finite numbers')
    if not scores and b.dtype.kind in 'fc' and not np.isfinite(b).all():
        raise ValueError('Predictions cannot contain NaN or infinity')
    return a, b
