"""Deterministic NumPy classification tree with chunk-buffered model refreshes.

The tree *does not* fit per chunk. It learns on the buffered chunks ONLY when
buffer_size is reached, then clears that buffer. Before its first refresh,
predict raises RuntimeError. After a refresh, prediction uses the last tree.
Each refresh replaces, rather than augments, its earlier training dataset.
"""
from dataclasses import dataclass
import numpy as np
from ._validation import matrix, labels


@dataclass
class _Node:
    label_index: int
    counts: np.ndarray
    feature: int = -1
    threshold: float = 0.0
    missing_left: bool = True
    left: object = None
    right: object = None

    @property
    def leaf(self):
        return self.feature == -1


class DecisionTreeClassifier:
    """Greedy numerical tree with Gini/entropy, reproducible tie-breaking.

    Parameters
    ----------
    max_depth : int or None
        Maximum root-to-leaf split count; None means unbounded.
    min_samples_split : int
        Minimum samples for a node to be split, at least 2.
    max_features : None, int, float, 'sqrt' or 'log2'
        Feature candidates per node; 0 < float <= 1.
    buffer_size : int
        Minimum number of accumulated samples triggering a new full retrain.
    criterion : {'gini','entropy'}
        Weighted impurity reduction to maximise.
    random_state : int
        Seed used only when max_features subsamples features.

    Feature matrix X has shape (n_samples,n_features). Numeric NaNs are allowed;
    for each split they are assigned to the side yielding greater gain. Infinite
    values are rejected. Labels may be integers or strings. Ties are broken by
    smaller feature index, smaller threshold, and then missing->left.
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion='gini', random_state=0):
        if max_depth is not None and (not isinstance(max_depth, (int,np.integer)) or max_depth < 0):
            raise ValueError('max_depth must be a nonnegative integer or None')
        if not isinstance(min_samples_split, (int,np.integer)) or min_samples_split < 2:
            raise ValueError('min_samples_split must be an integer >= 2')
        if not isinstance(buffer_size, (int,np.integer)) or buffer_size < 1:
            raise ValueError('buffer_size must be a positive integer')
        if criterion not in ('gini','entropy'):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if not (max_features is None or max_features in ('sqrt','log2') or
                (isinstance(max_features,(int,np.integer)) and max_features >= 1) or
                (isinstance(max_features,(float,np.floating)) and 0 < max_features <= 1)):
            raise ValueError("max_features must be None, a positive integer, (0,1], 'sqrt' or 'log2'")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.retrain_count_ = 0
        self.last_retrain_samples_ = 0
        self._buffer_x = []
        self._buffer_y = []
        self.buffered_samples_ = 0
        self._rng = np.random.default_rng(random_state)

    def _impurity(self, counts):
        total = counts.sum(axis=-1, keepdims=True)
        p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total != 0)
        if self.criterion == 'gini':
            return 1.0 - np.sum(p*p,axis=-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            terms = np.where(p > 0, p * np.log2(np.where(p > 0,p,1)), 0)
        return -np.sum(terms,axis=-1)

    def _feature_indices(self, p):
        m = self.max_features
        if m is None:
            return np.arange(p)
        if m == 'sqrt':
            k = max(1,int(np.sqrt(p)))
        elif m == 'log2':
            k = max(1,int(np.log2(p)))
        elif isinstance(m,(int,np.integer)):
            k = min(p,int(m))
        else:
            k = max(1,int(np.ceil(p * m)))
        return np.sort(self._rng.choice(p,size=k,replace=False))

    def _best_split(self, X, codes, parent_counts):
        """Evaluate all unique adjacent numerical thresholds via cumulative counts.

        For each feature: O(n log n) sort, then vectorised O(n*c) cumulative
        class tallies. Only a loop across features is needed.
        """
        n, p = X.shape
        before = float(self._impurity(parent_counts))
        best = None
        best_gain = 1e-12
        c = len(self.classes_)
        for feature in self._feature_indices(p):
            values = X[:,feature]
            finite_mask = ~np.isnan(values)
            n_finite = int(finite_mask.sum())
            n_missing = n - n_finite
            if n_finite == 0:
                continue
            # Missingness itself can be predictive, even if observed values
            # are constant. A -inf threshold sends NaNs left and all finite
            # observations right. Evaluate before continuous thresholds.
            if n_missing:
                missing_counts = np.bincount(codes[~finite_mask], minlength=c)
                other_counts = parent_counts - missing_counts
                after = (n_missing * float(self._impurity(missing_counts)) +
                         n_finite * float(self._impurity(other_counts))) / n
                gain = before - after
                if gain > best_gain + 1e-12:
                    best = (int(feature), float('-inf'), True)
                    best_gain = gain
            if n_finite < 2:
                continue
            order = np.argsort(values[finite_mask],kind='mergesort')
            vals = values[finite_mask][order]
            encoded = codes[finite_mask][order]
            boundaries = np.flatnonzero(vals[:-1] < vals[1:])
            if not len(boundaries):
                continue
            one_hot = np.eye(c,dtype=np.int64)[encoded]
            prefix = np.cumsum(one_hot,axis=0)
            left_base = prefix[boundaries].astype(float)
            nonmissing = prefix[-1].astype(float)
            missing_counts = parent_counts - nonmissing
            # Two candidate missing routes. A pure empty child is disallowed.
            left_counts = np.stack((left_base + missing_counts, left_base),axis=0)
            right_counts = parent_counts - left_counts
            ln = left_counts.sum(axis=-1)
            rn = right_counts.sum(axis=-1)
            valid = (ln > 0) & (rn > 0)
            weighted = (ln * self._impurity(left_counts) + rn * self._impurity(right_counts)) / n
            gains = np.where(valid, before - weighted, -np.inf)
            # np.argmax tie -> missing-left and earliest threshold.
            ind = np.unravel_index(np.argmax(gains),gains.shape)
            gain = float(gains[ind])
            # Check the gain before calculating midpoint to avoid overflow.
            if gain > best_gain + 1e-12:
                j = int(boundaries[ind[1]])
                # Half-sum prevents intermediate overflow for very large
                # opposite-signed floating point values.
                threshold = vals[j] / 2.0 + vals[j+1] / 2.0
                if threshold >= vals[j+1]:
                    threshold = vals[j]  # maintain <= split with adjacent floats
                best = (int(feature), float(threshold), bool(ind[0] == 0))
                best_gain = gain
        return best

    def _build(self,X,codes,depth):
        counts = np.bincount(codes,minlength=len(self.classes_))
        node = _Node(int(np.argmax(counts)), counts)
        if (counts.max() == len(codes) or len(codes) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node
        split = self._best_split(X,codes,counts)
        if split is None:
            return node
        feature, threshold, missing_left = split
        column = X[:,feature]
        left = ((column <= threshold) | np.isnan(column)) if missing_left else ((column <= threshold) & ~np.isnan(column))
        if not left.any() or left.all():
            return node
        node.feature, node.threshold, node.missing_left = feature,threshold,missing_left
        node.left = self._build(X[left],codes[left],depth+1)
        node.right = self._build(X[~left],codes[~left],depth+1)
        return node

    def fit(self,X,y):
        """Replace the trained tree directly; clear any pending streaming chunks."""
        X = matrix(X)
        y = labels(y,len(X))
        self._rng = np.random.default_rng(self.random_state)
        self.classes_, codes = np.unique(y,return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build(X,codes,0)
        self._buffer_x.clear()
        self._buffer_y.clear()
        self.buffered_samples_ = 0
        self.last_retrain_samples_ = len(X)
        self.retrain_count_ += 1
        return self

    def partial_fit(self,X_chunk,y_chunk):
        """Buffer a chunk; retrain from the entire buffer at a size threshold.

        If input feature count grows (as after an expanding one-hot encoder),
        zero-pad earlier pending chunks. This does not change a trained tree
        until the next retrain. Normal predict still requires its trained width.
        """
        X = matrix(X_chunk)
        y = labels(y_chunk,len(X))
        if self._buffer_x:
            width = max(X.shape[1], self._buffer_x[0].shape[1])
            self._buffer_x = [np.pad(b,((0,0),(0,width-b.shape[1]))) for b in self._buffer_x]
            X = np.pad(X,((0,0),(0,width-X.shape[1])))
        self._buffer_x.append(X.copy())
        self._buffer_y.append(y.copy())
        self.buffered_samples_ += len(y)
        if self.buffered_samples_ >= self.buffer_size:
            XX = np.concatenate(self._buffer_x,axis=0)
            yy = np.concatenate(self._buffer_y,axis=0)
            self.fit(XX, yy)  # resets the pending buffer after training
        return self

    def _predict_indices(self,X):
        if self.root_ is None:
            raise RuntimeError('Tree has not been trained; collect buffer_size samples first')
        x = matrix(X,allow_empty=True)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features; received {x.shape[1]}')
        out = np.empty(len(x),dtype=np.int64)

        def visit(node,ids):
            if ids.size == 0:
                return
            if node.leaf:
                out[ids] = node.label_index
                return
            col=x[ids,node.feature]
            goes_left = (col <= node.threshold) | np.isnan(col) if node.missing_left else (col <= node.threshold) & ~np.isnan(col)
            visit(node.left,ids[goes_left])
            visit(node.right,ids[~goes_left])
        visit(self.root_,np.arange(len(x)))
        return out

    def predict(self,X):
        return self.classes_[self._predict_indices(X)]

    def predict_proba(self,X):
        if self.root_ is None:
            raise RuntimeError('Tree has not been trained')
        x = matrix(X,allow_empty=True)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features; received {x.shape[1]}')
        out = np.empty((len(x),len(self.classes_)),dtype=float)

        def visit(node,ids):
            if ids.size == 0:
                return
            if node.leaf:
                out[ids,:] = node.counts / node.counts.sum()
                return
            col=x[ids,node.feature]
            goes_left = (col <= node.threshold) | np.isnan(col) if node.missing_left else (col <= node.threshold) & ~np.isnan(col)
            visit(node.left,ids[goes_left]); visit(node.right,ids[~goes_left])
        visit(self.root_,np.arange(len(x)))
        return out

    def summary(self):
        """Return tree size and depth for inspecting model complexity."""
        if self.root_ is None:
            return {'nodes':0,'leaves':0,'depth':0}
        def walk(n,d):
            if n.leaf:
                return 1,1,d
            a=walk(n.left,d+1); b=walk(n.right,d+1)
            return 1+a[0]+b[0],a[1]+b[1],max(a[2],b[2])
        nodes,leaves,depth = walk(self.root_,0)
        return {'nodes':nodes,'leaves':leaves,'depth':depth}
