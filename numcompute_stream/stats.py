"""Streaming sufficient statistics plus explicit-memory quantiles/histograms."""
from collections import deque
import numpy as np
from ._validation import matrix


class StreamingStats:
    """Exact mean/variance for observed values, ignoring NaNs independently.

    .update_stats(chunk) accepts 2D (samples,features) numeric data.
    Variance is population variance. Quantiles retain finite values explicitly;
    their memory use is O(number of observations), NOT constant memory.
    """
    def __init__(self):
        self.n_features_in_=None
        self.count_=None
        self.mean_=None
        self.m2_=None
        self._values=[]

    def update_stats(self,X_chunk):
        x=matrix(X_chunk)
        p=x.shape[1]
        if self.count_ is None:
            self.n_features_in_=p
            self.count_=np.zeros(p,dtype=int)
            self.mean_=np.zeros(p)
            self.m2_=np.zeros(p)
            self._values=[[] for _ in range(p)]
        if p!=self.n_features_in_:
            raise ValueError('Changing number of statistical features is unsupported')
        good=~np.isnan(x)
        nc=good.sum(axis=0)
        nm=np.divide(np.nansum(x,axis=0),nc,out=np.zeros(p),where=nc>0)
        nm2=np.sum(np.where(good,x-nm,0.0)**2,axis=0)
        total=self.count_+nc
        delta=nm-self.mean_
        self.mean_+=np.divide(delta*nc,total,out=np.zeros(p),where=total>0)
        self.m2_+=nm2+np.divide(delta*delta*self.count_*nc,total,out=np.zeros(p),where=total>0)
        self.count_=total
        for j in range(p):
            self._values[j].append(x[good[:,j],j].copy())
        return self

    update = update_stats

    @property
    def variance_(self):
        if self.count_ is None:
            raise RuntimeError('No chunks received')
        return np.divide(self.m2_,self.count_,out=np.zeros_like(self.m2_),where=self.count_>0)

    def quantile(self,q):
        if self.count_ is None:
            raise RuntimeError('No chunks received')
        if not np.isscalar(q) or not 0<=q<=1:
            raise ValueError('q must be between 0 and 1')
        return np.array([np.quantile(np.concatenate(col),q) if self.count_[j] else np.nan
                         for j,col in enumerate(self._values)])

    def reset(self):
        self.__init__()
        return self


class StreamingHistogram:
    """Exact histogram with fixed bin edges; optional rolling sample window.

    Edges are fixed for consistency across chunks. Values outside edges are
    ignored by np.histogram. NaNs are omitted. window tracks last N rows.
    """
    def __init__(self,bins=10,value_range=(0,1),window=None):
        if not isinstance(bins,int) or bins<1:
            raise ValueError('bins must be a positive integer')
        if (len(value_range)!=2 or not np.isfinite(value_range).all() or value_range[0]>=value_range[1]):
            raise ValueError('value_range must be two finite increasing numbers')
        if window is not None and (not isinstance(window,int) or window<1):
            raise ValueError('window must be a positive integer or None')
        self.edges_=np.linspace(value_range[0],value_range[1],bins+1)
        self.counts_=np.zeros(bins,dtype=np.int64)
        self.window=window
        self._history=deque()
        self._seen=0

    def update(self,values):
        x=np.asarray(values,dtype=float).ravel()
        if not np.isfinite(x[~np.isnan(x)]).all():
            raise ValueError('Histogram values cannot contain infinity')
        valid=x[~np.isnan(x)]
        self.counts_+=np.histogram(valid,bins=self.edges_)[0]
        if self.window is not None:
            self._history.extend(x.tolist())
            self._seen+=len(x)
            while self._seen>self.window:
                old=self._history.popleft()
                self._seen-=1
                if np.isfinite(old):
                    self.counts_-=np.histogram([old],bins=self.edges_)[0]
        return self

    update_stats=update

    def result(self):
        return self.counts_.copy(),self.edges_.copy()

    def reset(self):
        self.counts_.fill(0)
        self._history.clear()
        self._seen=0
        return self
