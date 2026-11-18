"""Chunk-wise numeric preprocessing and growing categorical encoding."""
import numpy as np
from ._validation import matrix


class StandardScaler:
    """Feature-wise running population mean/std using parallel Welford merges.

    NaNs are ignored in statistics and preserved in transform; infinite values
    are never accepted. Constant/all-missing features have unit scale.
    """
    def __init__(self):
        self.n_samples_seen_ = None
        self.mean_ = None
        self.m2_ = None
        self.n_features_in_ = None

    def fit(self,X):
        self.__init__()
        return self.partial_fit(X)

    def partial_fit(self,X):
        x=matrix(X)
        n,p=x.shape
        if self.n_features_in_ is None:
            self.n_features_in_=p
            self.n_samples_seen_=np.zeros(p,dtype=int)
            self.mean_=np.zeros(p,dtype=float)
            self.m2_=np.zeros(p,dtype=float)
        if p!=self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features; received {p}')
        ok=~np.isnan(x)
        count=ok.sum(axis=0)
        means=np.divide(np.nansum(x,axis=0),count,out=np.zeros(p),where=count>0)
        delta=np.where(ok,x-means,0.0)
        m2=np.sum(delta*delta,axis=0)
        old=self.n_samples_seen_
        total=old+count
        difference=means-self.mean_
        self.mean_+=np.divide(difference*count,total,out=np.zeros(p),where=total>0)
        self.m2_+=m2+np.divide(difference*difference*old*count,total,out=np.zeros(p),where=total>0)
        self.n_samples_seen_=total
        return self

    @property
    def var_(self):
        if self.mean_ is None:
            raise RuntimeError('StandardScaler must be fitted first')
        return np.divide(self.m2_,self.n_samples_seen_,out=np.zeros_like(self.m2_),where=self.n_samples_seen_>0)

    @property
    def scale_(self):
        var=self.var_
        return np.where(var>1e-24,np.sqrt(var),1.0)

    def transform(self,X):
        if self.mean_ is None:
            raise RuntimeError('StandardScaler must be fitted first')
        x=matrix(X,allow_empty=True)
        if x.shape[1]!=self.n_features_in_:
            raise ValueError('Feature count changed after StandardScaler fitting')
        return (x-self.mean_)/self.scale_

    def fit_transform(self,X):
        return self.fit(X).transform(X)


class Imputer:
    """Running per-feature mean imputation; all-missing features map to 0."""
    def __init__(self):
        self.sum_=None
        self.count_=None
        self.n_features_in_=None

    def fit(self,X):
        self.__init__()
        return self.partial_fit(X)

    def partial_fit(self,X):
        x=matrix(X)
        p=x.shape[1]
        if self.sum_ is None:
            self.n_features_in_=p
            self.sum_=np.zeros(p)
            self.count_=np.zeros(p,dtype=int)
        if p!=self.n_features_in_:
            raise ValueError('Feature count changed during Imputer update')
        self.sum_+=np.nansum(x,axis=0)
        self.count_+=(~np.isnan(x)).sum(axis=0)
        return self

    @property
    def statistics_(self):
        if self.sum_ is None:
            raise RuntimeError('Imputer must be fitted first')
        return np.divide(self.sum_,self.count_,out=np.zeros_like(self.sum_),where=self.count_>0)

    def transform(self,X):
        if self.sum_ is None:
            raise RuntimeError('Imputer must be fitted first')
        x=matrix(X,allow_empty=True)
        if x.shape[1]!=self.n_features_in_:
            raise ValueError('Feature count changed after Imputer fitting')
        return np.where(np.isnan(x),self.statistics_,x)

    def fit_transform(self,X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Incremental stable-index one-hot encoding of 2D categorical data.

    New categories append columns at the END of the matrix, leaving earlier
    feature positions unchanged. Unknown categories during transform are all 0.
    None/NaN are represented as one explicit missing-category token.
    """
    def __init__(self):
        self.n_features_in_=None
        self.categories_=[]
        self._mapping={}

    @staticmethod
    def _canonical(value):
        """Keep numeric categories distinct from equal-looking text labels."""
        if value is None:
            return ('missing', None)
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return ('missing', None)
        if isinstance(value, str):
            return ('text', value)
        if isinstance(value, (bool, np.bool_)):
            return ('boolean', bool(value))
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                raise ValueError('Numeric categories cannot be infinite')
            return ('number', float(value))
        raise ValueError('Categories must be strings, numbers, booleans, or missing')

    def fit(self,X):
        self.__init__()
        return self.partial_fit(X)

    def partial_fit(self,X):
        x=np.asarray(X,dtype=object)
        if x.ndim!=2 or x.shape[0]==0 or x.shape[1]==0:
            raise ValueError('OneHotEncoder needs nonempty 2D categorical X')
        if self.n_features_in_ is None:
            self.n_features_in_=x.shape[1]
            self.categories_=[[] for _ in range(x.shape[1])]
        if x.shape[1]!=self.n_features_in_:
            raise ValueError('Feature count changed during OneHotEncoder update')
        for j in range(x.shape[1]):
            for value in x[:,j]:
                key=(j,self._canonical(value))
                if key not in self._mapping:
                    self._mapping[key]=len(self._mapping)
                    self.categories_[j].append(value)
        return self

    @property
    def n_output_features_(self):
        return len(self._mapping)

    def transform(self,X):
        if self.n_features_in_ is None:
            raise RuntimeError('OneHotEncoder must be fitted first')
        x=np.asarray(X,dtype=object)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_:
            raise ValueError('OneHotEncoder expected a 2D array of original feature width')
        out=np.zeros((len(x),len(self._mapping)),dtype=float)
        # Categorical lookups cannot be vectorised without a third-party library.
        for j in range(x.shape[1]):
            ids=np.array([self._mapping.get((j,self._canonical(value)),-1)
                          for value in x[:,j]],dtype=int)
            valid=ids>=0
            out[np.flatnonzero(valid),ids[valid]]=1.0
        return out

    def fit_transform(self,X):
        return self.fit(X).transform(X)
