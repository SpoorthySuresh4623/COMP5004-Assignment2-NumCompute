"""Batch-compatible classification metrics and stateful per-chunk metrics.

Binary precision/recall/F1/AUC interpret label 1 as the positive class.
Missing denominator returns 0, while AUC on a single class returns NaN.
"""
from collections import deque
import numpy as np
from ._validation import pair


def accuracy(y_true,y_pred):
    a,b=pair(y_true,y_pred)
    return float(np.mean(a==b))


def _binary_counts(y_true,y_pred):
    a,b=pair(y_true,y_pred)
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Binary metrics require labels 0 and 1')
    return (int(((a==1)&(b==1)).sum()),int(((a==0)&(b==1)).sum()),
            int(((a==1)&(b==0)).sum()))


def precision(y_true,y_pred):
    tp,fp,_=_binary_counts(y_true,y_pred)
    return tp/(tp+fp) if tp+fp else 0.0


def recall(y_true,y_pred):
    tp,_,fn=_binary_counts(y_true,y_pred)
    return tp/(tp+fn) if tp+fn else 0.0


def f1(y_true,y_pred):
    tp,fp,fn=_binary_counts(y_true,y_pred)
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0


def confusion_matrix(y_true,y_pred,classes=None):
    a,b=pair(y_true,y_pred)
    classes=np.unique(np.concatenate([a,b])) if classes is None else np.asarray(classes)
    if classes.ndim!=1 or len(np.unique(classes))!=len(classes):
        raise ValueError('classes must be a unique 1D sequence')
    if not np.isin(a,classes).all() or not np.isin(b,classes).all():
        raise ValueError('Observed label not present in classes')
    t=np.searchsorted(classes,a) if np.array_equal(classes,np.sort(classes)) else np.array([np.flatnonzero(classes==v)[0] for v in a])
    p=np.searchsorted(classes,b) if np.array_equal(classes,np.sort(classes)) else np.array([np.flatnonzero(classes==v)[0] for v in b])
    return np.bincount(t*len(classes)+p,minlength=len(classes)**2).reshape(len(classes),len(classes))


def roc_curve(y_true,y_score):
    """Return (FPR,TPR,thresholds) with tied scores grouped."""
    y,s=pair(y_true,y_score,scores=True)
    if not np.isin(y,[0,1]).all():
        raise ValueError('ROC requires binary labels 0/1')
    order=np.argsort(-s,kind='mergesort')
    ys=y[order]
    scores=s[order]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
    tp=np.cumsum(ys==1)[ends]
    fp=np.cumsum(ys==0)[ends]
    positive=(y==1).sum()
    negative=(y==0).sum()
    tpr=np.r_[0.,tp/positive if positive else np.zeros(len(tp))]
    fpr=np.r_[0.,fp/negative if negative else np.zeros(len(fp))]
    return fpr,tpr,np.r_[np.inf,scores[ends]]


def auc(x,y):
    a,b=pair(x,y,scores=True)
    if (np.diff(a)<0).any():
        raise ValueError('AUC x coordinates must be nondecreasing')
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy 1.x compatibility
        integrate = np.trapz
    return float(integrate(b,a))


class StreamingAccuracy:
    """Cumulative accuracy across arbitrary-sized chunks."""
    def __init__(self): self.reset()
    def reset(self): self.correct=0; self.total=0; return self
    def update(self,y_true_chunk,y_pred_chunk):
        a,b=pair(y_true_chunk,y_pred_chunk)
        self.correct+=int((a==b).sum()); self.total+=len(a)
        return self
    def result(self): return self.correct/self.total if self.total else np.nan


class _StreamingBinary:
    def __init__(self): self.reset()
    def reset(self): self.tp=0; self.fp=0; self.fn=0; return self
    def update(self,y_true_chunk,y_pred_chunk):
        tp,fp,fn=_binary_counts(y_true_chunk,y_pred_chunk)
        self.tp+=tp;self.fp+=fp;self.fn+=fn
        return self


class StreamingPrecision(_StreamingBinary):
    def result(self): return self.tp/(self.tp+self.fp) if self.tp+self.fp else 0.0


class StreamingRecall(_StreamingBinary):
    def result(self): return self.tp/(self.tp+self.fn) if self.tp+self.fn else 0.0


class StreamingF1(_StreamingBinary):
    def result(self):
        d=2*self.tp+self.fp+self.fn
        return 2*self.tp/d if d else 0.0


class StreamingConfusionMatrix:
    """Accumulates multiclass confusion counts (true rows, predicted columns)."""
    def __init__(self,classes=None):
        self._fixed=np.asarray(classes) if classes is not None else None
        self.reset()
    def reset(self):
        self.classes_=np.asarray([]) if self._fixed is None else self._fixed.copy()
        self.counts_=np.zeros((len(self.classes_),len(self.classes_)),dtype=int)
        return self
    def update(self,y_true_chunk,y_pred_chunk):
        a,b=pair(y_true_chunk,y_pred_chunk)
        observed=np.unique(np.concatenate((a,b)))
        if self._fixed is not None and not np.isin(observed,self._fixed).all():
            raise ValueError('Observed label not in fixed class list')
        classes=np.unique(np.concatenate((self.classes_,observed))) if self._fixed is None else self._fixed
        if not np.array_equal(classes,self.classes_):
            revised=np.zeros((len(classes),len(classes)),dtype=int)
            pos=np.searchsorted(classes,self.classes_)
            revised[np.ix_(pos,pos)]=self.counts_
            self.classes_=classes
            self.counts_=revised
        self.counts_+=confusion_matrix(a,b,classes=self.classes_)
        return self
    def result(self): return self.counts_.copy()


class StreamingAUC:
    """Exact cumulative AUC; retains every score and class label (O(N) memory)."""
    def __init__(self): self.reset()
    def reset(self): self._y=[];self._s=[];return self
    def update(self,y_true_chunk,y_score_chunk):
        a,b=pair(y_true_chunk,y_score_chunk,scores=True)
        if not np.isin(a,[0,1]).all():
            raise ValueError('AUC requires binary class 0/1')
        self._y.append(a.copy()); self._s.append(np.asarray(b,dtype=float).copy())
        return self
    def result(self):
        if not self._y: return np.nan
        y=np.concatenate(self._y)
        if len(np.unique(y))<2: return np.nan
        fpr,tpr,_=roc_curve(y,np.concatenate(self._s))
        return auc(fpr,tpr)


class RollingAccuracy:
    """Exact accuracy across the most recent `window` samples, not chunks."""
    def __init__(self,window=100):
        if not isinstance(window,int) or window<1:
            raise ValueError('window must be a positive integer')
        self.window=window;self.reset()
    def reset(self): self._correct=deque(maxlen=self.window);return self
    def update(self,y_true_chunk,y_pred_chunk):
        a,b=pair(y_true_chunk,y_pred_chunk)
        self._correct.extend((a==b).tolist())
        return self
    def result(self): return float(np.mean(self._correct)) if self._correct else np.nan


class StreamingClassificationMetrics:
    """Paired running metrics; use scores only for optional cumulative ROC AUC."""
    def __init__(self,rolling_window=100):
        self.acc=StreamingAccuracy()
        self.prec=StreamingPrecision()
        self.rec=StreamingRecall()
        self.f1=StreamingF1()
        self.cm=StreamingConfusionMatrix()
        self.roc_auc=StreamingAUC()
        self.rolling=RollingAccuracy(rolling_window)
    def update(self,y_true,y_pred,y_score=None):
        for metric in [self.acc,self.prec,self.rec,self.f1,self.cm,self.rolling]:
            metric.update(y_true,y_pred)
        if y_score is not None: self.roc_auc.update(y_true,y_score)
        return self
    def reset(self):
        for metric in [self.acc,self.prec,self.rec,self.f1,self.cm,self.roc_auc,self.rolling]: metric.reset()
        return self
    def result(self):
        return dict(accuracy=self.acc.result(),precision=self.prec.result(),
                    recall=self.rec.result(),f1=self.f1.result(),rolling_accuracy=self.rolling.result(),
                    confusion_matrix=self.cm.result(),auc=self.roc_auc.result())
