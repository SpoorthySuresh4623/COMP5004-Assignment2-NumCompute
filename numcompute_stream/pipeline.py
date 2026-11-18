"""Simple, validated streaming transformer -> estimator pipeline."""
import numpy as np


class Pipeline:
    """Ordered named transformers ending with an estimator.

    Transformers must implement transform and fit/partial_fit. Pipeline updates
    each transformer before transforming its chunk; predictions do not update
    state. Categorical encoders can append stable one-hot columns. The final
    tree sees all new columns when it next retrains; before then it uses its
    old prefix of columns. For direct tree prediction feature count remains strict.
    """
    def __init__(self,steps):
        if not isinstance(steps,(list,tuple)) or len(steps)==0:
            raise ValueError('steps must contain at least one (name, component) tuple')
        names=[]
        for s in steps:
            if not isinstance(s,tuple) or len(s)!=2 or not isinstance(s[0],str):
                raise ValueError('Each step must be (string_name, component)')
            names.append(s[0])
        if len(set(names))!=len(names):
            raise ValueError('Pipeline step names must be unique')
        for _,transform in steps[:-1]:
            if not hasattr(transform,'transform') or not hasattr(transform,'partial_fit'):
                raise ValueError('Transformer must implement transform and partial_fit')
        if not hasattr(steps[-1][1],'partial_fit') or not hasattr(steps[-1][1],'predict'):
            raise ValueError('Final model must implement partial_fit and predict')
        self.steps=list(steps)
        self.named_steps=dict(steps)

    @property
    def model(self): return self.steps[-1][1]

    def _transformed(self,X,update=False):
        out=X
        for _,transform in self.steps[:-1]:
            if update: transform.partial_fit(out)
            out=transform.transform(out)
        return out

    def _prediction_features(self,transformed):
        x=np.asarray(transformed)
        expected=getattr(self.model,'n_features_in_',None)
        if expected is not None and x.ndim==2 and x.shape[1]>expected:
            return x[:,:expected] # newly discovered categories have stable appended indices
        if expected is not None and x.ndim==2 and x.shape[1]<expected:
            return np.pad(x,((0,0),(0,expected-x.shape[1])),constant_values=0)
        return transformed

    def partial_fit(self,X_chunk,y_chunk):
        return self._fit_call(X_chunk,y_chunk,incremental=True)

    def fit(self,X,y):
        return self._fit_call(X,y,incremental=False)

    def _fit_call(self,X,y,incremental):
        out=X
        for _,t in self.steps[:-1]:
            if incremental: t.partial_fit(out)
            else: t.fit(out)
            out=t.transform(out)
        if incremental: self.model.partial_fit(out,y)
        else: self.model.fit(out,y)
        return self

    def predict(self,X):
        return self.model.predict(self._prediction_features(self._transformed(X)))

    def predict_proba(self,X):
        return self.model.predict_proba(self._prediction_features(self._transformed(X)))
