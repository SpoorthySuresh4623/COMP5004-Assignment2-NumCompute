"""Prequential (test-then-train) streaming evaluator with per-chunk logs."""
from time import perf_counter
import numpy as np
from ._validation import matrix, labels
from .metrics import StreamingAccuracy,RollingAccuracy


class StreamTrainer:
    """Evaluate unseen chunks before training; update model and log resources.

    First chunk(s) before first buffer retrain cannot be scored: per-chunk
    accuracy is NaN; those samples are not counted in cumulative accuracy.
    A log dict is appended for every call to `fit_chunk`.
    """
    def __init__(self,pipeline,rolling_window=100):
        if not hasattr(pipeline,'partial_fit') or not hasattr(pipeline,'predict'):
            raise ValueError('pipeline must support partial_fit and predict')
        self.pipeline=pipeline
        self.accuracy=StreamingAccuracy()
        self.rolling=RollingAccuracy(rolling_window)
        self.logs=[]
        self.chunks_seen=0
        self.samples_seen=0

    def score_chunk(self,X,y):
        x=matrix(X)
        target=labels(y,len(x))
        preds=self.pipeline.predict(x)
        return float(np.mean(preds==target))

    def _memory_bytes(self):
        """Approximate array storage of model state and pending training chunks."""
        obj=self.pipeline.model if hasattr(self.pipeline,'model') else self.pipeline
        total=0
        for arr in (getattr(obj,'_buffer_x',[])+getattr(obj,'_buffer_y',[])):
            if isinstance(arr,np.ndarray):total+=arr.nbytes
        def visit(node):
            if node is None:return 0
            return node.counts.nbytes+visit(node.left)+visit(node.right)
        total+=visit(getattr(obj,'root_',None))
        return int(total)

    def fit_chunk(self,X,y):
        x=matrix(X)
        target=labels(y,len(x))
        self.chunks_seen+=1
        self.samples_seen+=len(x)
        trained = self.pipeline.model if hasattr(self.pipeline,'model') else self.pipeline
        ready=getattr(trained,'root_',None) is not None
        chunk_accuracy=np.nan
        if ready:
            pred=self.pipeline.predict(x)
            self.accuracy.update(target,pred)
            self.rolling.update(target,pred)
            chunk_accuracy=float(np.mean(pred==target))
        old_retrain=getattr(trained,'retrain_count_',0)
        started=perf_counter()
        self.pipeline.partial_fit(x,target)
        elapsed=perf_counter()-started
        row={
            'chunk':self.chunks_seen,
            'samples_seen':self.samples_seen,
            'chunk_size':len(x),
            'chunk_accuracy':chunk_accuracy,
            'cumulative_accuracy':self.accuracy.result(),
            'rolling_accuracy':self.rolling.result(),
            'train_seconds':elapsed,
            'retrained':getattr(trained,'retrain_count_',0)>old_retrain,
            'buffered_samples':getattr(trained,'buffered_samples_',0),
            'retrain_count':getattr(trained,'retrain_count_',0),
            'memory_bytes':self._memory_bytes(),
        }
        self.logs.append(row)
        return row

    def reset_logs(self):
        self.logs.clear();self.chunks_seen=0;self.samples_seen=0
        self.accuracy.reset();self.rolling.reset()
        return self
