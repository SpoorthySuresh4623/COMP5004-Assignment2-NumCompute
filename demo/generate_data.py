"""Reproducible numeric binary-classification stream; artificial concept drift."""
import csv
from pathlib import Path
import numpy as np


def generate_csv(path,n=1200,seed=2026):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,4))
    # Changes sign of feature 1 and decision offset in the second half.
    split=n//2
    score=X[:,0]*1.3 + X[:,1]*0.8 + X[:,2]*0.35
    score[split:]=X[split:,0]*1.3 - X[split:,1]*0.8 + X[split:,2]*0.35-0.25
    y=(score+rng.normal(scale=.33,size=n)>0).astype(int)
    missing=rng.uniform(size=X.shape)<.04
    X[missing]=np.nan
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['f1','f2','f3','f4','label'])
        for row,label in zip(X,y):
            writer.writerow(['' if np.isnan(v) else f'{v:.8f}' for v in row]+[int(label)])
    return path


if __name__=='__main__':generate_csv(Path(__file__).with_name('stream_data.csv'))
