"""Reproducible loop vs vector and buffer-size streaming benchmarks.

Run from project root: python benchmark/run_stream_benchmarks.py
Uses only stdlib, NumPy, matplotlib, and this project.
"""
import csv
import sys
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numcompute_stream import DecisionTreeClassifier,Pipeline,Imputer,StreamTrainer
from numcompute_stream.io import CSVReader


def loop_best_gain(x,y):
    """Pure Python-loop reference for best binary Gini split on one feature."""
    unique=np.unique(x)
    if len(unique)<2:return 0.0
    p=np.mean(y==1)
    before=2*p*(1-p)
    best=0.
    for j in range(len(unique)-1):
        threshold=(unique[j]+unique[j+1])/2
        left_y=[];right_y=[]
        for val,label in zip(x,y):
            if val<=threshold:left_y.append(int(label))
            else:right_y.append(int(label))
        if left_y and right_y:
            lp=sum(left_y)/len(left_y);rp=sum(right_y)/len(right_y)
            after=(len(left_y)*2*lp*(1-lp)+len(right_y)*2*rp*(1-rp))/len(y)
            best=max(best,before-after)
    return best


def vector_best_gain(x,y):
    """One sort + vectorised cumsum of positives at candidate boundaries."""
    order=np.argsort(x,kind='mergesort')
    xs=x[order]
    ys=y[order]
    n=len(y)
    boundaries=np.flatnonzero(xs[:-1]!=xs[1:])
    if not len(boundaries):return 0.0
    count_left=boundaries+1
    tp_left=np.cumsum(ys==1)[boundaries]
    count_right=n-count_left
    tp_right=int((y==1).sum())-tp_left
    p=(y==1).mean()
    before=2*p*(1-p)
    pl=tp_left/count_left
    pr=tp_right/count_right
    after=(count_left*2*pl*(1-pl)+count_right*2*pr*(1-pr))/n
    return float(np.maximum(0,before-after).max())


def benchmark_loops(sizes=(300,600,1200),seed=28):
    rng=np.random.default_rng(seed)
    output=[]
    for n in sizes:
        # Repeated integer feature values: only 35 candidate thresholds.
        x=rng.integers(0,36,size=n)
        y=(x>18).astype(int)
        t=[]
        for fn in (loop_best_gain,vector_best_gain):
            reps=[]
            for _ in range(4):
                started=perf_counter();out=fn(x,y);reps.append(perf_counter()-started)
            t.append((float(out),float(np.median(reps))))
        assert np.isclose(t[0][0],t[1][0],atol=1e-10), 'Reference and vector disagree'
        output.append(dict(samples=n,loop_seconds=t[0][1],vector_seconds=t[1][1],
                           speedup=t[0][1]/max(t[1][1],1e-12),best_gain=t[1][0]))
    return output


def benchmark_buffers(buffer_sizes=(40,120,300),chunk_size=30):
    output=[];history={}
    common_start = int(np.ceil(max(buffer_sizes)/chunk_size)) + 1
    for size in buffer_sizes:
        trainer=StreamTrainer(Pipeline([('impute',Imputer()),
           ('model',DecisionTreeClassifier(max_depth=5,min_samples_split=6,
                                            buffer_size=size,random_state=23))]))
        for X,y in CSVReader(ROOT/'demo'/'stream_data.csv').read_chunked(chunk_size):
            trainer.fit_chunk(X,y)
        logs=trainer.logs
        scored=[r for r in logs if np.isfinite(r['chunk_accuracy'])]
        mean_accuracy=float(np.mean([r['chunk_accuracy'] for r in scored])) if scored else np.nan
        retrains=[r['train_seconds'] for r in logs if r['retrained']]
        common=[r['chunk_accuracy'] for r in logs if r['chunk']>=common_start and np.isfinite(r['chunk_accuracy'])]
        row=dict(buffer_size=size,scored_chunks=len(scored),retrain_count=trainer.pipeline.model.retrain_count_,
                 mean_chunk_accuracy=mean_accuracy,
                 common_window_accuracy=float(np.mean(common)) if common else np.nan,
                 cumulative_accuracy=trainer.accuracy.result(),
                 mean_retrain_seconds=float(np.mean(retrains)) if retrains else np.nan,
                 total_training_seconds=sum(r['train_seconds'] for r in logs),
                 peak_memory_bytes=max(r['memory_bytes'] for r in logs))
        output.append(row)
        history[size]=logs
    return output,history


def _save_csv(rows,path):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def run():
    out=ROOT/'benchmark'/'stream_results'
    out.mkdir(parents=True,exist_ok=True)
    loop=benchmark_loops()
    sizes,history=benchmark_buffers()
    _save_csv(loop,out/'loop_vs_vector.csv')
    _save_csv(sizes,out/'buffer_comparison.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['samples'] for r in loop],[r['loop_seconds']*1000 for r in loop],marker='o',label='Python loops')
    ax.plot([r['samples'] for r in loop],[r['vector_seconds']*1000 for r in loop],marker='o',label='NumPy cumulative sums')
    ax.set(xlabel='Number of samples',ylabel='Median time (ms)',title='Best Gini split search: loop vs vector')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'loop_vs_vector.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    xs=[r['buffer_size'] for r in sizes]
    ax.bar([str(x) for x in xs],[r['mean_retrain_seconds']*1000 for r in sizes])
    ax.set(xlabel='Buffer size (samples)',ylabel='Mean model retrain duration (ms)',title='Retrain cost by buffer size')
    fig.tight_layout();fig.savefig(out/'buffer_retrain_time.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for s,logs in history.items():
        ax.plot([r['chunk'] for r in logs],
                [r['cumulative_accuracy'] for r in logs],label=f'buffer={s}')
    ax.set(xlabel='Chunk',ylabel='Cumulative prequential accuracy',title='Buffered tree streaming performance')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'buffer_accuracy.png',dpi=170);plt.close(fig)
    print('Loop comparison:')
    for row in loop:print(row)
    print('\nBuffer comparison:')
    for row in sizes:print(row)
    return loop,sizes


if __name__=='__main__':run()
