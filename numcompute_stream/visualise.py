"""Three reusable matplotlib visualisations for scripts and notebooks."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _finish(fig,save_path,show):
    if save_path is not None:
        path=Path(save_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=170,bbox_inches='tight')
    if show:plt.show()
    return fig


def plot_metric_over_time(metric_values,title='Streaming metric',ylabel='Value',*,
                          save_path=None,show=False):
    """Plot one metric per stream chunk, retaining NaN gaps for warm-up."""
    y=np.asarray(metric_values,dtype=float)
    if y.ndim!=1 or y.size==0:
        raise ValueError('metric_values must be a nonempty 1D array')
    fig,ax=plt.subplots(figsize=(8,3.8))
    ax.plot(np.arange(1,len(y)+1),y,marker='o',markersize=3,lw=1.6)
    ax.set(title=title,xlabel='Chunk',ylabel=ylabel)
    ax.grid(True,alpha=.2)
    fig.tight_layout()
    return _finish(fig,save_path,show)


def compare_models(metric1,metric2,labels=('Model A','Model B'),*,save_path=None,show=False):
    """Compare two 1D metric sequences, one measurement per chunk."""
    a=np.asarray(metric1,dtype=float);b=np.asarray(metric2,dtype=float)
    if a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or len(a)==0 or len(labels)!=2:
        raise ValueError('metric1, metric2 must be equally long 1D arrays; labels length 2')
    fig,ax=plt.subplots(figsize=(8,3.8))
    steps=np.arange(1,len(a)+1)
    ax.plot(steps,a,lw=1.7,marker='.',label=labels[0])
    ax.plot(steps,b,lw=1.7,marker='.',label=labels[1])
    ax.set(xlabel='Chunk',ylabel='Accuracy',title='Streaming model comparison')
    ax.legend();ax.grid(True,alpha=.2);fig.tight_layout()
    return _finish(fig,save_path,show)


def plot_predictions_vs_ground_truth(y_true,y_pred,*,save_path=None,show=False):
    """Show actual vs predicted labels on the latest scored chunk."""
    y=np.asarray(y_true);p=np.asarray(y_pred)
    if y.ndim!=1 or p.shape!=y.shape or len(y)==0:
        raise ValueError('y_true and y_pred must be nonempty equal-length 1D arrays')
    fig,ax=plt.subplots(figsize=(8,3.8))
    ids=np.arange(len(y))
    ax.scatter(ids,y,label='Ground truth',marker='o',s=38,alpha=.72)
    ax.scatter(ids,p,label='Prediction',marker='x',s=34,alpha=.8)
    ax.set(xlabel='Observation in latest chunk',ylabel='Class',title='Predictions vs ground truth')
    ax.legend();ax.grid(True,alpha=.2);fig.tight_layout()
    return _finish(fig,save_path,show)
