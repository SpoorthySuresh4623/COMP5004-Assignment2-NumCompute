"""Dependency-free CSV loading with chunk iteration. Numeric feature columns."""
import csv
from pathlib import Path
import numpy as np


class CSVReader:
    """Read a CSV with a header: one label column and numeric feature columns.

    Empty cells and the literal 'nan' become np.nan; infinity is rejected.
    `read_chunked(chunk_size)` yields (X,y) arrays of shapes (k,p),(k,).
    """
    def __init__(self,path,target='label',feature_columns=None):
        self.path=Path(path)
        self.target=target
        self.feature_columns=feature_columns

    def _rows(self):
        with self.path.open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or self.target not in reader.fieldnames:
                raise ValueError(f'CSV must contain a {self.target!r} target header')
            cols=self.feature_columns or [c for c in reader.fieldnames if c!=self.target]
            if not cols or not set(cols).issubset(reader.fieldnames):
                raise ValueError('CSV feature columns are empty or missing')
            for row_index,row in enumerate(reader,start=2):
                try:
                    x=[float(row[c]) if row[c] and row[c].strip() else np.nan for c in cols]
                    if not np.isfinite(np.asarray(x)[~np.isnan(x)]).all():
                        raise ValueError('infinite feature')
                    raw=row[self.target]
                    if raw is None or raw.strip()=='':
                        raise ValueError('blank target')
                    try: label=int(raw)
                    except ValueError: label=raw
                except (ValueError,TypeError) as exc:
                    raise ValueError(f'Invalid CSV data at line {row_index}: {exc}') from exc
                yield x,label

    def read_chunked(self,chunk_size=64):
        if not isinstance(chunk_size,int) or chunk_size<1:
            raise ValueError('chunk_size must be a positive integer')
        xs=[];ys=[]
        for x,y in self._rows():
            xs.append(x);ys.append(y)
            if len(xs)>=chunk_size:
                yield np.asarray(xs,dtype=float),np.asarray(ys)
                xs=[];ys=[]
        if xs:yield np.asarray(xs,dtype=float),np.asarray(ys)

    def read(self):
        chunks=list(self.read_chunked())
        if not chunks:
            raise ValueError('CSV has no data rows')
        return np.concatenate([x for x,_ in chunks]),np.concatenate([y for _,y in chunks])


def read_csv(path,target='label'):
    return CSVReader(path,target=target).read()
