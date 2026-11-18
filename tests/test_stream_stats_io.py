import unittest
import tempfile
from pathlib import Path
import numpy as np
from numcompute_stream.stats import StreamingStats,StreamingHistogram
from numcompute_stream.io import CSVReader


class StatsIOTests(unittest.TestCase):
    def test_stats_running(self):
        x=np.array([[1.,2.],[3.,4.],[5.,6.]])
        s=StreamingStats().update_stats(x[:1]).update_stats(x[1:])
        np.testing.assert_allclose(s.mean_,x.mean(0))
        np.testing.assert_allclose(s.variance_,x.var(0))
    def test_stats_nan(self):
        s=StreamingStats().update_stats([[np.nan,2],[np.nan,4]])
        np.testing.assert_allclose(s.mean_,[0,3])
        self.assertTrue(np.isnan(s.quantile(.5)[0]))
    def test_stats_quantile(self):
        s=StreamingStats().update_stats([[1],[2]]).update_stats([[3],[4]])
        self.assertEqual(s.quantile(.5)[0],2.5)
    def test_stats_invalid_q(self):
        s=StreamingStats().update_stats([[1]])
        with self.assertRaises(ValueError):s.quantile(2)
    def test_stats_feature_width(self):
        s=StreamingStats().update_stats([[1,2]])
        with self.assertRaises(ValueError):s.update_stats([[1]])
    def test_stats_reset(self):
        s=StreamingStats().update_stats([[1]]).reset()
        self.assertIsNone(s.count_)
    def test_histogram_counts(self):
        h=StreamingHistogram(bins=2,value_range=(0,2)).update([0,.5,1,2])
        np.testing.assert_array_equal(h.result()[0],[2,2])
    def test_histogram_window(self):
        h=StreamingHistogram(bins=2,value_range=(0,2),window=2).update([0,.5]).update([1,1.5])
        np.testing.assert_array_equal(h.result()[0],[0,2])
    def test_histogram_nan(self):
        h=StreamingHistogram(bins=2,value_range=(0,2)).update([np.nan,1])
        self.assertEqual(h.result()[0].sum(),1)
    def test_histogram_reset(self):
        h=StreamingHistogram().update([.2]).reset()
        self.assertEqual(h.result()[0].sum(),0)
    def test_csv_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('a,b,label\n1,2,0\n3,,1\n5,6,0\n')
            chunks=list(CSVReader(p).read_chunked(2))
            self.assertEqual([len(c[0]) for c in chunks],[2,1])
            self.assertTrue(np.isnan(chunks[0][0][1,1]))
    def test_csv_missing_header(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('x,y\n1,2\n')
            with self.assertRaises(ValueError):CSVReader(p).read()
    def test_csv_bad_numeric(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('a,label\noops,0\n')
            with self.assertRaises(ValueError):CSVReader(p).read()
