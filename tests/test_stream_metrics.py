import unittest
import numpy as np
from numcompute_stream.metrics import *


class MetricsTests(unittest.TestCase):
    def test_accuracy(self):self.assertAlmostEqual(accuracy([1,0,1],[1,1,1]),2/3)
    def test_precision(self):self.assertAlmostEqual(precision([1,0,1],[1,1,0]),.5)
    def test_recall(self):self.assertAlmostEqual(recall([1,0,1],[1,1,0]),.5)
    def test_f1(self):self.assertAlmostEqual(f1([1,0,1],[1,1,0]),.5)
    def test_zero_denominator(self):self.assertEqual(precision([0,0],[0,0]),0)
    def test_invalid_binary(self):
        with self.assertRaises(ValueError):precision([0,2],[0,1])
    def test_multiclass_confusion(self):
        np.testing.assert_array_equal(confusion_matrix([0,2,1],[0,1,2]),
                                      [[1,0,0],[0,0,1],[0,1,0]])
    def test_auc_reference(self):
        f,t,_=roc_curve([0,0,1,1],[.1,.4,.35,.8])
        self.assertAlmostEqual(auc(f,t),.75)
    def test_auc_ties(self):
        f,t,_=roc_curve([0,1],[.5,.5])
        self.assertAlmostEqual(auc(f,t),.5)
    def test_nonmonotonic_auc(self):
        with self.assertRaises(ValueError):auc([1,0],[0,1])
    def test_streaming_accuracy(self):
        m=StreamingAccuracy().update([1,0],[1,1]).update([1],[1])
        self.assertAlmostEqual(m.result(),2/3)
    def test_streaming_precision(self):
        m=StreamingPrecision().update([1,0],[1,1]).update([1],[0])
        self.assertAlmostEqual(m.result(),.5)
    def test_streaming_recall(self):
        m=StreamingRecall().update([1,0],[1,1]).update([1],[0])
        self.assertAlmostEqual(m.result(),.5)
    def test_streaming_f1(self):
        m=StreamingF1().update([1,0],[1,1]).update([1],[0])
        self.assertAlmostEqual(m.result(),.5)
    def test_dynamic_confusion_expansion(self):
        m=StreamingConfusionMatrix().update([1],[1]).update([0],[1])
        np.testing.assert_array_equal(m.result(),[[0,1],[0,1]])
    def test_fixed_confusion_invalid_label(self):
        with self.assertRaises(ValueError):StreamingConfusionMatrix([0,1]).update([2],[1])
    def test_rolling_samples_not_chunks(self):
        m=RollingAccuracy(window=3).update([0,0],[0,0]).update([0,0],[1,0])
        self.assertAlmostEqual(m.result(),2/3)
    def test_streaming_auc_invariant_to_chunks(self):
        y=np.array([0,0,1,1]);s=np.array([.1,.4,.35,.8])
        m=StreamingAUC().update(y[:1],s[:1]).update(y[1:],s[1:])
        self.assertAlmostEqual(m.result(),.75)
    def test_auc_single_class_nan(self):
        self.assertTrue(np.isnan(StreamingAUC().update([0,0],[.1,.8]).result()))
    def test_reset(self):
        m=StreamingAccuracy().update([0],[0]).reset()
        self.assertTrue(np.isnan(m.result()))
    def test_aggregate(self):
        m=StreamingClassificationMetrics().update([0,1],[0,1],[.1,.9])
        self.assertEqual(m.result()['accuracy'],1.0)
    def test_nan_predictions_rejected(self):
        with self.assertRaises(ValueError):accuracy([0],[np.nan])
    def test_empty_metric_rejected(self):
        with self.assertRaises(ValueError):accuracy([],[])
