import unittest
import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numcompute_stream import (Pipeline,Imputer,StandardScaler,OneHotEncoder,
                               DecisionTreeClassifier,StreamTrainer)
from numcompute_stream.visualise import *


class PipelineStreamTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0],[0.2],[0.8],[1.]],dtype=float)
        self.y=np.array([0,0,1,1])
    def test_pipeline_chain_fit(self):
        p=Pipeline([('scale',StandardScaler()),('tree',DecisionTreeClassifier())]).fit(self.X,self.y)
        np.testing.assert_array_equal(p.predict(self.X),self.y)
    def test_pipeline_partial_fit(self):
        p=Pipeline([('imp',Imputer()),('tree',DecisionTreeClassifier(buffer_size=4))])
        p.partial_fit(self.X[:2],self.y[:2])
        p.partial_fit(self.X[2:],self.y[2:])
        np.testing.assert_array_equal(p.predict(self.X),self.y)
    def test_pipeline_duplicate_names(self):
        with self.assertRaises(ValueError):Pipeline([('x',Imputer()),('x',DecisionTreeClassifier())])
    def test_pipeline_missing_transform(self):
        with self.assertRaises(ValueError):Pipeline([('s',object()),('t',DecisionTreeClassifier())])
    def test_onehot_dynamic_prefix_prediction(self):
        p=Pipeline([('encode',OneHotEncoder()),('tree',DecisionTreeClassifier(buffer_size=2))])
        p.partial_fit([['red'],['blue']],[0,1])
        preds=p.predict([['red'],['blue']])
        p.partial_fit([['green']],[1])
        np.testing.assert_array_equal(p.predict([['red'],['blue']]),preds)
        self.assertEqual(p.model.buffered_samples_,1)
    def test_onehot_retrain_after_growing_width(self):
        p=Pipeline([('encode',OneHotEncoder()),('tree',DecisionTreeClassifier(buffer_size=2))])
        p.partial_fit([['red'],['blue']],[0,1]);p.partial_fit([['green']],[1]);p.partial_fit([['red']],[0])
        self.assertEqual(p.model.n_features_in_,3)
    def test_trainer_warmup_is_nan(self):
        p=Pipeline([('t',DecisionTreeClassifier(buffer_size=4))]);s=StreamTrainer(p)
        a=s.fit_chunk(self.X[:2],self.y[:2])
        self.assertTrue(np.isnan(a['chunk_accuracy']))
        self.assertEqual(a['retrain_count'],0)
    def test_trainer_retrain_logging(self):
        s=StreamTrainer(Pipeline([('t',DecisionTreeClassifier(buffer_size=2))]))
        a=s.fit_chunk(self.X[:2],self.y[:2]);b=s.fit_chunk(self.X[2:],self.y[2:])
        self.assertTrue(a['retrained']);self.assertEqual(b['chunk'],2)
        self.assertEqual(len(s.logs),2)
        self.assertEqual(b['samples_seen'],4)
    def test_trainer_score_chunk(self):
        s=StreamTrainer(Pipeline([('t',DecisionTreeClassifier(buffer_size=2))]))
        s.fit_chunk(self.X[:2],self.y[:2])
        self.assertEqual(s.score_chunk(self.X[:2],self.y[:2]),1)
    def test_trainer_score_unfitted(self):
        s=StreamTrainer(Pipeline([('t',DecisionTreeClassifier(buffer_size=5))]))
        with self.assertRaises(RuntimeError):s.score_chunk(self.X[:2],self.y[:2])
    def test_trainer_metrics_cumulative(self):
        s=StreamTrainer(Pipeline([('t',DecisionTreeClassifier(buffer_size=2))]))
        for i in [0,2]:s.fit_chunk(self.X[i:i+2],self.y[i:i+2])
        self.assertEqual(s.accuracy.total,2)
    def test_trainer_log_memory_nonnegative(self):
        s=StreamTrainer(Pipeline([('t',DecisionTreeClassifier(buffer_size=3))]))
        self.assertGreater(s.fit_chunk(self.X[:2],self.y[:2])['memory_bytes'],0)
    def test_metric_plot_save(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.png'
            fig=plot_metric_over_time([np.nan,.5,.8],save_path=p)
            self.assertTrue(p.exists());plt.close(fig)
    def test_model_compare_save(self):
        fig=compare_models([.5,.7],[.6,.8],labels=('small','large'))
        self.assertIsNotNone(fig);plt.close(fig)
    def test_predictions_plot_save(self):
        fig=plot_predictions_vs_ground_truth([0,1],[1,1])
        self.assertIsNotNone(fig);plt.close(fig)
