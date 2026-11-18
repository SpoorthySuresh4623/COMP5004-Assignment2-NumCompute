import unittest
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0.0],[0.2],[0.8],[1.0]])
        self.y=np.array([0,0,1,1])

    def test_fit_predict(self):
        t=DecisionTreeClassifier().fit(self.X,self.y)
        np.testing.assert_array_equal(t.predict(self.X),self.y)

    def test_predict_proba(self):
        t=DecisionTreeClassifier().fit(self.X,self.y)
        probs=t.predict_proba(self.X)
        self.assertEqual(probs.shape,(4,2))
        np.testing.assert_allclose(probs.sum(axis=1),1)

    def test_entropy_criterion(self):
        t=DecisionTreeClassifier(criterion='entropy').fit(self.X,self.y)
        self.assertTrue(np.array_equal(t.predict(self.X),self.y))

    def test_depth_zero_majority(self):
        t=DecisionTreeClassifier(max_depth=0).fit(self.X,np.array([1,1,1,0]))
        self.assertEqual(t.summary()['depth'],0)
        self.assertTrue((t.predict(self.X)==1).all())

    def test_min_samples_split(self):
        t=DecisionTreeClassifier(min_samples_split=10).fit(self.X,self.y)
        self.assertEqual(t.summary()['nodes'],1)

    def test_constant_features_no_split(self):
        t=DecisionTreeClassifier().fit(np.ones((4,3)),self.y)
        self.assertEqual(t.summary()['leaves'],1)

    def test_ties_choose_smallest_label(self):
        t=DecisionTreeClassifier(max_depth=0).fit(self.X,self.y)
        self.assertTrue((t.predict(self.X)==0).all())

    def test_ties_choose_first_feature(self):
        X=np.column_stack([self.X[:,0],self.X[:,0]])
        t=DecisionTreeClassifier().fit(X,self.y)
        self.assertEqual(t.root_.feature,0)

    def test_string_classes(self):
        y=np.array(['no','no','yes','yes'])
        self.assertTrue(np.array_equal(DecisionTreeClassifier().fit(self.X,y).predict(self.X),y))

    def test_nan_training_and_prediction(self):
        X=np.array([[0.],[1.],[np.nan],[1.],[np.nan]])
        y=np.array([0,1,0,1,0])
        t=DecisionTreeClassifier().fit(X,y)
        np.testing.assert_array_equal(t.predict(X),y)

    def test_missingness_predictive_with_constant_observations(self):
        X=np.array([[1.],[1.],[np.nan],[np.nan]])
        y=np.array([0,0,1,1])
        model=DecisionTreeClassifier().fit(X,y)
        np.testing.assert_array_equal(model.predict(X),y)

    def test_extreme_float_split(self):
        X=np.array([[-1e308],[-1e300],[1e300],[1e308]])
        y=np.array([0,0,1,1])
        model=DecisionTreeClassifier().fit(X,y)
        np.testing.assert_array_equal(model.predict(X),y)

    def test_nan_route_can_be_right(self):
        X=np.array([[0.],[0.1],[0.9],[1.0],[np.nan],[np.nan]])
        y=np.array([0,0,1,1,1,1])
        t=DecisionTreeClassifier().fit(X,y)
        self.assertFalse(t.root_.missing_left)
        np.testing.assert_array_equal(t.predict(X),y)

    def test_reject_inf(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit([[np.inf]], [0])

    def test_reject_dimension(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit([0,1],[0,1])

    def test_reject_mismatched_y(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit(self.X,[0])

    def test_reject_bad_hyperparams(self):
        for kwargs in [dict(max_depth=-1),dict(buffer_size=0),dict(min_samples_split=1),
                       dict(criterion='other'),dict(max_features=0)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):DecisionTreeClassifier(**kwargs)

    def test_fit_clears_pending(self):
        t=DecisionTreeClassifier(buffer_size=5)
        t.partial_fit(self.X,self.y)
        t.fit(self.X,self.y)
        self.assertEqual(t.buffered_samples_,0)

    def test_not_fitted_predict(self):
        with self.assertRaises(RuntimeError):DecisionTreeClassifier().predict(self.X)

    def test_not_fitted_predict_proba(self):
        with self.assertRaises(RuntimeError):DecisionTreeClassifier().predict_proba(self.X)

    def test_buffer_no_early_fit(self):
        t=DecisionTreeClassifier(buffer_size=5).partial_fit(self.X,self.y)
        self.assertEqual(t.buffered_samples_,4)
        self.assertEqual(t.retrain_count_,0)

    def test_exact_boundary_retrain(self):
        t=DecisionTreeClassifier(buffer_size=4).partial_fit(self.X,self.y)
        self.assertEqual(t.buffered_samples_,0)
        self.assertEqual(t.retrain_count_,1)

    def test_threshold_overshoot_includes_all(self):
        t=DecisionTreeClassifier(buffer_size=3).partial_fit(self.X,self.y)
        self.assertEqual(t.last_retrain_samples_,4)

    def test_no_partial_fit_overwrite_between_retrains(self):
        t=DecisionTreeClassifier(buffer_size=4).partial_fit(self.X,self.y)
        first=t.predict(self.X)
        t.partial_fit(self.X[:2],(1-self.y)[:2])
        np.testing.assert_array_equal(t.predict(self.X),first)
        self.assertEqual(t.buffered_samples_,2)
        self.assertEqual(t.retrain_count_,1)

    def test_new_retrain_replaces_old_data(self):
        t=DecisionTreeClassifier(buffer_size=4).partial_fit(self.X,self.y)
        t.partial_fit(self.X,1-self.y)
        self.assertTrue(np.array_equal(t.predict(self.X),1-self.y))
        self.assertEqual(t.retrain_count_,2)

    def test_feature_width_mismatch_predict(self):
        t=DecisionTreeClassifier().fit(self.X,self.y)
        with self.assertRaises(ValueError):t.predict(np.zeros((2,2)))

    def test_empty_prediction_shape(self):
        t=DecisionTreeClassifier().fit(self.X,self.y)
        self.assertEqual(t.predict(np.empty((0,1))).shape,(0,))
        self.assertEqual(t.predict_proba(np.empty((0,1))).shape,(0,2))

    def test_max_features_options(self):
        for x in [1,.5,'sqrt','log2',None]:
            with self.subTest(max_features=x):
                model=DecisionTreeClassifier(max_features=x)
                model.fit(np.column_stack((self.X,self.X,self.X)),self.y)
                self.assertEqual(len(model.predict(np.column_stack((self.X,self.X,self.X)))),4)

    def test_reproducible_feature_sampling(self):
        X=np.column_stack((self.X,self.X,self.X))
        t1=DecisionTreeClassifier(max_features=1,random_state=8).fit(X,self.y)
        t2=DecisionTreeClassifier(max_features=1,random_state=8).fit(X,self.y)
        self.assertEqual(t1.root_.feature,t2.root_.feature)

    def test_buffer_growing_columns(self):
        t=DecisionTreeClassifier(buffer_size=4)
        t.partial_fit(self.X[:2],self.y[:2])
        t.partial_fit(np.column_stack((self.X[2:],np.ones(2))),self.y[2:])
        self.assertEqual(t.n_features_in_,2)
        self.assertEqual(t.buffered_samples_,0)

    def test_nan_labels_rejected(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit(self.X,[0,0,1,np.nan])

    def test_tree_summary(self):
        t=DecisionTreeClassifier().fit(self.X,self.y)
        self.assertEqual(t.summary(),{'nodes':3,'leaves':2,'depth':1})

    def test_two_feature_rule(self):
        X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
        y=np.array([0,1,1,1])
        self.assertTrue(np.array_equal(DecisionTreeClassifier(max_depth=2).fit(X,y).predict(X),y))
