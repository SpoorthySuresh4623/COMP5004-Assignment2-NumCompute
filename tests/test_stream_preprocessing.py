import unittest
import numpy as np
from numcompute_stream.preprocessing import StandardScaler,Imputer,OneHotEncoder


class PreprocessingTests(unittest.TestCase):
    def setUp(self):self.X=np.array([[1.,2.],[3.,4.],[5.,6.]])
    def test_scaler_running_equals_batch(self):
        a=StandardScaler().fit(self.X)
        b=StandardScaler().partial_fit(self.X[:1]).partial_fit(self.X[1:])
        np.testing.assert_allclose(a.mean_,b.mean_)
        np.testing.assert_allclose(a.var_,b.var_)
    def test_scaler_standardizes(self):
        res=StandardScaler().fit_transform(self.X)
        np.testing.assert_allclose(res.mean(axis=0),[0,0],atol=1e-12)
        np.testing.assert_allclose(res.var(axis=0),[1,1],atol=1e-12)
    def test_scaler_constant(self):
        s=StandardScaler().fit(np.ones((4,2)))
        self.assertTrue(np.all(s.transform(np.ones((1,2)))==0))
    def test_scaler_nan_ignored(self):
        s=StandardScaler().fit([[1,np.nan],[3,4]])
        np.testing.assert_allclose(s.mean_,[2,4])
        self.assertTrue(np.isnan(s.transform([[np.nan,3]])[0,0]))
    def test_scaler_unfitted(self):
        with self.assertRaises(RuntimeError):StandardScaler().transform(self.X)
    def test_scaler_feature_change(self):
        s=StandardScaler().fit(self.X)
        with self.assertRaises(ValueError):s.partial_fit([[1]])
    def test_imputer_running(self):
        s=Imputer().partial_fit([[1,np.nan]]).partial_fit([[3,8]])
        np.testing.assert_allclose(s.statistics_,[2,8])
    def test_imputer_all_missing(self):
        s=Imputer().fit([[np.nan,1],[np.nan,3]])
        np.testing.assert_allclose(s.transform([[np.nan,np.nan]]),[[0,2]])
    def test_imputer_no_mutation(self):
        X=np.array([[np.nan,1]])
        y=Imputer().fit([[2,1]]).transform(X)
        self.assertTrue(np.isnan(X[0,0]));self.assertEqual(y[0,0],2)
    def test_encoder_initial_width(self):
        e=OneHotEncoder().fit([['red','S'],['blue','M']])
        self.assertEqual(e.n_output_features_,4)
        self.assertEqual(e.transform([['blue','S']]).sum(),2)
    def test_encoder_new_categories_append(self):
        e=OneHotEncoder().fit([['red','S']])
        old=e.transform([['red','S']])
        e.partial_fit([['blue','M']])
        new=e.transform([['red','S']])
        np.testing.assert_array_equal(new[:,:2],old)
        self.assertEqual(new.shape[1],4)
    def test_encoder_unseen_is_zero(self):
        e=OneHotEncoder().fit([['red']])
        self.assertEqual(e.transform([['blue']]).sum(),0)
    def test_encoder_missing_category(self):
        e=OneHotEncoder().fit([[None],[np.nan]])
        self.assertEqual(e.n_output_features_,1)
    def test_encoder_reset_fit(self):
        e=OneHotEncoder().fit([['a'],['b']]).fit([['z']])
        self.assertEqual(e.n_output_features_,1)
    def test_encoder_incorrect_shape(self):
        e=OneHotEncoder().fit([['a','b']])
        with self.assertRaises(ValueError):e.transform([['a']])
    def test_encoder_unfitted(self):
        with self.assertRaises(RuntimeError):OneHotEncoder().transform([['a']])

    def test_encoder_numeric_and_string_distinct(self):
        enc = OneHotEncoder().fit([[1], ['1']])
        self.assertEqual(enc.n_output_features_, 2)
        self.assertEqual(enc.transform([[1]])[0].sum(), 1)
        self.assertEqual(enc.transform([['1']])[0].sum(), 1)

    def test_encoder_int_float_equivalent(self):
        enc = OneHotEncoder().fit([[1], [1.0]])
        self.assertEqual(enc.n_output_features_, 1)

    def test_encoder_infinite_category_rejected(self):
        with self.assertRaises(ValueError):
            OneHotEncoder().fit([[np.inf]])
