from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from pipeline import Pipeline
from pipeline.pipes_df import ManyToOne, OneToOne, ApplyToAll, OneToMany, SelectToNumpy, Drop, Select, Callback, ToXy


class TestManyToOne(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [1, 2]})
        df1 = ManyToOne(['a', 'b'], FunctionTransformer(lambda x: x[:, 0] + x[:, 1]), 'c').fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [2, 4]}))


class TestOneToOne(TestCase):

    def test_same_column(self):
        df = pd.DataFrame({'a': [1, 2]})
        df1 = OneToOne('a', FunctionTransformer(lambda x: 2*x)).fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [2, 4]}))

    def test_other_column(self):
        df = pd.DataFrame({'a': [1, 2]})
        df1 = OneToOne('a', FunctionTransformer(lambda x: 2*x), 'a2').fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [1, 2], 'a2': [2, 4]}))

    def test_in_pipeline(self):
        pipe1 = OneToOne('a', FunctionTransformer(lambda x: 2 * x), 'a2')
        pipe2 = OneToOne('a2', FunctionTransformer(lambda x: 2 * x), 'a4')

        pipeline = Pipeline([pipe1, pipe2])

        df = pd.DataFrame({'a': [1, 2]})

        result = pipeline.fit_transform(df)

        assert_frame_equal(result, pd.DataFrame({'a': [1, 2], 'a2': [2, 4], 'a4': [4, 8]}))


class TestApplyToAll(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        df1 = ApplyToAll(FunctionTransformer(lambda x: 2*x)).fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [2, 4], 'b': [4, 6]}))


class TestOneToMany(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2]})
        df1 = OneToMany('a', OneHotEncoder(sparse=False)).fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [1, 2], 'a_0': [1.0, 0.0], 'a_1': [0.0, 1.0]}))


class TestSelectToNumpy(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2]})
        matrix = SelectToNumpy().fit_transform(df)
        np.testing.assert_equal(matrix, np.array([[1], [2]]))

    def test_one_column(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        matrix = SelectToNumpy(['a']).fit_transform(df)
        np.testing.assert_equal(matrix, np.array([[1], [2]]))


class TestSelect(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        df1 = Select(['a']).fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [1, 2]}))


class TestDrop(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        df1 = Drop(['b']).fit_transform(df)
        assert_frame_equal(df1, pd.DataFrame({'a': [1, 2]}))


class TestCallback(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})

        # a way to check that callback gets called
        global was_called
        was_called = 0

        def callback(x):
            global was_called
            was_called = True
        Callback(fit=callback).fit_transform(df)
        self.assertTrue(was_called)


class TestToXy(TestCase):

    def test_basic(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})

        pipe = ToXy('a')

        assert_frame_equal(pipe.transform(df), df)
        x, y = pipe.transform(df, supervised=True)
        assert_frame_equal(x, df[['b']])
        assert_series_equal(y, df['a'])
