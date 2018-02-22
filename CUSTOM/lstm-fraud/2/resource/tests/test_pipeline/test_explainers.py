import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline import Pipe, SK
from pipeline.explainers import LocalExplainer, ExplanatoryPipeline, LimeExplainer
from pipeline.pipes_np import ToXy
from .auxiliar import generate_logistic


class TestLimeExplainer(unittest.TestCase):

    def test_basic(self):
        """
        Check that when the we fit a 2-variables logistic regression with its underlying process, the two
        variables have the same explanatory power.
        """
        def to_x_y(x):
            return x[:, :-1], x[:, -1]

        p = ExplanatoryPipeline([
            Pipe(),
            LimeExplainer(to_x_y, feature_names=['a', 'b']),
            Pipe(),
            ToXy(),
            SK(LogisticRegression())
        ])

        data = generate_logistic([1, 1], 100000)
        p.fit(data)

        result = p.explain(np.array([[1, 1]]), num_samples=5000).as_list()
        # both are similar
        self.assertTrue(abs(result[0][1] - result[1][1]) < 0.1)

        # See https://github.com/marcotcr/lime/issues/113 for a detailed discussion on the remaining of the test


class TestLocalExplainer(unittest.TestCase):
    def test_basic(self):
        """
        Check that when the we fit a 2-variables logistic regression with its underlying process, the two
        variables have the same explanatory power and their value is close to 0.1: the local derivative of the Logistic function at [1,1]
        """
        try:
            import numdifftools
        except ImportError:
            raise unittest.SkipTest('numdifftools not installed.')

        def to_x_y(x):
            return x[:, :-1], x[:, -1]

        p = LocalExplainer(to_x_y, columns=[0, 1], names=['a', 'b'])
        classifier = LogisticRegression(C=1e100)

        data = generate_logistic([1, 1], 1000)
        classifier.fit(*to_x_y(data))
        p.fit(data)

        x = np.array([1, 1], dtype=np.float64)

        result = p.explain(x, classifier.predict_proba)
        self.assertAlmostEqual(result['a'], result['b'], delta=0.1)
        self.assertAlmostEqual(result['a'], 0.1, delta=0.1)
