from unittest import TestCase
import numpy as np

from pipeline import Pipeline, Pipe, F
from pipeline.pipes_np import ToXy


class TestPipeline(TestCase):

    def test_fit_transformers(self):
        """
        `Pipeline.fit_transformers` does not fit the classifier.
        """
        a1 = Pipe()
        a2 = Pipe()
        a2.fitted = False

        def fit(x):
            a2.fitted = True

        a2.fit = fit

        Pipeline([a1, a2]).fit_transformers([])

        self.assertFalse(a2.fitted)

    def test_fit_classifier(self):
        """
        `Pipeline.fit_classifier` does not fit the transformers.
        """
        a1 = Pipe()
        a1.fitted = False
        a2 = Pipe()

        def fit(x):
            a1.fitted = True

        a1.fit = fit
        Pipeline([a1, a2]).fit_classifier([])

        self.assertFalse(a1.fitted)

    def test_fit(self):
        a1 = Pipe()
        a1.fitted = False
        a2 = Pipe()
        a2.fitted = False

        def fit2(x): a2.fitted = True

        def fit1(x): a1.fitted = True

        a2.fit = fit2
        a1.fit = fit1

        Pipeline([a1, a2]).fit([])

        self.assertTrue(a2.fitted)
        self.assertTrue(a1.fitted)

    def test_transform(self):
        double = Pipe()
        double.transform = lambda x, supervised: [2*x_i for x_i in x]

        p = Pipeline([double, double])

        r = p.transform_transformers([1, 2, 3])
        self.assertEqual(r, [2, 4, 6])

        r = p.transform([1, 2, 3])
        self.assertEqual(r, [4, 8, 12])

    def test_transformers_iterator(self):
        double = Pipe()
        double.transform = lambda x, supervised: [2 * x_i for x_i in x]
        p = Pipeline([double, double])

        r = p.transformers_iterator()(iter([[1], [2], [3]]))

        import types
        self.assertEqual(type(r), types.GeneratorType)

        for i, x in enumerate(r):
            self.assertEqual(x, [2*(i+1)])

    def test_iterator(self):
        double = Pipe()
        double.transform = lambda x, supervised: [2 * x_i for x_i in x]
        p = Pipeline([double, double])

        r = p.iterator()(iter([[1], [2], [3]]))

        import types
        self.assertEqual(type(r), types.GeneratorType)

        for i, x in enumerate(r):
            self.assertEqual(x, [4 * (i + 1)])

    def test_iterator_supervised(self):
        x_y_data = [
            np.array([[0, 0]]),
            np.array([[1, 2]]),
            np.array([[2, 4]])]  # 3 x-y tuples in batches of 1

        x_data = [
            np.array([[0]]),
            np.array([[1]]),
            np.array([[2]])]  # 3 xs in batches of 1

        p = Pipeline([ToXy()])

        # the list is x, y
        r = p.iterator(supervised=True)(iter(x_y_data))

        import types
        self.assertEqual(type(r), types.GeneratorType)

        for i, x_y in enumerate(r):
            x, y = x_y
            self.assertEqual(x[0], np.array([i]))
            self.assertEqual(y[0], np.array([2*i]))

        # the list is now x
        r = p.iterator(supervised=False)(iter(x_data))

        self.assertEqual(type(r), types.GeneratorType)

        for i, x in enumerate(r):
            self.assertEqual(x, [i])
