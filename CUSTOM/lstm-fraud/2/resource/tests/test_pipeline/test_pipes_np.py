from unittest import TestCase

import numpy as np

from pipeline import pipes_np, F


class TestF(TestCase):

    def test_basic(self):

        def square_first(x):
            x[:, 0] = x[:, 0] ** 2
            return x

        ds = np.array([[1, 2], [3, 4]])
        F(square_first).fit_transform(ds)
        np.testing.assert_equal(ds, [[1, 2], [9, 4]])


class TestDrop(TestCase):

    def test_basic(self):
        ds = np.array([[1, 2], [3, 4]])
        ds1 = pipes_np.Drop([0]).fit_transform(ds)
        np.testing.assert_equal(ds1, [[2], [4]])


class TestZscore(TestCase):
    def test_basic(self):
        ds = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7]
        ], dtype=np.float64)

        p = pipes_np.Zscore([1, 2])
        p.fit(ds)

        np.testing.assert_almost_equal(p.mean, np.array([np.mean([2, 4, 6]), np.mean([3, 5, 7])]))
        np.testing.assert_almost_equal(p.std, np.array([np.std([2, 4, 6]), np.std([3, 5, 7])]))

        bla = p.transform(ds)

        z1 = 2/np.std([2, 4, 6])
        np.testing.assert_almost_equal(bla, np.array([
            [1, -z1, -z1],
            [3, 0, 0],
            [5, z1, z1]
        ], dtype=np.float64))


class TestToXy(TestCase):
    def test_basic(self):
        ds = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7]
        ], dtype=np.float64)

        pipe = pipes_np.ToXy()

        np.testing.assert_equal(pipe.transform(ds), ds)
        x, y = pipe.transform(ds, supervised=True)
        np.testing.assert_equal(x, ds[:, :-1])
        np.testing.assert_equal(y, ds[:, -1])
