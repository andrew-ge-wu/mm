from unittest import TestCase
import numpy as np

from iterators import TrainTestIterator


class TestTrainTest(TestCase):
    samples = np.random.random(size=100)

    def iterator(self):
        for x in self.samples:
            yield x

    def test_splits(self):
        gen = TrainTestIterator(lambda: self.iterator(), seed=1)

        training = []
        for x in gen:
            training.append(x[0])

        test = gen.test_samples()

        self.assertEqual(set(test).intersection(set(training)), set())

    def test_fixed_seed_gives_same_results(self):
        gen = TrainTestIterator(lambda: self.iterator(), seed=1)

        training1 = []
        for x in gen:
            training1.append(x[0])

        test1 = gen.test_samples()

        gen = TrainTestIterator(lambda: self.iterator(), seed=1)
        training2 = []
        for x in gen:
            training2.append(x[0])
        test2 = gen.test_samples()

        self.assertEqual(test1, test2)
        self.assertEqual(training1, training2)

    def test_reset(self):
        gen = TrainTestIterator(lambda: self.iterator(), seed=1)
        training1 = []
        for x in gen:
            training1.append(x[0])

        gen.reset()
        training2 = []
        for x in gen:
            training2.append(x[0])

        self.assertEqual(training1, training2)
