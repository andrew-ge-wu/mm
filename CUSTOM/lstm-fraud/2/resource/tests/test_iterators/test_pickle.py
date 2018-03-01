from unittest import TestCase
import shutil
import os
import numpy as np
import pickle

from iterators import PickleIterator


class TestFile(TestCase):
    directory = 'tmp'

    def setUp(self):
        try:
            shutil.rmtree(self.directory)
        except OSError:
            pass
        os.makedirs(self.directory)

    def tearDown(self):
        try:
            shutil.rmtree(self.directory)
        except OSError:
            pass

    def test_basic(self):
        # create file
        path = self.directory + '/tmp.pkl'

        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            for i in range(100):
                other_data = [np.random.random(10), np.random.random(20), np.random.random(30)]
                pickler.dump(other_data)

        iterator = PickleIterator(path, 10)
        for i, chunk in enumerate(iterator):
            self.assertEqual(len(chunk), 10)
            self.assertEqual(len(chunk[0][0]), 10)
            self.assertEqual(len(chunk[0][1]), 20)
            self.assertEqual(len(chunk[0][2]), 30)
        self.assertTrue(iterator._descriptor.close)

    def test_stop_middle(self):
        # create file
        path = self.directory + '/tmp.pkl'

        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            for i in range(100):
                other_data = [np.random.random(10), np.random.random(20), np.random.random(30)]
                pickler.dump(other_data)

        iterator = PickleIterator(path, 10)
        for i, chunk in enumerate(iterator):
            break
        descriptor = iterator._descriptor
        del iterator
        self.assertTrue(descriptor.closed)

    def test_reset(self):
        path = self.directory + '/tmp.pkl'

        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            for i in range(100):
                other_data = i
                pickler.dump(other_data)

        iterator = PickleIterator(path, 1)
        x = iterator.__next__()
        self.assertEqual(x, 0)
        iterator.reset()
        x = iterator.__next__()
        self.assertEqual(x, 0)
