from unittest import TestCase
import shutil
import os
import numpy as np
import pandas as pd

from iterators.csv import CSVIterator


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
        path = self.directory + '/tmp.csv'

        df = pd.DataFrame({'a': np.arange(0, 100), 'b': np.random.choice(['a', 'b'], 100)})
        df.to_csv(path)

        for i, chunk in enumerate(CSVIterator(path, 10)):
            self.assertEqual(len(chunk), 10)
            self.assertEqual(chunk['a'].iloc[0], 10*i)
