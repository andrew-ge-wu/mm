import unittest

import numpy as np
from sklearn.linear_model import SGDClassifier

from iterators import DummyIterator
from pipeline import Pipeline, F, SK
from pipeline.pipes_np import ToXy


def square_first_column(dataset):
    dataset[:, 0] = dataset[:, 0] ** 2
    return dataset


class BatchTestCase(unittest.TestCase):
    """Test case for the batch training."""

    def test_dummy_iterator_with_SGDClassifier(self):
        """Test dummy_iterator with SGDClassifier."""

        classifier = SGDClassifier()
        classifier.classes_ = np.array([0, 1])
        iterator = DummyIterator(sample_size=10, batch_size=3)

        pipeline = Pipeline([
            F(square_first_column),
            ToXy(),
            SK(classifier)
        ])

        # train the model using the data
        for chunk in iterator:
            pipeline.fit_partial(chunk)

        x = np.array([[0.0, 0.0]])

        y_pred_train = pipeline.predict(x)

        self.assertEqual(y_pred_train, [0])
