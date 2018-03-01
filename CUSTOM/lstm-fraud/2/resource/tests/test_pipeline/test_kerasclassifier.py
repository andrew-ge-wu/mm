from unittest import TestCase

import numpy as np

from pipeline import KerasClassifier


class SimpleKerasClassifier(KerasClassifier):

    def __init__(self, n_classes=None):
        super().__init__(compile_kwargs={'optimizer': 'adam', 'metrics': ['accuracy']}, n_classes=n_classes, fit_kwargs={'verbose': 0})

    @staticmethod
    def build_model(input_shape, output_shape):
        # this is the architecture. Feel free to change to your favourite architecture
        from keras import layers, models
        inputs = predictions = layers.Input(shape=input_shape, name='input')

        predictions = layers.Dense(128, activation='relu', name='dense_%d' % 1)(predictions)
        predictions = layers.Dropout(0.1)(predictions)
        predictions = layers.concatenate([inputs, predictions])

        predictions = layers.Dense(output_shape, activation='softmax', name='output')(predictions)

        model = models.Model(inputs=inputs, outputs=predictions)

        return model


class TestKeras(TestCase):
    def test_fit(self):
        classifier = SimpleKerasClassifier(2)

        data = np.array([[1, 0], [2, 1], [3, 1], [4, 0]])
        classifier.fit((data[:, [0]], data[:, 1]))

    def test_fit_partial(self):
        classifier = SimpleKerasClassifier(2)
        classifier.fit_kwargs = {}

        data = np.array([[1, 0], [2, 1], [3, 1], [4, 0]])
        classifier.fit_partial((data[:, [0]], data[:, 1]))

    def test_fit_architecture(self):
        classifier = SimpleKerasClassifier(2)

        data = np.array([[1, 0], [2, 1], [3, 1], [4, 0]])
        classifier.fit_architecture(data[:, [0]], data[:, 1])

    def test_fit_classes(self):
        # data with 2 classes
        classifier = SimpleKerasClassifier(None)
        data = np.array([[1, 0], [2, 1], [3, 1], [4, 0]])
        classifier.fit_architecture(data[:, [0]], data[:, 1])

        self.assertEqual(classifier.states['n_classes'], 2)

        # data with 3 classes
        classifier = SimpleKerasClassifier(None)
        data = np.array([[1, 0], [2, 1], [3, 1], [4, 2]])
        classifier.fit_architecture(data[:, [0]], data[:, 1])

        self.assertEqual(classifier.states['n_classes'], 3)

    def test_fit_predict_binary(self):
        classifier = SimpleKerasClassifier(None)
        data = np.array([[1, 0], [1, 0], [3, 1], [3, 1]])
        classifier.fit((data[:, [0]], data[:, 1]))

    def test_fit_predict_multi(self):
        classifier = SimpleKerasClassifier(None)
        data = np.array([[1, 0], [1, 0], [3, 1], [3, 1], [10, 2], [10, 2]])
        classifier.fit((data[:, [0]], data[:, 1]))
