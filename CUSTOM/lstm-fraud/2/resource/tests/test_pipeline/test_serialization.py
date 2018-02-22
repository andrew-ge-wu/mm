import shutil
from unittest import TestCase

import numpy as np

from pipeline import Pipeline, Pipe
from pipeline import save_untrained, load_untrained, save_trained, load_trained
from pipeline.pipes_np import ToXy
from tests.test_pipeline.test_kerasclassifier import SimpleKerasClassifier


class TestIO(TestCase):
    directory = 'tmp'

    def setUp(self):
        try:
            shutil.rmtree(self.directory)
        except OSError:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.directory)
        except OSError:
            pass

    def test_untrained_basic(self):
        pipeline_original = Pipeline([Pipe(), Pipe()])

        save_untrained(pipeline_original, 'tmp')
        pipeline = load_untrained('tmp')

        self.assertEqual(len(pipeline_original.pipes), len(pipeline.pipes))

    def test_untrained_keras(self):
        classifier = SimpleKerasClassifier()

        pipeline_original = Pipeline([Pipe(), ToXy(), classifier])

        pipeline_original.pipes[-1].fit_architecture(np.array([[1], [2]]), np.array([[0], [1]]))

        save_untrained(pipeline_original, 'tmp')

        pipeline = load_untrained('tmp')

        self.assertEqual(len(pipeline.pipes[-1].model.layers), len(pipeline_original.pipes[-1].model.layers))

    def test_trained_keras(self):
        classifier = SimpleKerasClassifier()

        pipeline_original = Pipeline([Pipe(), ToXy(), classifier])

        data = np.array([[1, 0], [2, 1], [3, 1], [4, 0]])
        pipeline_original.fit(data)

        save_trained(pipeline_original, 'tmp')

        pipeline = load_trained('tmp')

        self.assertEqual(len(pipeline.pipes[-1].model.layers), len(pipeline_original.pipes[-1].model.layers))
