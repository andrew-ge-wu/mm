import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pipeline.classifiers import LogisticRegressionClassifier
from pipeline.classifiers import TensorialLogisticRegressionClassifier
from tests.auxiliar import generate_logistic


class TestLogisticRegression(unittest.TestCase):

    @staticmethod
    def model():
        # the optimizers sgd and rmsprop seem to find solutions that are closer to that found by sklearn than the
        # optimizers adam and adagrad.
        return LogisticRegressionClassifier(compile_kwargs={'optimizer': 'sgd',
                                                            'loss': 'categorical_crossentropy'},
                                            fit_kwargs={"epochs": 200, "batch_size": 100, 'verbose': 0})

    def test_predictions(self):
        """
        Tests that the predictions from Keras match those from Logistic Regression from SKLearn.
        See https://stackoverflow.com/questions/44930153/keras-and-sklearn-logreg-returning-different-results
        for a discussion of why the two approaches to logistic regression can be expected to yield different results.
        """
        data = generate_logistic(np.array([1., 1.]), 5000, random_state=1)
        X, X_test, y, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=1)

        classifier = self.model()
        classifier.fit((X, y))
        predictions_keras = classifier.predict_proba(X_test)

        classifier_sklearn = LogisticRegression(random_state=0, C=1e20)
        classifier_sklearn.fit(X, y)

        predictions_sklearn = classifier_sklearn.predict_proba(X_test)

        n_obs = np.shape(predictions_keras)[0]

        mean_sq_diff = np.sqrt(np.sum(np.square(predictions_sklearn[:, 0]-predictions_keras[:, 0]))/n_obs)

        self.assertLess(mean_sq_diff, 1e-3)

    def test_weights(self):
        """
        Tests that the weights of the Logistic Regression in Keras make sense.
        """
        def to_x_y(data):
            return data[:, :-1], data[:, -1]

        coefficients = np.array([1, 1])
        data = generate_logistic(coefficients, 50000, random_state=42)

        classifier_sklearn = LogisticRegression(random_state=0, C=1e14)
        classifier_sklearn.fit(*to_x_y(data))

        # sklearn logistic: coefficients match the ones from generated data
        np.testing.assert_allclose(classifier_sklearn.coef_[0], coefficients, rtol=0.05)  # 5% close

        classifier = self.model()
        classifier.fit(to_x_y(data))

        weights = classifier.model.layers[0].get_weights()

        coef_weigts = weights[0]

        #np.testing.assert_allclose(coef_weigts[:, 1] - coef_weigts[:, 0], coefficients, rtol=0.05)  # 5% close
        np.testing.assert_allclose(coef_weigts[:, 1] - coef_weigts[:, 0], classifier_sklearn.coef_[0], rtol=0.05)  # 5% close

        np.testing.assert_allclose(weights[1][1]-weights[1][0], classifier_sklearn.intercept_[0], rtol=0.05, atol=0.005)  # 5% close


class TestTensorialLogisticRegressionClassifier(unittest.TestCase):

    def test_discrimination_keras(self):
        nsamples_perclass = 200
        ndim = 25
        nfeats_order = np.int(np.sqrt(ndim))
        cov1 = np.identity(ndim)
        cov2 = 5 * np.identity(ndim)
        mu1 = np.repeat(0, repeats=ndim)
        mu2 = np.repeat(5, repeats=ndim)
        X_train = np.concatenate((
            np.random.multivariate_normal(mean=mu1, cov=cov1, size=nsamples_perclass).
                reshape((nsamples_perclass, nfeats_order, nfeats_order)),
            np.random.multivariate_normal(mean=mu2, cov=cov2, size=nsamples_perclass).
                reshape((nsamples_perclass, nfeats_order, nfeats_order))))

        y_train = np.concatenate((np.repeat(0, repeats=nsamples_perclass),
                                  np.repeat(1, repeats=nsamples_perclass)))


        X_test = np.concatenate((
            np.random.multivariate_normal(mean=mu1, cov=cov1, size=nsamples_perclass).
                reshape((nsamples_perclass, nfeats_order, nfeats_order)),
            np.random.multivariate_normal(mean=mu2, cov=cov2, size=nsamples_perclass).
                reshape((nsamples_perclass, nfeats_order, nfeats_order))))

        y_test = np.concatenate((np.repeat(0, repeats=nsamples_perclass),
                                  np.repeat(1, repeats=nsamples_perclass)))

        permutation = np.random.permutation(range(0, 2 * nsamples_perclass))

        keras_model = TensorialLogisticRegressionClassifier()

        keras_model.fit((X_train[permutation,...], y_train[permutation]))

        expectedlabels = keras_model.predict_proba(X_test)

        binary_preds = expectedlabels[:, 1] > 0.5

        self.assertGreaterEqual(np.mean(binary_preds == y_test), 0.5)
