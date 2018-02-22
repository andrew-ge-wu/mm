from .keras_classifier import KerasClassifier


class LogisticRegressionClassifier(KerasClassifier):
    """
    A Logistic Regression model classifier with a pipeline API.

    - pass `n_classes` to pre-select the number of classes to use (otherwise they are fitted from the first batch).

    If `'loss'` is not provided in `compile_kwargs`, it fits the number of categories from the training set
    and uses `'binary_crossentropy'` or `'categorical_crossentropy'` depending on the number of categories.
    """

    @staticmethod
    def build_model(input_shape, output_dim):
        """
        Builds the model from the shapes. Must return a Keras model
        """
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add(Dense(output_dim, input_shape=input_shape, activation='softmax'))
        return model


from .tensor_logistic_regression_classifier import TensorialLogisticRegressionClassifier
