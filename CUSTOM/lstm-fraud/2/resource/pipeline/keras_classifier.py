import numpy as np
from .pipe import Pipe


class KerasClassifier(Pipe):
    """
    A Keras model classifier with a pipeline API.

    - overwrite `build_model` to change the model architecture
    - pass `fit_kwargs` or `compile_kwargs` to initialization to pass them to Keras.
    - pass `n_classes` to pre-select the number of classes to use (otherwise they are fitted from the first batch).

    If `'loss'` is not provided in `compile_kwargs`, it fits the number of categories from the training set
    and uses `'binary_crossentropy'` or `'categorical_crossentropy'` depending on the number of categories.
    """
    def __init__(self, fit_kwargs=None, compile_kwargs=None, n_classes=None):
        if fit_kwargs is None:
            fit_kwargs = {}
        if compile_kwargs is None:
            compile_kwargs = {}
        self._compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

        # whether to fit the classes or not
        self._fit_n_classes = n_classes is None

        super().__init__({'n_classes': n_classes, 'model': None})

    @property
    def model(self):
        return self.states['model']

    @model.setter
    def model(self, model):
        self.states['model'] = model

    @staticmethod
    def build_model(input_shape, output_shape):
        """
        Builds the model from the shapes. Must return a Keras model
        """
        raise NotImplementedError

    def _fit_classes(self, y):
        if self._fit_n_classes:
            if len(y.shape) == 2 and y.shape[1] > 1:
                classes = np.arange(y.shape[1])
            elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
                classes = np.unique(y)
            else:
                raise ValueError('Invalid shape for y: ' + str(y.shape))
            self.states['n_classes'] = len(classes)

    def fit_architecture(self, x, y):
        self._fit_classes(y)

        # must be called after `_fit_classes` because it requires `self.states['n_classes']`
        self.model = self.build_model(x.shape[1:], self.states['n_classes'])

    def compile_model(self):
        if 'loss' not in self._compile_kwargs:
            if self.states['n_classes'] == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
            self.model.compile(loss=loss, **self._compile_kwargs)
        else:
            self.model.compile(**self._compile_kwargs)

    def fit(self, dataset):
        """
        Fit (train on batch) the data.
        """
        x, y = dataset
        self.fit_architecture(x, y)
        self.compile_model()

        from keras.utils import to_categorical
        return self.model.fit(x, to_categorical(y, self.states['n_classes']), **self.fit_kwargs)

    def fit_partial(self, dataset):
        """
        Partial fit (train on batch) the data.

        On the first batch, the number of classes is fitted from y and the model is compiled.
        On every batch, the model is trained on batch.
        """
        x, y = dataset
        if self.model is None:
            # first time, we build and compile the model
            self.fit_architecture(x, y)
            self.compile_model()

        from keras.utils import to_categorical
        return self.model.train_on_batch(x, to_categorical(y, self.states['n_classes']), **self.fit_kwargs)

    def predict_proba(self, x):
        probs = self.model.predict(x)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, x):
        """
        Class predictions is the maximal probability for multi-class or > 0.5 for binary classes
        """
        proba = self.model.predict(x)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def __getstate__(self):
        states = dict((x, self.states[x]) for x in self.states if x != 'model')
        state = dict((x, self.__dict__[x]) for x in self.__dict__ if x != 'states')
        state['states'] = states
        return state

    def __setstate__(self, state):
        self.__dict__ = state
