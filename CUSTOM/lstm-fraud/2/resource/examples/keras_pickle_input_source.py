"""
Transverses a pickle file with one data row per line, use the pipeline to transform it, and use fit_generator
to train the Keras model.
"""
import os
import pickle

import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from iterators import PickleIterator
from pipeline import Pipeline, F

FILE_NAME = 'test.pkl'
BATCH_SIZE = 16


def _export(file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'wb') as f:
            pickler = pickle.Pickler(f)
            for i in range(100*BATCH_SIZE):
                other_data = [np.random.random(10), np.random.random(20), np.random.random(30), np.random.choice([0, 1])]
                pickler.dump(other_data)


def get_data(file_name):
    _export(file_name)

    with open(file_name, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        result = []
        while True:
            try:
                line = unpickler.load()
            except EOFError:
                break
            result.append(line)
    return result


def get_model():
    from keras import layers, models
    inputs = predictions = layers.Input(shape=(10,), name='input10')
    inputs1 = layers.Input(shape=(20,), name='input20')

    predictions = layers.Dense(128, activation='relu', name='dense_%d' % 1)(predictions)
    predictions = layers.Dropout(0.1)(predictions)
    predictions = layers.concatenate([inputs1, predictions])

    predictions = layers.Dense(2, activation='softmax', name='output')(predictions)

    model = models.Model(inputs=[inputs, inputs1], outputs=predictions)
    model.compile('adam', 'binary_crossentropy')

    return model


class ToMultipleInputs(F):
    def __init__(self):
        super().__init__(
            # we are dropping the third time-series here
            lambda x:  {'input10': np.array([x_i[0] for x_i in x]),
                        'input20': np.array([x_i[1] for x_i in x])},
            lambda x: ({'input10': np.array([x_i[0] for x_i in x]),
                        'input20': np.array([x_i[1] for x_i in x])},
                       np.array([x_i[-1] for x_i in x])))


class ToCategorical(F):
    def __init__(self):
        # assumes binary classification
        super().__init__(
            lambda x: x,
            lambda x: (x[0], to_categorical(x[1], 2)))


pipeline = Pipeline([
    ToMultipleInputs(),  # split the data in x,y (during fit)
    ToCategorical(),  # transform y to categorical matrix
])


data = get_data(FILE_NAME)


pipeline.fit(data)  # all data is loaded to memory... Still figuring how to solve this correctly.

iterator = PickleIterator(FILE_NAME, batch_size=BATCH_SIZE, is_infinite=True)
# make iteration passes through all the transformers (with x,y splitting)
iterator = pipeline.iterator(supervised=True)(iterator)

keras_model = get_model()

# use fit_generator over the file
keras_model.fit_generator(iterator, steps_per_epoch=90, epochs=2)

# we do not want y_test to be a class-encoded matrix.
# Thus we remove the last element of the pipe that prepares the y-data for Keras
test_pipeline = Pipeline(pipeline.pipes[:-1])
x_test, y_test = test_pipeline.transform(data[10*BATCH_SIZE:].copy())

y_pred = keras_model.predict(x_test).argmax(axis=-1)

print(confusion_matrix(y_test, y_pred))

# clean in the end
if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)
