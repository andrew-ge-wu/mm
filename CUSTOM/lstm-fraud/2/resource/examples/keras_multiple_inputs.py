"""
Use a pipeline to feature eng. Fit a keras model using keras.fit_generator with multiple inputs
"""
import pandas as pd
from keras import layers, models
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from iterators import CSVIterator
from pipeline import Pipeline, Callback, F
from pipeline.pipes_df import Drop, ToXy, OneToOne

BATCH_SIZE = 128

data = pd.read_csv('../data/creditcard.csv')[:100*BATCH_SIZE]


def get_model():
    inputs = predictions = layers.Input(shape=(28,), name='input')  # 28 because we drop 2
    inputs1 = layers.Input(shape=(28,), name='input1')  # 28 because we drop 2

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
            lambda x: {'input': x.values, 'input1': x.values},
            lambda x: ({'input': x[0].values, 'input1': x[0].values}, x[1]))


class ToCategorical(F):
    def __init__(self):
        # assumes binary classification
        super().__init__(
            lambda x: x,
            lambda x: (x[0], to_categorical(x[1], 2)))


pipeline = Pipeline([
    Drop(['V2', 'V3']),
    OneToOne('V4', StandardScaler()),  # standarize V4
    Callback(lambda x: print(x['V4'].mean(), x['V4'].std())),  #  we can print in the middle of the pipeline for debugging
    ToXy('Class'),  # split the data in x,y (during fit)
    ToMultipleInputs(),  # transform x to a dictionary of inputs
    ToCategorical(),  # transform y to categorical matrix
])


pipeline.fit(data.copy())  # all data is loaded to memory... Still figuring how to solve this correctly.


iterator = CSVIterator('../data/creditcard.csv', batch_size=BATCH_SIZE, is_infinite=True)
# make iteration passes through all the transformers (with x,y splitting)
iterator = pipeline.iterator(supervised=True)(iterator)

keras_model = get_model()

keras_model.fit_generator(iterator, steps_per_epoch=90, epochs=2)

# we do not want y_test to be a class encoded matrix.
# Thus we remove the last element of the pipe that prepares the y-data for Keras
test_pipeline = Pipeline(pipeline.pipes[:-1])
x_test, y_test = test_pipeline.transform(data.iloc[10*BATCH_SIZE:].copy())

y_pred = keras_model.predict(x_test).argmax(axis=-1)

print(confusion_matrix(y_test, y_pred))
