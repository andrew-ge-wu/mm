"""
Use a pipeline to feature eng. Fit a keras model using model.fit_generator.
"""
import numpy as nu
import os
import pandas as pd
import urllib.request
from iterators import TeradataIterator, CSVIterator
from keras import backend as K
from keras import layers, models
from keras.utils import to_categorical
from pipeline import Pipeline, Callback, F
from pipeline.pipes_df import Drop, ToXy
from sklearn.metrics import confusion_matrix
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

batch_size = int(os.getenv('BATCH_SIZE', "128"))
nr_epoch = int(os.getenv('EPOCH', "2"))
epoch_steps = int(os.getenv('EPOCH_STEPS', "90"))

urllib.request.urlretrieve('https://s3-us-west-2.amazonaws.com/ukkhdeh3ptsk8kh3/creditcard_validation.csv',
                           'validation.csv')
validation_data = pd.read_csv('validation.csv')[:100 * batch_size]

metrics = ["mae", "acc"]


def get_model():
    inputs = predictions = layers.Input(shape=(28,), name='input')
    predictions = layers.Dense(128, activation='relu', name='dense_%d' % 1)(predictions)
    predictions = layers.Dropout(0.1)(predictions)
    predictions = layers.concatenate([inputs, predictions])
    predictions = layers.Dense(2, activation='softmax', name='output')(predictions)
    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile('adam', 'binary_crossentropy', metrics=metrics)
    return model


class ToKeras(F):
    def __init__(self):
        # assumes binary classification
        super().__init__(lambda x: x.values, lambda x: (x[0].values, to_categorical(x[1], 2)))


def get_pipeline(y_column):
    return Pipeline([
        Drop(['V2', 'V3']),
        # OneToOne('V4', StandardScaler()),  # standarize V4
        Callback(lambda x: print(x['V4'].mean(), x['V4'].std())),
        # we can print in the middle of the pipeline for debugging
        ToXy(y_column),  # split the data in x,y (during fit)
        ToKeras(),  # transform x to numpy, y to categorical matrix
    ])


training_iterator = TeradataIterator(user='dbc', passwd='Teradata123', host='10.89.16.159',
                                     query="SELECT TIMES, V1 , V2 , V3 , V4 , V5 , V6 , V7 , V8 , V9 , V10 , V11 , V12 , V13 , V14 , V15 , V16 , V17 , V18 , V19 , V20 , V21 , V22 , V23 , V24 , V25 , V26 , V27 , V28 , AMOUNT, REGEXP_REPLACE(CLASS_INDICATOR, '\"') CLASS_INDICATOR from AIFraud.credit_card",
                                     batch_size=batch_size)
training_iterator = get_pipeline('CLASS_INDICATOR').iterator(supervised=True)(training_iterator)

validation_iterator = CSVIterator('validation.csv', batch_size=batch_size, is_infinite=True)
validation_iterator = get_pipeline('Class').iterator(supervised=True)(validation_iterator)
# make iteration passes through all the transformers (with x,y splitting)
keras_model = get_model()


keras_model.fit_generator(training_iterator, steps_per_epoch=epoch_steps, epochs=int(nr_epoch))
# we do not want y_test to be a class encoded matrix.
# Thus we remove the last element of the pipe that prepares the data for Keras

data = keras_model.evaluate_generator(validation_iterator, steps=epoch_steps)
performance = {"loss": data[0], metrics[0]: data[1], metrics[1]: data[2]}
import json

with open('performance.json', 'w') as fp:
    json.dump(performance, fp)
print(performance)
print("Saving output")

builder = saved_model_builder.SavedModelBuilder("output")
signature = predict_signature_def(inputs={'data': keras_model.input},
                                  outputs={'scores': keras_model.output})
with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()

    # Save the Keras classifier in the h5 format.
    keras_model.save("output" + "/keras_model.h5")
