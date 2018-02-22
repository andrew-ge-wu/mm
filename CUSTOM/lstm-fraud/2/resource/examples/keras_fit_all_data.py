"""
Use a pipeline to feature eng. Fit a keras model using keras.fit.
"""
import pandas as pd
from keras import layers, models
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from pipeline import Pipeline, Callback
from pipeline.pipes_df import Drop, ToXy, OneToOne

data = pd.read_csv('../data/creditcard.csv')[:10000]  # 10000 to be faster

# say this is the split
train = data.iloc[9000:].copy()
test = data.iloc[:1000].copy()


def get_model():
    inputs = predictions = layers.Input(shape=(28,), name='input')  # 28 because we drop 2

    predictions = layers.Dense(128, activation='relu', name='dense_%d' % 1)(predictions)
    predictions = layers.Dropout(0.1)(predictions)
    predictions = layers.concatenate([inputs, predictions])

    predictions = layers.Dense(2, activation='softmax', name='output')(predictions)

    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile('adam', 'binary_crossentropy')

    return model


pipeline = Pipeline([
    Drop(['V2', 'V3']),
    OneToOne('V4', StandardScaler()),  # standarize V4
    Callback(lambda x: print(x['V4'].mean(), x['V4'].std())),  #  we can print in the middle of the pipeline for debugging
    ToXy('Class'),  # split the data in x,y (during fit)
])

pipeline.fit(train.copy())

keras_model = get_model()

x, y = pipeline.transform(train)

keras_model.fit(x.values, to_categorical(y, 2))

x_test, y_test = pipeline.transform(test)

y_pred = keras_model.predict(x_test.values).argmax(axis=-1)

print(confusion_matrix(y_test, y_pred))
