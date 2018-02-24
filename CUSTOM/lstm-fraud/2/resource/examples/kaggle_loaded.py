import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from pipeline import Pipeline, SK, Callback
from pipeline.pipes_df import Drop, ToXy, OneToOne

data = pd.read_csv('../data/creditcard.csv')[:10000]  # 10000 to be faster

# say this is the split
train = data.iloc[9000:]
test = data.iloc[:1000]

pipeline = Pipeline([
    Drop(['V2', 'V3']),
    OneToOne('V4', StandardScaler()),  # standarize V4
    Callback(lambda x: print(x['V4'].mean(), x['V4'].std())),  #  we can print in the middle of the pipeline (or e.g. contact a service)
    ToXy('Class'),  # split the data in x,y (during fit)
    SK(LogisticRegression(C=1e10))
])

pipeline.fit(train.copy())  # copy so train is not modified

### predict and performance

x_test, y_test = ToXy('Class').transform(test, supervised=True)

y_pred = pipeline.predict(x_test)

print(confusion_matrix(y_test, y_pred))
