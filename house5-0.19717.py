from input_data2 import read_data

X_train,y_train=read_data("./data/train.csv")
X_test,y_test=read_data("./data/test.csv",True)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Flatten,Reshape
from keras.layers.convolutional import Conv1D
from keras import backend as K
import pandas as pd
import os
from keras.layers.merge import Concatenate
from sklearn.neighbors import KNeighborsRegressor


knn=KNeighborsRegressor()


knn.fit(X_train,y_train)

X_test,y_test=read_data("./data/test.csv",False)

y_pred=knn.predict(X_test)
print(type(y_pred))
print(y_pred)

result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)