from input_data3 import read_data

X_train,y_train=read_data("./data/train.csv")
X_test,y_test=read_data("./data/test.csv",True)

split_val=1000

X_valid,y_valid=X_train[split_val:],y_train[split_val:]
X_train,y_train=X_train[:split_val],y_train[:split_val]


from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Flatten,Reshape
from keras.layers.convolutional import Conv1D
from keras import backend as K
import pandas as pd
import os
from keras.layers.merge import Concatenate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_absolute_error


model=Sequential()
model.add()


y_pred=model.predict(X_test)
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)