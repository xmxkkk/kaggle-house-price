from input_data5 import read_data
import numpy as np

X_train,y_train=read_data("./data/train.csv")
X_test,y_test=read_data("./data/test.csv",True)

# X_train=np.append(X_train,X_test,axis=0)
# y_train=np.append(y_train,y_test,axis=0)

print(X_train.shape,y_train.shape)


from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Flatten,Reshape
from keras.layers.convolutional import Conv1D
from keras import backend as K
import pandas as pd
import os
from keras.layers.merge import Concatenate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# 100=0.20267
# 20 =0.18192
# 5  =0.17847
# 4  =0.17963
# 3  =0.17830
# 2  =0.18685
# 1  =0.20278
model=DecisionTreeRegressor()

model.fit(X_train,y_train)

X_test,y_test=read_data("./data/test.csv",True)

y_pred=model.predict(X_test)
print(type(y_pred))
print(y_pred)
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)