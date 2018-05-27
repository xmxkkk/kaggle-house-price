from input_data7 import read_data

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


# data3=23628.5231884
# data7=22182.8652174
model=KNeighborsRegressor(n_neighbors=3)#data3  22424.4862319 data5 fan=27056.1391304
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(mean_absolute_error(y_valid,y_pred_valid))



'''
model=TheilSenRegressor()#0.602697732762
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=LogisticRegression()#0.602697732762
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=ARDRegression()#0.844699171819
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=1)#0.677865825512
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=2)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=4)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=6)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model=KNeighborsRegressor(n_neighbors=10)
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))

model = ElasticNet()
model.fit(X_train,y_train)
y_pred_valid=model.predict(X_valid)
print(r2_score(y_valid,y_pred_valid))
'''

y_pred=model.predict(X_test)
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)
