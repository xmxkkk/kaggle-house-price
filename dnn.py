import pandas as pd
import numpy as np

x_train=pd.read_csv('./data/x_train.csv')
y_train=pd.read_csv('./data/y_train.csv')

x_test=pd.read_csv('./data/x_test.csv')

from keras.models import Sequential
from keras.layers import Dense,Dropout

print(x_train.shape)
model=Sequential()
model.add(Dense(221*4,input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(221,activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(optimizer='adam',loss='mae')

model.fit(x_train.values,y_train.values,batch_size=x_train.shape[0],epochs=500)
y_test_pred=model.predict(x_test.values)

y_test_pred=np.expm1(y_test_pred.reshape(-1))


test=pd.read_csv('./data/test.csv')

pd.DataFrame({"Id":test['Id'].values,"SalePrice":y_test_pred}).to_csv("result_dnn.csv",index=False)



