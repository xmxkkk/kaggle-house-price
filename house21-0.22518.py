from input_data7 import read_data

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

model=Sequential()

# model.add(BatchNormalization(input_shape=(X_train.shape[1],)))

model.add(Dense(389*4,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dense(389*4,activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(389,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mae',metrics=['mae'])

# for i in range(15):
model.fit(X_train,y_train,batch_size=X_train.shape[0],epochs=80,verbose=1)
model.save_weights("./x.w")

y_pred=model.predict(X_test,batch_size=X_test.shape[1])
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)