from input_data import Data

data=Data()
X_train,y_train=data.read_data("./data/train.csv",'train')
print(X_train.shape,y_train.shape)

X_test,y_test=data.read_data("./data/test.csv",'test')
print(X_test.shape,y_test.shape)

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

model.add(Dense(256,input_shape=(X_train.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(1,activation='relu'))

'''
model.add(BatchNormalization(input_shape=(X_train.shape[1],)))
model.add(Reshape((X_train.shape[1],1)))
model.add(Conv1D(32,kernel_size=3,strides=1,padding='same'))
model.add(Conv1D(32,kernel_size=2,strides=1,padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1))
'''

# for i in range(10):
#     model.add(Dropout(0.05))
#     model.add(Dense(79,activation='selu'))
#
# model.add(Dense(128))
# model.add(Dense(64))
#
# model.add(Dense(1))

# model.add(Dropout(0.5))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(1))

model.compile(optimizer='adam',loss='mae',metrics=['mae'])

if os.path.exists("./x.w"):
    model.load_weights("./x.w")

for i in range(30):
    model.fit(X_train,y_train,batch_size=X_train.shape[0],epochs=100,verbose=1)
    print(i)
    model.save_weights("./x.w")

y_pred=model.predict(X_test,batch_size=X_test.shape[1])
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)