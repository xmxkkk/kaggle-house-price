import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def read_data(filepath,is_test=False):
    df=pd.read_csv(filepath)

    datas=[]
    for i in range(len(df)):
        datas.append([])


    col_set={
        "MSSubClass":16,
        "MSZoning":8,
        "Street":2,
        "Alley":2,
        "LotShape":4,
        "LandContour":4,
        "Utilities":4,
        "LotConfig":5,
        "LandSlope":3,
        "Neighborhood":25,
        "Condition1":9,
        "Condition2":9,
        "BldgType":5,
        "HouseStyle":8,
        "OverallQual":10,
        "OverallCond":10,
        "RoofStyle":6,
        "RoofMatl":8,
        "Exterior1st":17,
        "Exterior2nd":17,
        "MasVnrType":5,
        "ExterQual":5,
        "ExterCond":5,
        "Foundation":6,
        "BsmtQual":6,
        "BsmtCond":6,
        "BsmtExposure":5,
        "BsmtFinType1":6,
        "BsmtFinType2":6,
        "Heating":6,
        "HeatingQC":5,
        "CentralAir":2,
        "Electrical":5,
        "KitchenQual":5,
        "Functional":8,
        "FireplaceQu":6,
        "GarageType":6,
        "GarageFinish":3,
        "GarageQual":5,
        "GarageCond":5,
        "PavedDrive":3,
        "PoolQC":4,
        "Fence":4,
        "MiscFeature":5,
        "SaleType":10,
        "SaleCondition":6
    }


    for k,v in col_set.items():
        col_data=LabelEncoder().fit_transform(df[k].fillna('***'))
        idx=0
        for x in col_data:
            bb=[0] * (v+1)
            bb[x]=1
            datas[idx]=datas[idx]+bb
            idx=idx+1

    col_val=[
        "LotFrontage",
        "LotArea",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "MoSold",
        "YrSold"
    ]


    for col in col_val:
        mean=df[col].mean()
        xx=df[col].fillna(mean)
        col_data=MinMaxScaler().fit_transform(xx.values.reshape(-1,1))
        col_data=col_data.reshape(-1)
        idx=0
        for x in col_data:
            datas[idx].append(np.asscalar(x))
            idx=idx+1

    col0_val=[

    ]

    if False==is_test:
        labels=df['SalePrice'].values
    else:
        labels=df['Id'].values

    datas=np.array(datas)

    return datas,labels

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

model.add(Dense(389*4,input_shape=(X_train.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(389))
model.add(Dropout(0.25))
model.add(Dense(120))
model.add(Dense(1))

model.compile(optimizer='adam',loss=root_mean_squared_error,metrics=['mae'])

if os.path.exists("./x.w"):
    model.load_weights("./x.w")

for i in range(10):
    model.fit(X_train,y_train,batch_size=X_train.shape[0],epochs=100,verbose=1)
    print(i)
    model.save_weights("./x.w")

y_pred=model.predict(X_test,batch_size=X_test.shape[1])
result=pd.DataFrame({"Id":y_test,"SalePrice":y_pred.flatten()})
result.to_csv("./sample_submission.csv",index=False)