import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def read_data(filepath,is_test=False):
    df=pd.read_csv(filepath)
    df=shuffle(df)
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