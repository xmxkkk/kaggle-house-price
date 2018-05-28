import pandas as pd
import numpy as np
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",1000)
np.set_printoptions(threshold=np.nan,)



test_df=pd.read_csv("./data/test.csv")
df=pd.read_csv("./data/train.csv")

df=pd.concat([df,test_df])
df=df.drop(['SalePrice','Id'],axis=1)

# print(df.columns)

# print(df.corr()['LotFrontage'])

df['MSSubClass']=df['MSSubClass'].apply(str)
df['GarageYrBlt']=df['GarageYrBlt'].apply(str)

temp=df['LotArea'].copy(True)
df['LotArea']=(df['LotArea']/1000).apply(int)
df["LotFrontage"] = df.groupby("LotArea")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df['LotArea']=temp

df['Functional']=df['Functional'].fillna('Typ')
df['OverallCond'] = df['OverallCond'].astype(str)

col_fill_none=[
    'MSSubClass','Street','Alley','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
    'Condition1','Condition2','BldgType','HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl',
    'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
    'BsmtFinType2','Heating','HeatingQC','CentralAir','Functional','FireplaceQu',
    'GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature',
    'YrSold','SaleCondition','OverallCond'
]

col_fill_mode=[
    'MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrArea','Electrical','KitchenQual','SaleType'
]

col_fill_0=[
    'OverallQual','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
    'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
    'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
    'ScreenPorch','PoolArea','MiscVal','MoSold','LotFrontage'
]

for col in col_fill_none:
    df[col]=LabelEncoder().fit_transform(df[col].fillna('None'))

for col in col_fill_mode:
    df[col] = LabelEncoder().fit_transform(df[col].fillna(df[col].mode()[0]))

for col in col_fill_0:
    df[col] = df[col].fillna(0)

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df=df.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','Utilities'],axis=1)

# print(df.head(10))

for x in df:
    if df[x].isnull().any():
        print(x)


numeric_feats = df.dtypes[df.dtypes != "object"].index

        # Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
print(skewness.head(10))

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)

df = pd.get_dummies(df)
print(df.shape)
# df['Neighborhood']=LabelEncoder().fit_transform(df['Neighborhood'])
# print(df['LotFrontage'].corr(df['Neighborhood']))