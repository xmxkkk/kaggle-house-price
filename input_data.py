import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",1000)
np.set_printoptions(threshold=np.nan,)



test_df=pd.read_csv("./data/test.csv")
df=pd.read_csv("./data/train.csv")

df=pd.concat([df,test_df])
df.drop('SalePrice')
df.drop('Id')

# print(df.columns)

# print(df.corr()['LotFrontage'])

df['MSSubClass']=df['MSSubClass'].apply(str)
df['MSSubClass']=LabelEncoder().fit_transform(df['MSSubClass'])

df['MSZoning']=df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['MSZoning']=LabelEncoder().fit_transform(df['MSZoning'])

temp=df['LotArea'].copy(True)
df['LotArea']=(df['LotArea']/1000).apply(int)
df["LotFrontage"] = df.groupby("LotArea")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
df['LotArea']=temp


df['Street']=LabelEncoder().fit_transform(df['Street'])
df['LotShape']=LabelEncoder().fit_transform(df['LotShape'])
df['LandContour']=LabelEncoder().fit_transform(df['LandContour'])

df['Utilities']=LabelEncoder().fit_transform(df['Utilities'].fillna(df['Utilities'].mode()[0]))

df['LotConfig']=LabelEncoder().fit_transform(df['LotConfig'])

df['LandSlope']=LabelEncoder().fit_transform(df['LandSlope'])

df['Neighborhood']=LabelEncoder().fit_transform(df['Neighborhood'])

df['Condition1']=LabelEncoder().fit_transform(df['Condition1'])

df['Condition2']=LabelEncoder().fit_transform(df['Condition2'])

df['BldgType']=LabelEncoder().fit_transform(df['BldgType'])

df['HouseStyle']=LabelEncoder().fit_transform(df['HouseStyle'])

df['OverallQual']=LabelEncoder().fit_transform(df['OverallQual'])

df['OverallCond']=LabelEncoder().fit_transform(df['OverallCond'])

df['YearBuilt']=LabelEncoder().fit_transform(df['YearBuilt'])

df['YearRemodAdd']=LabelEncoder().fit_transform(df['YearRemodAdd'])

df['RoofStyle']=LabelEncoder().fit_transform(df['RoofStyle'])

df['RoofMatl']=LabelEncoder().fit_transform(df['RoofMatl'])

df['Exterior1st']=LabelEncoder().fit_transform(df['Exterior1st'].fillna(df['Exterior1st'].mode()[0]))

df['Exterior2nd']=LabelEncoder().fit_transform(df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0]))

df['MasVnrType']=LabelEncoder().fit_transform(df['MasVnrType'].fillna('None'))

df['MasVnrArea']=df['MasVnrArea'].fillna(0)

df['ExterQual']=LabelEncoder().fit_transform(df['ExterQual'])

df['ExterCond']=LabelEncoder().fit_transform(df['ExterCond'])

df['Foundation']=LabelEncoder().fit_transform(df['Foundation'])

df['BsmtQual']=LabelEncoder().fit_transform(df['BsmtQual'].fillna('None'))

df['BsmtCond']=LabelEncoder().fit_transform(df['BsmtCond'].fillna('None'))

df['BsmtExposure']=LabelEncoder().fit_transform(df['BsmtExposure'].fillna('None'))

df['BsmtFinType1']=LabelEncoder().fit_transform(df['BsmtFinType1'].fillna('None'))

# df['BsmtFinSF1']=LabelEncoder().fit_transform(df['BsmtFinSF1'])

df['BsmtFinSF1']=df['BsmtFinSF1'].fillna(0)

df['BsmtFinSF2']=df['BsmtFinSF2'].fillna(0)

df['BsmtUnfSF']=df['BsmtUnfSF'].fillna(0)

df['TotalBsmtSF']=df['TotalBsmtSF'].fillna(0)
df['1stFlrSF']=df['1stFlrSF'].fillna(0)
df['2ndFlrSF']=df['2ndFlrSF'].fillna(0)


df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df.drop('TotalBsmtSF')
df.drop('1stFlrSF')
df.drop('2ndFlrSF')

df['BsmtFinType2']=LabelEncoder().fit_transform(df['BsmtFinType2'].fillna('None'))

df['Heating']=LabelEncoder().fit_transform(df['Heating'])

df['HeatingQC']=LabelEncoder().fit_transform(df['HeatingQC'])

df['CentralAir']=LabelEncoder().fit_transform(df['CentralAir'])

df['Electrical']=LabelEncoder().fit_transform(df['Electrical'].fillna('None'))
# print(df['Electrical'])

df['BsmtFullBath']=df['BsmtFullBath'].fillna(0)

df['BsmtHalfBath']=df['BsmtHalfBath'].fillna(0)

print(df['TotRmsAbvGrd'].isnull().any())

df['KitchenQual']=LabelEncoder().fit_transform(df['KitchenQual'].fillna(df['KitchenQual'].mode()[0]))

df['Functional']=LabelEncoder().fit_transform(df['Functional'].fillna('None'))
print(df['Functional'])


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtCond']=LabelEncoder().fit_transform(df['BsmtCond'])

df['Alley']=LabelEncoder().fit_transform(df['Alley'].fillna('None'))



# for x in df:
#     if df[x].isnull().any():
#         print(x)

# df['Neighborhood']=LabelEncoder().fit_transform(df['Neighborhood'])
# print(df['LotFrontage'].corr(df['Neighborhood']))