import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')

'''
观察数据的方法
1.查看跟SalePrice相关性高的字段,通过观察图片，找到不呈现正太分布的数据，并删除异常的数据
OverallQual      0.790982
GrLivArea        0.708624
GarageCars       0.640409
GarageArea       0.623431
TotalBsmtSF      0.613581
1stFlrSF         0.605852
'''
train_corr=train.corr()
print(train_corr['SalePrice'].sort_values(ascending=False))

plt.subplot(2,3,1)
plt.scatter(x=train['OverallQual'],y=train['SalePrice'])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")

plt.subplot(2,3,2)
plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

plt.subplot(2,3,3)
plt.scatter(x=train['GarageCars'],y=train['SalePrice'])
plt.xlabel("GarageCars")
plt.ylabel("SalePrice")

plt.subplot(2,3,4)
plt.scatter(x=train['GarageArea'],y=train['SalePrice'])
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")

plt.subplot(2,3,5)
plt.scatter(x=train['TotalBsmtSF'],y=train['SalePrice'])
plt.xlabel("TotalBsmtSF")
plt.ylabel("SalePrice")

plt.subplot(2,3,6)
plt.scatter(x=train['1stFlrSF'],y=train['SalePrice'])
plt.xlabel("1stFlrSF")
plt.ylabel("SalePrice")


print(train[(train['OverallQual']==10)&(train['SalePrice']<200000)].index)
print(train[(train['GrLivArea']>4000)&(train['SalePrice']<200000)].index)
print(train[(train['GarageArea']>1200)&(train['SalePrice']<300000)].index)
print(train[(train['TotalBsmtSF']>4000)&(train['SalePrice']<200000)].index)
print(train[(train['1stFlrSF']>4000)&(train['SalePrice']<200000)].index)

train=train.drop(train[(train['OverallQual']==10)&(train['SalePrice']<200000)].index)
train=train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<200000)].index)
train=train.drop(train[(train['GarageArea']>1200)&(train['SalePrice']<300000)].index)
train=train.drop(train[(train['TotalBsmtSF']>4000)&(train['SalePrice']<200000)].index)
train=train.drop(train[(train['1stFlrSF']>4000)&(train['SalePrice']<200000)].index)

'''
2.观察train和test的空值率
'''

n_train=train.shape[0]
y_train=train['SalePrice']
X_train=train.drop(["Id","SalePrice"],axis=1)
id_test=test['Id']
X_test=test.drop(["Id"],axis=1)

all_data=X_train.append(X_test,ignore_index=True)
all_data=pd.DataFrame(all_data)


def missing_data(data_):
    all_data_na = data_.isnull().sum() / data_.isnull().count()
    all_data_na = all_data_na.sort_values(ascending=False)
    missing_ratio=pd.DataFrame({"MissingRatio":all_data_na})
    missing_ratio=missing_ratio.drop(missing_ratio[missing_ratio['MissingRatio']==0].index,axis=0)

    return missing_ratio
missing_ratio=missing_data(all_data)
print(missing_ratio)

'''
字段空置率
PoolQC            0.995876
MiscFeature       0.963574
Alley             0.937457
Fence             0.806873
FireplaceQu       0.473540
LotFrontage       0.177320
GarageCond        0.055670
GarageType        0.055670
GarageYrBlt       0.055670
GarageFinish      0.055670
GarageQual        0.055670
BsmtExposure      0.026117
BsmtFinType2      0.026117
BsmtFinType1      0.025430
BsmtCond          0.025430
BsmtQual          0.025430
MasVnrArea        0.005498
MasVnrType        0.005498
Electrical        0.000687
'''

all_data_na=all_data[missing_ratio.index]
'''
todo
这些字段是字符串字段，但是有空值，可以做不同处理，有些需要用最多出现的值填充，有些需要用None填充
Index(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',
       'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure',
       'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType',
       'Electrical'],
      dtype='object')
'''
str_features=all_data_na.dtypes[all_data_na.dtypes=='object'].index
for feat in str_features:
    all_data[feat]=all_data[feat].fillna('None')

print(str_features)

'''
这些字段是数字不适用None填充
Index(['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath',
       'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
       'GarageArea', 'GarageCars']
'''
num_features = all_data_na.dtypes[all_data_na.dtypes != 'object'].index
print(num_features)

# all_data['GarageYrBlt']=all_data['GarageYrBlt'].append(str)
# all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(None)
all_data=all_data.drop(['GarageYrBlt'],axis=1)

all_data['BsmtHalfBath']=all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0])
all_data['BsmtFullBath']=all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0])

all_data['BsmtFinSF1']=all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2']=all_data['BsmtFinSF2'].fillna(0)
all_data['BsmtUnfSF']=all_data['BsmtUnfSF'].fillna(0)
all_data['TotalBsmtSF']=all_data['TotalBsmtSF'].fillna(0)
all_data['GarageArea']=all_data['GarageArea'].fillna(0)
all_data['GarageCars']=all_data['GarageCars'].fillna(0)

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
'''
没有车库用0填充
'''
# all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(0)

'''
LotFrontage 距离街边的直线距离，因为不好用NaN表示没有这个数据，不方便用0填充
可以选择用0填充，也可以用相关数据判断这个字段，我们看最相关的字段是1stFlrSF。
'''
print(train_corr['LotFrontage'].sort_values(ascending=False))

'''
    todo
    1.使用相关联最大的字段填充
'''
temp=all_data['1stFlrSF'].copy(True)
all_data['1stFlrSF']=(all_data['1stFlrSF']/20).apply(int)
all_data["LotFrontage"] = all_data.groupby("1stFlrSF")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_data['1stFlrSF']=temp
all_data['LotFrontage']=all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
'''
    2.使用平均值或中间值填充
'''
# all_data['LotFrontage']=all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())

'''
再来看一下数据空值率
'''
missing_ratio=missing_data(all_data)
print(missing_ratio)

all_data['MSSubClass']=all_data['MSSubClass'].apply(str)

for x in all_data:
    if 'object'==all_data[x].dtypes:
        all_data[x]=LabelEncoder().fit_transform(all_data[x])


# all_data['SalePrice']=y_train
# print(all_data.corr()['SalePrice'].sort_values(ascending=False))


# all_data=pd.get_dummies(all_data)
print(all_data.shape)

#

X_train = all_data[:n_train]
X_test = all_data[n_train:]
id_test=id_test.values

'''
对SalePrice的处理
因为SalePrice是偏态分布，对SalePrice进行
'''
plt.figure()
plt.subplot(1,2,1)
sns.distplot(y_train)

y_train=np.log1p(y_train.values)
plt.subplot(1,2,2)
sns.distplot(y_train)




# plt.show()

