import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew,norm,probplot
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",1000)
# np.set_printoptions(threshold=np.nan,)
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')

train=train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']>300000)].index)

n_train=train.shape[0]
y_train=train['SalePrice'].values
id_test=test['Id'].values

all_data=pd.concat([train,test])

# print(pd.get_dummies(all_data).shape)
# print(pd.get_dummies(all_data).head(2))


all_data=all_data.drop(['Id','SalePrice'],axis=1).reset_index(drop=True)

all_data_na=all_data.isnull().sum()/len(all_data) * 100
all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]
missing_data=pd.DataFrame({'Missing Ratio':all_data_na})

# 对x轴显示的文字旋转90°
plt.xticks(rotation=90)
plt.yticks(rotation=90)
# sns.barplot(x=all_data_na.index,y=all_data_na.values)
# plt.show()

# corrmat=train.corr()
# sns.heatmap(corrmat,vmax=0.9,square=True)
#
# plt.show()


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data_na=all_data.isnull().sum()/all_data.count() * 100


all_data['MSSubClass']=all_data['MSSubClass'].astype(str)
all_data['OverallCond']=all_data['OverallCond'].astype(str)

all_data['YrSold']=all_data['YrSold'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

for c in cols:
    all_data[c]=LabelEncoder().fit_transform(all_data[c].values)

all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']

numeric_feats=all_data.dtypes[all_data.dtypes!='object'].index
# print(numeric_feats)
# print(all_data.head(5))

skewed_feats=all_data[numeric_feats].apply(lambda x:skew(x)).sort_values(ascending=False)


skewness=pd.DataFrame({'Skew':skewed_feats})

# print(skewness)

skewness=skewness[abs(skewness)>0.75]

# print(skewness)

from scipy.special import boxcox1p

skewed_features=skewness.index
lam=0.15
# print(all_data.head(5))
for feat in skewed_features:
    all_data[feat]=boxcox1p(all_data[feat],lam)
# print(all_data.head(5))

# print(all_data.shape)
all_data=pd.get_dummies(all_data)

# print(all_data.shape)
# print(all_data.columns.values)

# print(all_data.head(50))

# print(y_train)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# pd.DataFrame({"SalePrice":y_train}).to_csv("./learn01.csv")
# fig = plt.figure()
# sns.distplot(y_train)
# plt.show()

y_train=np.log1p(y_train)

# fig = plt.figure()
# sns.distplot(y_train)
# plt.show()

train=all_data[:n_train]
test=all_data[n_train:]

n_folds=5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

knn=make_pipeline(RobustScaler(),KNeighborsRegressor())

'''
score=rmsle_cv(lasso)
print("\n Lasso score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(ENet)
print("\n ENet score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(KRR)
print("\n KRR score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(GBoost)
print("\n GBoost score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(model_xgb)
print("\n model_xgb score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(model_lgb)
print("\n model_lgb score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

score=rmsle_cv(knn)
print("\n knn score: {:0.4f} ({:.4f})\n".format(score.mean(),score.std()))

Lasso
score: 0.1237(0.0159)
ENet
score: 0.1237(0.0159)
KRR
score: 0.1260(0.0112)
GBoost
score: 0.1227(0.0118)
model_xgb
score: 0.1197(0.0073)
model_lgb
score: 0.1211(0.0083)
knn
score: 0.1919(0.0097)
'''

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# averaged_models = AveragingModels(models = (model_xgb,model_lgb,KRR, GBoost))

# score = rmsle_cv(averaged_models)
# print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# averaged_models.fit(train.values, y_train)
# y_pred_test=np.expm1(averaged_models.predict(test.values))

# print(y_pred_test)

model_xgb.fit(train.values, y_train)
y_pred_test_1=np.expm1(model_xgb.predict(test.values))

model_lgb.fit(train.values, y_train)
y_pred_test_2=np.expm1(model_lgb.predict(test.values))

KRR.fit(train.values, y_train)
y_pred_test_3=np.expm1(KRR.predict(test.values))

GBoost.fit(train.values, y_train)
y_pred_test_4=np.expm1(GBoost.predict(test.values))

lasso.fit(train.values, y_train)
y_pred_test_5=np.expm1(lasso.predict(test.values))

knn.fit(train.values, y_train)
y_pred_test_6=np.expm1(knn.predict(test.values))

# y_pred_test_3*0.1+y_pred_test_4*0.05+
y_pred_test=y_pred_test_1*0.4+y_pred_test_2*0.4+y_pred_test_6*0.2

pd.DataFrame({"Id":id_test,"SalePrice":y_pred_test}).to_csv("./result.csv",index=False)

'''
Lasso
score: 0.1237(0.0159)
ENet
score: 0.1237(0.0159)
KRR
score: 0.1260(0.0112)
GBoost
score: 0.1227(0.0118)
model_xgb
score: 0.1197(0.0073)
model_lgb
score: 0.1211(0.0083)
knn
score: 0.1919(0.0097)
'''