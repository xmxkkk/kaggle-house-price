from data_process import X_train,y_train,X_test,id_test
from sklearn.model_selection import KFold,cross_val_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,LassoCV,HuberRegressor,LassoLarsIC,LogisticRegression,OrthogonalMatchingPursuit\
    ,OrthogonalMatchingPursuitCV,PassiveAggressiveRegressor,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


'''
实验模型的准确率,讲训练数据分成5组，分别训练，并统计预测值得平均值和标准差
'''
n_folds = 5

def rmsle_cv(model,del_col=None):
    temp_train=None
    if del_col is not None:
        temp_train=X_train.drop([del_col],axis=1)
    else:
        temp_train=X_train
    print(temp_train.shape)
    temp_train=pd.get_dummies(temp_train)
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(temp_train.values)
    rmse = np.sqrt(-cross_val_score(model, temp_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

lr=make_pipeline(RobustScaler(),LinearRegression())
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KNN=make_pipeline(RobustScaler(),KNeighborsRegressor(n_neighbors=3))
Lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005, random_state=3))
LassoCV=make_pipeline(RobustScaler(),LassoCV(random_state=3))
HuberRegressor=make_pipeline(RobustScaler(),HuberRegressor(alpha=0.001))
LassoLarsIC_aic=make_pipeline(RobustScaler(),LassoLarsIC(criterion='aic'))
LassoLarsIC_bic=make_pipeline(RobustScaler(),LassoLarsIC(criterion='bic'))
OrthogonalMatchingPursuit=make_pipeline(RobustScaler(),OrthogonalMatchingPursuit())
OrthogonalMatchingPursuitCV=make_pipeline(RobustScaler(),OrthogonalMatchingPursuitCV())
PassiveAggressiveRegressor=make_pipeline(RobustScaler(),PassiveAggressiveRegressor(C=0.01))
Ridge=make_pipeline(RobustScaler(),Ridge(alpha=0.0001,random_state=42))


'''
scores=pd.DataFrame()

for name,model in {"Ridge":Ridge,"PassiveAggressiveRegressor":PassiveAggressiveRegressor,"OrthogonalMatchingPursuitCV":OrthogonalMatchingPursuitCV,"OrthogonalMatchingPursuit":OrthogonalMatchingPursuit,"LassoLarsIC_bic":LassoLarsIC_bic,"LassoLarsIC_aic":LassoLarsIC_aic,"ENet":ENet,"Lasso":Lasso,"LassoCV":LassoCV,"HuberRegressor":HuberRegressor}.items():
    score=rmsle_cv(model)
    print("{}   score:        {:.5f} ({:.5f})".format(name,score.mean(),score.std()))
    scores=scores.append(pd.Series([name,score.mean(),score.std()]),ignore_index=True)

scores.columns=['name','mean','std']
print(scores.sort_values(by='mean'))
print(scores.sort_values(by='std'))
'''

'''
最好的模型是Lasso
                          name      mean       std
7                        Lasso  0.111857  0.006180
6                         ENet  0.111968  0.006319
3    OrthogonalMatchingPursuit  0.118106  0.006487
2  OrthogonalMatchingPursuitCV  0.118127  0.006424
8                      LassoCV  0.124482  0.005702
0                        Ridge  0.127379  0.010475
5              LassoLarsIC_aic  0.132548  0.002327
4              LassoLarsIC_bic  0.151539  0.006531
1   PassiveAggressiveRegressor  0.829536  0.323152
9               HuberRegressor  2.152432  0.469725
'''

'''
Lasso                           0.12049
ENet                            0.12036
OrthogonalMatchingPursuit       0.12685
'''
scores=pd.DataFrame()

idx=0
for col in X_train:
    if idx>20:
        pass
#"Lasso":Lasso,"OrthogonalMatchingPursuit":OrthogonalMatchingPursuit
    for name,model in {"ENet":ENet}.items():
        score = rmsle_cv(model,col)
        print("{}   score:        {}        {:.5f} ({:.5f})".format(name,col, score.mean(), score.std()))
        scores = scores.append(pd.Series([name,col, score.mean(), score.std()]), ignore_index=True)

    idx=idx+1

scores.columns=['name','delcol','mean','std']

print(scores.sort_values(by='mean'))

X_train=X_train.drop(scores.sort_values(by='mean').head(20)['delcol'].values,axis=1)
X_test=X_test.drop(scores.sort_values(by='mean').head(20)['delcol'].values,axis=1)

X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)


# print(scores.sort_values(by='std'))

ENet.fit(X_train.values,y_train)
y_test_pred=ENet.predict(X_test.values)
y_test_pred=np.expm1(y_test_pred)


result=pd.DataFrame({"Id":id_test,"SalePrice":y_test_pred})
result.to_csv("./result_model.csv",index=False)

