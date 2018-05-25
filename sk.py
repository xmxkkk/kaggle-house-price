from input_data import Data

data=Data()
X_train,y_train=data.read_data("./data/train.csv",'train')
print(X_train.shape,y_train.shape)

X_test,y_test=data.read_data("./data/test.csv",'test')
print(X_test.shape,y_test.shape)


from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

'''
# 1
clf=NearestCentroid()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
'''
'''
clf=KNeighborsRegressor()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(len(y_pred))
'''
'''
X_train=preprocessing.normalize(X_train)
clf=DecisionTreeRegressor()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(len(y_pred))
'''
'''
clf=SVR()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
'''
'''
clf=ExtraTreesClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
'''
'''
clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
'''

clf=LogisticRegression()
# clf=RFE(model,3)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

result = pd.DataFrame({"Id": y_test, "SalePrice": y_pred})
print(result)
result.to_csv("./sample_submission.csv", index=False)
