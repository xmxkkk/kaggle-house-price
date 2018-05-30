import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
train=pd.read_csv('./data/train.csv')
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

df=pd.DataFrame({"x":[6,1,2,3],"y":[110,6,7,8]})

print(df.apply(lambda x:skew(x)))

print(train[['SalePrice']].apply(lambda x:skew(x)))


model=make_pipeline(RobustScaler())

print(df.values)
result=RobustScaler().fit_transform(df.values)
print(result)

data=np.random.normal(0,10,1000)

data=RobustScaler().fit_transform(data.reshape((-1,1)))

plt.subplot(2,2,1)
sns.distplot(data)

plt.subplot(2,2,2)
data=np.expm1(data)
sns.distplot(data)

plt.subplot(2,2,3)
data=np.log1p(data)
sns.distplot(data)

# plt.subplot(2,2,4)
# data=np.log1p(data)
# sns.distplot(data)


plt.show()


