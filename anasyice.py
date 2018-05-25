import pandas as pd
import numpy as np

train=pd.read_csv('./data/train.csv')

numbers=train.select_dtypes(include=[np.number])

import seaborn as sns

corr=numbers.corr()

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

corrWithSales = corr['SalePrice']
data=corrWithSales.sort_values(ascending=False)

cols=[]
for k,v in data.iteritems():
    print(k,v)
    if abs(v)<0.1:
        continue
    else:
        cols.append(k)
print(cols)
