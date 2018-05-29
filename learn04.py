import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
train=pd.read_csv('./data/train.csv')


df=pd.DataFrame({"x":[6,1,2,3],"y":[110,6,7,8]})

print(df.apply(lambda x:skew(x)))

print(train[['SalePrice']].apply(lambda x:skew(x)))


