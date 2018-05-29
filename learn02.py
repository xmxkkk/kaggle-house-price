import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm,probplot
import pandas as pd
import scipy.stats as stats
def cc(mu,sigma):
    sampleNo=500

    np.random.seed(0)

    s=np.random.normal(mu,sigma,sampleNo)

    print(s)

    plt.figure()
    sns.distplot(s,bins=20)


    plt.figure()
    stats.probplot(s,plot=plt)
cc(40,100)


plt.show()
