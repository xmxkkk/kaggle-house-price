import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import backend as K

class Data():
    def __init__(self):
        pass
    def read_data(self,filepath=None,type='train'):
        data=pd.read_csv(filepath)

        result2=None
        if type=='train':
            result2=data['SalePrice'].values
            data = data.drop(['Id', 'SalePrice'], axis=1)
        else:
            result2=data['Id'].values
            data = data.drop(['Id'], axis=1)

        columns = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle'
            ,'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1'
            ,'BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir'
            ,'Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath'
            ,'HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces'
            ,'FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual'
            ,'GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea'
            ,'PoolQC','Fence','MiscFeature','MiscVal','MoSold','SaleType','SaleCondition','YearBuilt','YearRemodAdd','YrSold','LotArea']

        data = data.fillna({'Alley': "", 'MasVnrType': "", "BsmtQual": "", "Foundation": "", "BsmtCond": ""
                                 , "BsmtExposure": "", "BsmtFinType1": "", "BsmtFinType2": "", "CentralAir": ""
                                 , "Electrical": "", "FireplaceQu": "", "GarageType": "", "GarageFinish": ""
                                 , "GarageQual": "", "GarageCond": "", "PoolQC": "", "Fence": "", "MiscFeature": ""
                                 , "MSZoning": "", "Utilities": "", "Exterior1st": "", "Exterior2nd": "",
                              "KitchenQual": ""
                                 , "Functional": "", "SaleType": "","LotFrontage":0,"BsmtFinSF1":0})

        self.columns=columns

        output = data.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)

        # output=output[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 'ScreenPorch', 'EnclosedPorch', 'KitchenAbvGr']]

        # output['YearBuilt']=output['YearBuilt']-1900;
        # output['YearRemodAdd'] = output['YearRemodAdd'] - 1900;
        # output['YrSold'] = output['YrSold'] - 2000;
        # output['YrSold'] = output['YrSold'] - 2000;

        output.to_csv("./{}.csv".format(type),index=False)
        return output.values,result2


# data=Data()
# X_train,y_train=data.read_data("./data/train.csv",'train')
# print(X_train.shape,y_train.shape)
#
# X_test,y_test=data.read_data("./data/test.csv",'test')
# print(X_test.shape,y_test.shape)