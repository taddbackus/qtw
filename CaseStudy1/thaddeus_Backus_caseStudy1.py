import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

scData = pd.read_csv('/Users/taddbackus/School/fall23/qtw/cs1/data/train.csv')

scUnique = pd.read_csv('/Users/taddbackus/School/fall23/qtw/cs1/data/unique_m.csv')
scUnique = scUnique.drop(['critical_temp','material'],axis=1)


critTempDF = pd.concat([scData,scUnique],axis=1)
print(critTempDF)

X = critTempDF.drop('critical_temp',axis=1)
y = critTempDF['critical_temp']
print(X.shape)
print(y.shape)

lr = LinearRegression()
yPred = cross_val_predict(lr, X, y, cv=5)
print(cross_val_score(lr, X, y, cv=5,scoring='neg_mean_squared_error'))
plt.scatter(y,yPred)
#plt.show()
print(yPred.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(data=X_scaled,columns=X.columns)
print(X_scaled)

lrS = LinearRegression()
lrS.fit(X_scaled,y)
for i in range(len(X_scaled.columns)):
    print(X_scaled.columns[i],lrS.coef_[i])

lrL1 = Lasso(alpha=1)
lrL1.fit(X_scaled,y)

cvLassoModel = Lasso()
for i in np.arange(0.0,1.0,.01):
    cvLassoModel.alpha = i
    print(i, cross_val_score(cvLassoModel,X_scaled,y,scoring='neg_mean_squared_error'))
    print('==========================')