import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

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

