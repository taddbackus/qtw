from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

stuff = fetch_california_housing()
x = pd.DataFrame(stuff['data'],columns=stuff['feature_names'])
y = stuff['target']
print(x)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled = pd.DataFrame(data=X_scaled,columns=x.columns)


lambdas_ = np.logspace(-6,6,20)

lassoModel = Lasso()
print('Lasso Model')
for i in np.logspace(-8,8,20):
    lassoModel.alpha=i
    print(i, cross_val_score(lassoModel,X_scaled,y,scoring='neg_mean_squared_error').mean())
print('==========')
print('Ridge Model')
ridgeModel = Ridge()
for i in np.logspace(-8,8,20):
    ridgeModel.alpha=i
    print(i, cross_val_score(ridgeModel,X_scaled,y,scoring='neg_mean_squared_error').mean())
print('==============')

elasticModel = ElasticNet()
params = {'alpha':np.logspace(-6,6,20),
          'l1_ratio':np.linspace(0,1,11)}
print(params)
#grid_search = GridSearchCV(elasticModel, params, cv=50, n_jobs=-1, scoring='neg_mean_squared_error')
#grid_search.fit(X_scaled,y)
#print('Elastic Net Model')
#print(grid_search.best_score_)
#print(grid_search.best_params_)



'''


Elastic Net Model
-0.5383919080958011
{'alpha': 0.0014384498882876629, 'l1_ratio': 0.1}
'''