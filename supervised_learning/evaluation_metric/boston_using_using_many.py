from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import numpy as np


boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

a = 'regression'
b = 'classification'
c = 'both regression and classification'

models = {
    'decision trees': c,
    'random forest': c,
    'adaptive boosting': c,
    'logistic regression': b,
    'linear regression': a
}

decision_trees = DecisionTreeRegressor()
decision_trees.fit(X_train, y_train)
y1 = decision_trees.predict(X_test)

random_forrest = RandomForestRegressor()
random_forrest.fit(X_train, y_train)
y2 = random_forrest.predict(X_test)

adaptive_boosting = AdaBoostRegressor()
adaptive_boosting.fit(X_train, y_train)
y3 = adaptive_boosting.predict(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y4 = linear_model.predict(X_test)

a = 'regression'
b = 'classification'
c = 'both regression and classification'

metrics = {
    'precision': b,
    'recall': b,
    'accuracy': b,
    'r2_score': a,
    'mean_squared_error': a,
    'area_under_curve': b,
    'mean_absolute_area': a
}

from sklearn.metrics import r2_score
def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst

from sklearn.metrics import mean_squared_error
def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''

    return np.mean((actual-preds)**2)

from sklearn.metrics import mean_absolute_error
def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''

    return np.mean(np.abs(actual-preds))

print('------------ using decision trees-----')
print(r2(y_test, y1))
print(r2_score(y_test, y1))
print(mse(y_test, y1))
print(mean_squared_error(y_test, y1))
print(mae(y_test, y1))
print(mean_absolute_error(y_test, y1))

print('------------ using random forrest -----')
print(r2_score(y_test, y2))
print(mean_squared_error(y_test, y2))
print(mean_absolute_error(y_test, y2))

print('------------ using adaptive boosting -----')
print(r2_score(y_test, y3))
print(mean_squared_error(y_test, y3))
print(mean_absolute_error(y_test, y3))

print('------------ using linear model -----')
print(r2_score(y_test, y4))
print(mean_squared_error(y_test, y4))
print(mean_absolute_error(y_test, y4))



