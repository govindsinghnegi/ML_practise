import pandas as pd
from sklearn.linear_model import Lasso

data = pd.read_csv('data_regularization.csv', header=None)

x1 = data.iloc[:, 0]
print(x1.shape)

x2 = data.iloc[:, 1]
print(x2.shape)

x3 = data.iloc[:, 2]
print(x3.shape)

x4 = data.iloc[:, 3]
print(x4.shape)

x5 = data.iloc[:, 4]
print(x5.shape)

x6 = data.iloc[:, 5]
print(x6.shape)

x = data.iloc[:, 0:6]
print(x.head(5))

y = data.iloc[:, 6]
print(y.head(5))

lasso_reg = Lasso()
lasso_reg.fit(x, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

'''
# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data_regularization.csv', header = None)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]
print(X.head(5))
print(X.shape)
print(y.head(5))
print(y.shape)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

'''

