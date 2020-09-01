import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_regularization.csv', header=None)

x = data.iloc[:, 0:6]
print(x.head(5))

y = data.iloc[:, 6]
print(y.head(5))

standard_scaler = StandardScaler()
x_scaled = standard_scaler.fit_transform(x)

lasso_reg = Lasso()
lasso_reg.fit(x_scaled, y)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)