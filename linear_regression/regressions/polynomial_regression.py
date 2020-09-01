from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data_polynomial.csv')
print(train_data.head(5))

X = train_data[['Var_X']]
print(X.shape)

Y = train_data[['Var_Y']]
print(Y.shape)

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = None
X_poly = None

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = None
