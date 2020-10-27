import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('bmi_and_life_expectancy.csv')

print(data.head(5))

x = data[['BMI']]
print(x)
print(x.shape)

y = data[['Life expectancy']]
print(y)
print(y.shape)

model = LinearRegression()
model.fit(x, y)

print(model.predict([[21.07931]]))