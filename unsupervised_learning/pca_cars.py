import pandas as pd
from IPython import display

import unsupervised_learning.test_code2_pca as t
from unsupervised_learning.helper_functions_pca import do_pca, pca_results

df = pd.read_csv('cars.csv')

print(df.head())
print(df.describe())
print(df.shape)

a = 7
b = 66
c = 387
d = 18
e = 0.23
f = 0.05


solution_1_dict = {
    'The number of cars in the dataset': c,
    'The number of car features in the dataset': d,
    'The number of dummy variables in the dataset': a,
    'The proportion of minivans in the dataset': f,
    'The max highway mpg for any car': b
}

display.HTML(t.check_question_one(solution_1_dict))

a = True
b = False

solution_2_dict = {
    'The components span the directions of maximum variability.': a,
    'The components are always orthogonal to one another.': a,
    'Eigenvalues tell us the amount of information a component holds': a
}

t.check_question_two(solution_2_dict)

pca, X_pca = do_pca(3, df)
result = pca_results(df, pca)
print(result.head())

a = 'car weight'
b = 'sports cars'
c = 'gas mileage'
d = 0.4352
e = 0.3061
f = 0.1667
g = 0.7053

solution_5_dict = {
    'The first component positively weights items related to': c,
    'The amount of variability explained by the first component is': d,
    'The largest weight of the second component is related to': b,
    'The total amount of variability explained by the first three components': g
}

t.check_question_five(solution_5_dict)

for comp in range(3, df.shape[1]):
    pca, X_pca = do_pca(comp, df)
    comp_check = pca_results(df, pca)
    if comp_check['Explained Variance'].sum() > 0.85:
        break

num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(comp_check.shape[0],
                                                                                                comp_check[
                                                                                                    'Explained Variance'].sum()))
display.HTML(t.question_check_six(num_comps))