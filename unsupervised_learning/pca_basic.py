import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import unsupervised_learning.test_code_pca as t
from unsupervised_learning.helper_functions_pca import fit_random_forest_classifier, do_pca, plot_components
from unsupervised_learning.helper_functions_pca import show_images_by_digit

train = pd.read_csv('train_pca.csv')
train.fillna(0, inplace=True)

# save the labels to a Pandas series target
y = train['label']

# Drop the label feature
X = train.drop("label",axis=1)

#Check Your Solution
t.question_two_check(y, X)

show_images_by_digit(2) # Try looking at a few other digits

fit_random_forest_classifier(X, y)

pca, X_pca = do_pca(2, X) #performs PCA to create two components

fit_random_forest_classifier(X_pca, y)

plot_components(X_pca[:100], y[:100])

comps = []
accs = []
for comp in range(10, 70):
    comps.append(comp)
    pca, X_pca = do_pca(comp, X)
    acc = fit_random_forest_classifier(X_pca, y)
    accs.append(acc)
    if acc > .90:
        print("With only {} components, a random forest acheived an accuracy of {}.".format(comp, acc))
        break

plt.plot(comps, accs, 'bo');
plt.xlabel('Number of Components');
plt.ylabel('Accuracy');
plt.title('Number of Components by Accuracy');
plt.show()

# The max accuracy and corresponding number of components
np.max(accs), comps[np.where(accs == np.max(accs))[0][0]]