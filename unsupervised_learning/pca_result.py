import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unsupervised_learning.helper_functions_pca import do_pca
from unsupervised_learning.helper_functions_pca import show_images

#read in our dataset
train = pd.read_csv('train_pca.csv')
train.fillna(0, inplace=True)

# save the labels to a Pandas series target
y = train['label']
# Drop the label feature
train.fillna(0, inplace=True)
X = train.drop("label",axis=1)

show_images(5)
pca, X_pca = do_pca(10, X)

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components
    INPUT: pca - the result of instantian of PCA in scikit learn
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i] * 100)[:4])), (ind[i] + 0.2, vals[i]), va="bottom", ha="center",
                    fontsize=12)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    plt.show()


scree_plot(pca)

print(pca.components_.shape)


def plot_component(pca, comp):
    '''
    Plots an image associated with each component to understand how the weighting
    of the components
    INPUT:
          pca - pca object created from PCA in sklearn
          comp - int - the component you want to see starting at 0
    '''
    if comp <= len(pca.components_):
        mat_data = np.asmatrix(pca.components_[comp]).reshape(28, 28)  # reshape images
        plt.imshow(mat_data);  # plot the data
        plt.xticks([])  # removes numbered labels on x-axis
        plt.yticks([])  # removes numbered labels on y-axis
        plt.show()
    else:
        print('That is not the right input, please read the docstring before continuing.')


# Plot the first component
plot_component(pca, 3)

solution_five = {
    'This component looks like it will assist in identifying zero': 0,
    'This component looks like it will assist in identifying three': 3
}

'''
you have had an opportunity to look at the two major parts of PCA:
I. The amount of variance explained by each component. This is called an eigenvalue.
II. The principal components themselves, each component is a vector of weights. In this case, the principal components 
help us understand which pixels of the image are most helpful in identifying the difference between between digits. 
Principal components are also known as eigenvectors.
'''


