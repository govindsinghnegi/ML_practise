import matplotlib.pyplot as plt
from sklearn import datasets

n_samples = 1000
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[5, 1, 0.5], random_state=3)
X, y = varied[0], varied[1]

plt.figure( figsize=(16,12))
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='black', lw=1.5, s=100, cmap=plt.get_cmap('viridis'))
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
pred = kmeans.fit_predict(X)

plt.figure( figsize=(16,12))
plt.scatter(X[:,0], X[:,1], c=pred, edgecolor='black', lw=1.5, s=100, cmap=plt.get_cmap('viridis'))
plt.show()

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3).fit(X)
gmm = gmm.fit(X)
pred_gmm = gmm.predict(X)

plt.figure( figsize=(16,12))
plt.scatter(X[:,0], X[:,1], c=pred_gmm, edgecolor='black', lw=1.5, s=100, cmap=plt.get_cmap('viridis'))
plt.show()

import seaborn as sns
iris = sns.load_dataset("iris")
print('iris dataset : {}'.format(iris.head()))

g = sns.PairGrid(iris, hue="species", palette=sns.color_palette("cubehelix", 3), vars=['sepal_length','sepal_width',
                                                                                       'petal_length','petal_width'])
g.map(plt.scatter)
plt.show()

kmeans_iris = KMeans(n_clusters=3)
pred_kmeans_iris = kmeans_iris.fit_predict(iris[['sepal_length','sepal_width','petal_length','petal_width']])

iris['kmeans_pred'] = pred_kmeans_iris
g = sns.PairGrid(iris, hue="kmeans_pred", palette=sns.color_palette("cubehelix", 3), vars=['sepal_length','sepal_width',
                                                                                           'petal_length','petal_width'])
g.map(plt.scatter)
plt.show()

from sklearn.metrics import adjusted_rand_score

iris_kmeans_score = adjusted_rand_score(iris['species'], iris['kmeans_pred'])

# Print the score
print('iris_kmeans_score: {}'.format(iris_kmeans_score))

gmm_iris = GaussianMixture(n_components=3).fit(iris[['sepal_length','sepal_width','petal_length','petal_width']])
pred_gmm_iris = gmm_iris.predict(iris[['sepal_length','sepal_width','petal_length','petal_width']])

iris['gmm_pred'] = pred_gmm_iris
iris_gmm_score = adjusted_rand_score(iris['species'], iris['gmm_pred'])

print('iris_gmm_score: {}'.format(iris_gmm_score))