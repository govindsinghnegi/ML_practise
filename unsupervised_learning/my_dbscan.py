import pandas as pd
dataset_1 = pd.read_csv('dbscan_blobs.csv')[:80].values

from unsupervised_learning import dbscan_lab_helper as helper
helper.plot_dataset(dataset_1)

from sklearn import cluster

dbscan = cluster.DBSCAN()
clustering_labels_1 = dbscan.fit_predict(dataset_1)
helper.plot_clustered_dataset(dataset_1, clustering_labels_1)

# Epsilon-> the radius of each point's neighborhood, the default value in sklearn is 0.5
helper.plot_clustered_dataset(dataset_1, clustering_labels_1, neighborhood=True)

# Epsilon value of 0.5 is too small for this dataset. We need to increase it so the points in a blob overlap each others'
# neighborhoods, but not to the degree where a single cluster would span two blobs.

epsilon=2
dbscan = cluster.DBSCAN(eps=epsilon)
clustering_labels_2 = dbscan.fit_predict(dataset_1)
helper.plot_clustered_dataset(dataset_1, clustering_labels_2, neighborhood=True, epsilon=epsilon)

dataset_2 = pd.read_csv('dbscan_varied.csv')[:300].values
helper.plot_dataset(dataset_2, xlim=(-14, 5), ylim=(-12, 7))
dbscan = cluster.DBSCAN()
clustering_labels_3 = dbscan.fit_predict(dataset_2)
helper.plot_clustered_dataset(dataset_2, clustering_labels_3, xlim=(-14, 5), ylim=(-12, 7), neighborhood=True, epsilon=0.5)

eps=1.32
min_samples=50

# Cluster with DBSCAN
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
clustering_labels_4 = dbscan.fit_predict(dataset_2)
helper.plot_clustered_dataset(dataset_2, clustering_labels_4, xlim=(-14, 5), ylim=(-12, 7), neighborhood=True, epsilon=0.5)

'''
Scenario 1: Break the dataset up into three clusters: the blob on the left, the blob on the right, and the central area 
(even though it's less dense than the blobs on either side).
Scenario 2: Break the dataset up into two clusters: the blob on the left, and the blob on the right. Marking all the 
points in the center as noise.
'''

#following grid plots the DBSCAN clustering results of a range of parameter values
eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]
helper.plot_dbscan_grid(dataset_2, eps_values, min_samples_values)

'''
Epsilon=1.3, min_samples=5 seems to do a good job here. There are other similar ones as well (1,2), for example.
Epsilon=1.3, min_samples=20 does the best job to satisfy scenario 2
'''
