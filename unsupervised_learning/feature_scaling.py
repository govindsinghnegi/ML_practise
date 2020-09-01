import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from unsupervised_learning import tests2 as t

# DSND colors: UBlue, Salmon, Gold, Slate
plot_colors = ['#02b3e4', '#ee2e76', '#ffb613', '#2e3d49']

# Light colors: Blue light, Salmon light
plot_lcolors = ['#88d0f3', '#ed8ca1', '#fdd270']

# Gray/bg colors: Slate Dark, Gray, Silver
plot_grays = ['#1c262f', '#aebfd1', '#fafbfc']


def create_data():
    n_points = 120
    X = np.random.RandomState(3200000).uniform(-3, 3, [n_points, 2])
    X_abs = np.absolute(X)

    inner_ring_flag = np.logical_and(X_abs[:, 0] < 1.2, X_abs[:, 1] < 1.2)
    outer_ring_flag = X_abs.sum(axis=1) > 5.3
    keep = np.logical_not(np.logical_or(inner_ring_flag, outer_ring_flag))

    X = X[keep]
    X = X[:60]  # only keep first 100
    X1 = np.matmul(X, np.array([[2.5, 0], [0, 100]])) + np.array([22.5, 500])

    plt.figure(figsize=[15, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=64, c=plot_colors[-1])

    plt.xlabel('5k Completion Time (min)', size=30)
    plt.xticks(np.arange(15, 30 + 5, 5), fontsize=30)
    plt.ylabel('Test Score (raw)', size=30)
    plt.yticks(np.arange(200, 800 + 200, 200), fontsize=30)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width=2)
    plt.savefig('C18_FeatScalingEx_01.png', transparent=True)
    plt.show()

    data = pd.DataFrame(X1)
    data.columns = ['5k_Time', 'Raw_Test_Score']

    return data


data = create_data()
data.info()
data.describe()

# Use the dictionary to match the values to the corresponding statements
a = 0
b = 60
c = 22.9
d = 4.53
e = 511.7

q1_dict = {
'number of missing values': a,
'the mean 5k time in minutes': c,
'the mean test score as a raw value': e,
'number of individuals in the dataset': b
}

# check your answer against ours here
t.check_q1(q1_dict)

n_clusters = 2
model = KMeans(n_clusters = n_clusters)
preds = model.fit_predict(data)


def plot_clusters(data, preds, n_clusters):
    plt.figure(figsize=[15, 15])

    for k, col in zip(range(n_clusters), plot_colors[:n_clusters]):
        my_members = (preds == k)
        plt.scatter(data['5k_Time'][my_members], data['Raw_Test_Score'][my_members], s=64, c=col)

    plt.xlabel('5k Completion Time (min)', size=30)
    plt.xticks(np.arange(15, 30 + 5, 5), fontsize=30)
    plt.ylabel('Test Score (raw)', size=30)
    plt.yticks(np.arange(200, 800 + 200, 200), fontsize=30)
    plt.show()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width=2)


plot_clusters(data, preds, 2)

# scaling
# test_scaled-> subtracting the mean test score and dividing by the standard deviation test score
data['test_scaled'] = (data['Raw_Test_Score'] - np.mean(data['Raw_Test_Score']))/np.std(data['Raw_Test_Score'])
data['5k_time_sec'] = data['5k_Time'] * 60

n_clusters = 2
model = KMeans(n_clusters = n_clusters)
preds = model.fit_predict(data)

plot_clusters(data, preds, n_clusters)