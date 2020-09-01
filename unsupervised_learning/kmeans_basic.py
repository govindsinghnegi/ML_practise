import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from unsupervised_learning import helpers2 as h
from unsupervised_learning import tests as t

# Make the images larger
plt.rcParams['figure.figsize'] = (16, 9)

data = h.simulate_data(200, 5, 4)

# This will check that your dataset appears to match ours before moving forward
t.test_question_1(data)

k_value = 4

# Check your solution against ours.
t.test_question_2(k_value)

# Try instantiating a model with 4 centers
kmeans_4 = KMeans(n_clusters=4)

# Then fit the model to your data using the fit method
model_4 = kmeans_4.fit(data)

# Finally predict the labels on the same data to show the category that point belongs to
labels_4 = model_4.predict(data)

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_4)

#################

# Try instantiating a model with 2 centers
kmeans_2 = KMeans(n_clusters=2)

# Then fit the model to your data using the fit method
model_2 = kmeans_2.fit(data)

# Finally predict the labels on the same data to show the category that point belongs to
labels_2 = model_2.predict(data)

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_2)

##################

# Try instantiating a model with 7 centers
kmeans_7 = KMeans(n_clusters=7)

# Then fit the model to your data using the fit method
model_7 = kmeans_7.fit(data)

# Finally predict the labels on the same data to show the category that point belongs to
labels_7 = model_7.predict(data)

# If you did all of that correctly, this should provide a plot of your data colored by center
h.plot_data(data, labels_7)


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score


scores = []
centers = list(range(1, 11))

for center in centers:
    scores.append(get_kmeans_score(data, center))

plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');
plt.show()