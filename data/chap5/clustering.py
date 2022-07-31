#Import the K-Means Class
from sklearn.cluster import KMeans
# Import the function that enables us to plot clusters
from sklearn.datasets import make_blobs
# Import Spectral Clustering from scikit-learn
from sklearn.cluster import SpectralClustering
#Import the half moon function from scikit-learn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns

# Get points such that they form 3 visually separable clusters
X, y = make_blobs(n_samples=300, centers=3,cluster_std=0.50, random_state=0)

# Plot the points on a scatterplot
plt.scatter(X[:, 0], X[:, 1], s=50)

# Initialize the K-Means object. Set number of clusters to 3,
# centroid initialization as 'random' and maximum iterations to 10
kmeans = KMeans(n_clusters=3, init='random', max_iter=10)
# Compute the K-Means clustering
kmeans.fit(X)
# Predict the classes for every point
y_pred = kmeans.predict(X)
# Plot the data points again but with different colors for different classes
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)
# Get the list of the final centroids
centroids = kmeans.cluster_centers_
# Plot the centroids onto the same scatterplot.
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')


plt.show()

# Define the Spectral Clustering Model
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
# Fit and predict the labels
y_m_sc = model.fit_predict(X)
# Plot the colored clusters as identified by Spectral Clustering
plt.scatter(X[:, 0], X[:, 1], c=y_m_sc, s=50)
#
#
# # List that will hold the sum of square values for different cluster sizes
# ss = []
# # We will compute SS for cluster sizes between 1 and 8.
# for i in range(1,9):
#     # Initialize the KMeans object and call the fit method to compute clusters
#     kmeans = KMeans(n_clusters=i, random_state=0, max_iter=10, init='random').fit(X)
#     # Append the value of SS for a particular iteration into the ss list
#     ss.append(kmeans.inertia_)
#
# # Plot the Elbow Plot of SS v/s K
# sns.pointplot(x=[j for j in range(1,9)], y=ss)
#
plt.show()