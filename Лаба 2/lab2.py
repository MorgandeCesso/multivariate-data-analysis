# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Truncated data for demonstration
data_truncated = pd.DataFrame({
    "X": [
        34.35, 40.81, 34.43, 26.34, 21.86, 20.94, 20.37, 13.67, 50.89, 17.29, 37.33,
        19.88, 41.07, 16.89, 49.03, 27.46, 31.91, 36.66, 33.13, 18.84
    ],
    "Y": [
        33.92, 41.33, 32.54, 31.99, 31.49, 24.46, 37.41, 31.42, 24.2, 43.01, 23.4,
        29.85, 30.85, 33.19, 26.02, 26.09, 25.99, 36.1, 18.91, 31.39
    ]
})

# Clustering with KMeans
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data_truncated['Cluster'] = kmeans.fit_predict(data_truncated[['X', 'Y']])

# Plotting the results
plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = data_truncated[data_truncated['Cluster'] == cluster]
    plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster}')

# Mark centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

plt.title('KMeans Clustering Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()