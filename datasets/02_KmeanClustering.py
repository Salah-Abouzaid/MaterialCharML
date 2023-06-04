"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import numpy as np
import scipy.io as io
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# Number of clusters
K = 1000

# Load dataset
dataset = io.loadmat('utils_generated/Dataset_eps_d_ndgrid_135751.mat')
features = dataset['s_if_cal']
labels = dataset['labels']

# Function to normalize rows individually
def normalize_rows(XX):
    num_rows = XX.shape[0]
    for row in range(num_rows):
        x = XX[row, :]
        x_normalized = np.interp(x, (x.min(), x.max()), (0, 1))
        XX[row, :] = x_normalized
    return XX

# Normalize features
features_normalized = normalize_rows(features)

# Train KMeans model
#kmeans_model = KMeans(n_clusters=K, random_state=0)
kmeans_model = MiniBatchKMeans(n_clusters=K, batch_size=1024, n_init='auto')

kmeans_model.fit(features_normalized)
#kmeans_model.fit(features)

# Save model
pickle.dump(kmeans_model, open(f"utils_generated/model_k{K}_135751_n.pkl", "wb"))

# Assign a cluster to each example
predicted_clusters = kmeans_model.predict(features_normalized)

# Retrieve unique clusters
unique_clusters = np.unique(predicted_clusters)

# Get row indexes for each unique cluster
clustered_indexes = []
for cluster in unique_clusters:
    indexes = np.where(predicted_clusters == cluster)
    clustered_indexes.append(indexes)

# Store indexes of each cluster in a dictionary
cluster_dict = {}
for cluster in range(K):
    indexes_in_cluster = []
    for i in clustered_indexes[cluster][0]:
        indexes_in_cluster.append(i)
    cluster_dict[cluster] = np.array(indexes_in_cluster)

# Save indexes in numpy array
cluster_array = np.empty((K,), dtype=object)
for i in range(K):
    cluster_array[i] = cluster_dict[i]

# Save clusters to .mat file
io.savemat(f"utils_generated/clusters_k{K}_d2_n.mat", {f"FrameStack_k{K}": cluster_array})
