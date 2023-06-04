"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""
# If the thickness is known, one can disable all clusters
# that do not contain the input thickness to ensure that the
# predicted cluster includes the input thickness.

import numpy as np
import scipy.io as io

# Load datasets
dataset = io.loadmat('utils_generated/Dataset_eps_d_ndgrid_135751.mat')
cluster_data = io.loadmat('utils_generated/clusters_k1000_d2_n.mat')

# Extract features and labels from dataset
features = dataset['s_if_cal']
labels = dataset['labels']

# Reshape cluster data
cluster_array = cluster_data['FrameStack_k1000'].reshape(-1, 1)

# Number of clusters
num_clusters = 1000

# Initialize a list to store unique rounded values for each cluster
unique_values_per_cluster = []

# Iterate over clusters
for cluster in range(num_clusters):
    # Extract thickness values for current cluster
    thickness_values = [labels[index, 0] for index in cluster_array[cluster][0]]

    # Append unique rounded thickness values to the list
    unique_values_per_cluster.append(list(np.unique(np.around(thickness_values, decimals=5))))

unique_values_per_cluster = np.array(unique_values_per_cluster, dtype=object)

# Define thickness range and matrix to hold final values
thickness_range = np.around(np.linspace(0.5e-3, 5e-3, 451), decimals=5)
final_values = np.zeros((451, num_clusters), dtype=np.byte)

# Populate final_values matrix
for thickness_index in range(451):
    for cluster_index in range(num_clusters):
        # If thickness is in unique_values for the current cluster, mark it in the final_values matrix
        if thickness_range[thickness_index] in unique_values_per_cluster[cluster_index]:
            final_values[thickness_index, cluster_index] = 1

# Save the final matrix
io.savemat("utils_generated/d_matrix_451_1000clusters.mat", {"d_v": final_values})
