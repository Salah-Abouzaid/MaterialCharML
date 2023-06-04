"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""
###### IMPORTANT:
# This script uses K=1000 for the target clusters, and K=600 as helper for splitting the classes.

import json
import random
import scipy.io as sio
import numpy as np
import sys
random.seed(1000)

def save_class_split(dataset, trial, classes):
    file_path = f"{dataset}/class_splits/{trial}.json"
    with open(file_path, 'w') as file:
        json.dump(classes, file)
    print(f"Saved {dataset} trial {trial} to {file_path}")

# Load labels
labels = sio.loadmat('utils_generated/labels_135751.mat')['labels']

# Round the first column to 5 decimals and the second column to 2 decimals
labels[:, 0] = np.around(labels[:, 0], decimals=5)
labels[:, 1] = np.around(labels[:, 1], decimals=2)

# Load K-clusters (larger cluster elements)
clusters_data = sio.loadmat('utils_generated/clusters_k600_d2_n.mat')
clusters = clusters_data['FrameStack_k600'].reshape(-1, 1)

reference_value = np.array([2e-3, 2])

# Search clusters for a specific reference value
matched_cluster_values = None
for cluster_id in range(clusters.size):
    cluster_values = [labels[i] for i in clusters[cluster_id][0]]
    cluster_values_array = np.array(cluster_values).reshape((-1, 2))
    if np.any(np.all(cluster_values_array == reference_value, axis=1)):
        print(cluster_id)
        matched_cluster_values = cluster_values_array
if matched_cluster_values is None:
    print("Error: Condition not met.")
    sys.exit()

# Function to find nearest value and its index in an array
def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return array[index], index

# Load target K-clusters
target_clusters_data = sio.loadmat('utils_generated/clusters_k1000_d2_n.mat')
target_clusters = target_clusters_data['FrameStack_k1000'].reshape(-1, 1)

known_classes = []
unknown_classes = []

# Classify clusters based on comparison with reference values
for cluster_id in range(1000):
    cluster_values = [labels[i] for i in target_clusters[cluster_id][0]]
    cluster_values_array = np.array(cluster_values).reshape((-1, 2))

    initial_thickness = cluster_values_array[0][0]
    initial_permittivity = np.round(cluster_values_array[0][1], 2)

    nearest_permittivity, nearest_index = find_nearest(np.round(matched_cluster_values[:, 1], 2),
                                                       value=initial_permittivity)
    reference_thickness = matched_cluster_values[nearest_index, 0]

    # Classify as known if initial thickness is greater or equal to reference thickness
    if initial_thickness >= reference_thickness:
        known_classes.append(cluster_id)
    else:
        unknown_classes.append(cluster_id)

#####
print('Num of known classes for trial 0: ',np.size(known_classes))
print('Num of unknown classes for trial 0: ',np.size(unknown_classes))
print('Attention!! update num_known_classes in the config file.')

save_class_split('Materials', 0, {'Known': known_classes, 'Unknown': unknown_classes})
save_class_split('Materials', 1, {'Known': unknown_classes, 'Unknown': known_classes})