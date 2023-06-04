"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import scipy.io as io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset = io.loadmat('utils_generated/Dataset_Complex_k1000_d2_v1.mat')

# Extract features and labels from dataset
features = dataset['s_crf_uncal']
label_clusters = dataset['labels']

# Extract specific label attributes: thickness, permittivity, and loss tangent
label_thickness = dataset['labels2'][:, 0].reshape(-1, 1)
label_permittivity = dataset['labels2'][:, 1].reshape(-1, 1)
label_loss_tangent = dataset['labels2'][:, 2].reshape(-1, 1)

# Normalize loss tangent values
scaler = MinMaxScaler()
label_loss_tangent_normalized = scaler.fit_transform(label_loss_tangent)

# Split the dataset into training and test sets
features_train, features_test, labels_train, labels_test, loss_tangent_train, loss_tangent_test = train_test_split(
    features, label_clusters, label_loss_tangent_normalized, test_size=0.2, random_state=20)

# Store training and test sets into dictionaries
training_data = {"X": features_train, "y": labels_train, "y2": loss_tangent_train}
test_data = {"X": features_test, "y": labels_test, "y2": loss_tangent_test}

# Save training and test sets into .mat files
io.savemat("data/Materials/train/train_256x1.mat", training_data)
io.savemat("data/Materials/val/test_256x1.mat", test_data)