"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

from networks import openSetClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import random
import sys

# Define the argument parser
parser = argparse.ArgumentParser(description='Open Set Classifier Prediction')
parser.add_argument('--dataset', default="Materials", type=str, help='Dataset for prediction', choices=['Materials'])
parser.add_argument('--model', required=True, type=int, help='Classifier selection (0: Model-A, 1:Model-B)', choices=[0, 1])
parser.add_argument('--name', default='', type=str, help='Name of training script')
parser.add_argument('--material', required=True, type=str, help='Material under test', choices=['mut1'])
parser.add_argument('--none_thickness', action='store_true', help='Thickness is unknown')
args = parser.parse_args()

# Set global variables
FREQUENCY_RANGE = (126e9, 182e9)
FREQUENCY_COUNT = 256
THICKNESS_RANGE = (0.5e-3, 5e-3)
THICKNESS_COUNT = 451
PERMITTIVITY_RANGE = (2, 5)
PERMITTIVITY_COUNT = 301

# Load material under test (MUT) configurations
with open('measurements/config_measurements.json') as config_file:
    mut_config = json.load(config_file)[args.material]

# Load test samples and reference data
test_samples = sio.loadmat(f"measurements/MUT/generated/{mut_config['radar']}")
reference_data = sio.loadmat(f"measurements/MUT/{mut_config['vna']}")
material_string = mut_config["material_string"]
filename_fig="figures/{}".format(mut_config["filename_fig"])

# Format test samples
test_samples = test_samples['s_crf_cal_mut'][0, :5120:20]
test_samples = test_samples.reshape(1, 1, -1, 1)  # Complex
test_samples = test_samples.astype(np.complex64)

# Load dataset configurations
with open('datasets/config.json') as config_file:
    dataset_config = json.load(config_file)[args.dataset]

# Determine input thickness
input_thickness = mut_config["thickness"]
if args.none_thickness:
    input_thickness = None

# Get the selected model number
model_num = args.model

# Determine the processing device (CUDA if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize accuracy and auroc lists
all_accuracy = []
all_auroc = []

# Load thickness-clusters matrix
thickness_values = sio.loadmat('datasets/utils_generated/d_matrix_451_1000clusters.mat')['d_v']

# Load the class split corresponding to the selected model
num_classes = dataset_config['num_known_classes'][model_num]
with open(f"datasets/{args.dataset}/class_splits/{model_num}.json") as f:
    class_splits = json.load(f)
    known_classes = class_splits['Known']
    thickness_values = thickness_values[:, known_classes]

# Load labels and K-clusters
labels = sio.loadmat('datasets/utils_generated/labels_135751.mat')['labels']
clusters = sio.loadmat('datasets/utils_generated/clusters_k1000_d2_n.mat')['FrameStack_k1000'].reshape(-1, 1)

# Set up thickness for prediction
thickness_axis = np.around(np.linspace(*THICKNESS_RANGE, THICKNESS_COUNT), decimals=5)
frequency_axis = np.linspace(*reversed(FREQUENCY_RANGE), FREQUENCY_COUNT)  # Down-chirp

if input_thickness is None:
    thickness_vector = np.ones((1, num_classes), dtype=np.int8)  # Default
else:
    input_thickness = np.around(input_thickness * 1e-3, decimals=5)
    thickness_index = np.where(thickness_axis == input_thickness)[0].astype(int)
    thickness_vector = thickness_values[thickness_index].astype(np.byte)

# Set up tan_loss scaler
tan_loss_scaler = MinMaxScaler()
tan_loss_scaler.fit(np.array([0.0001, 0.01]).reshape(-1, 1))

###############################Open Set Network Prediction###############################
# Build the network
print('==> Building network')

network = openSetClassifier.openSetClassifier(num_classes, dataset_config['sig_channels'],
                                              dataset_config['sig_length'], dropout=dataset_config['dropout'])

# Load network weights
checkpoint = torch.load(f'networks/weights/{args.dataset}/{args.dataset}_{model_num}_{args.name}CACclassifierAccuracy.pth')

network = network.to(device)
network_dict = network.state_dict()
pretrained_dict = {key: value for key, value in checkpoint['net'].items() if key in network_dict}

# Handle anchor points
if 'anchors' not in pretrained_dict.keys():
    pretrained_dict['anchors'] = checkpoint['net']['means']

network.load_state_dict(pretrained_dict)
network.eval()

# Prepare data and prediction tools
softmax = torch.nn.Softmax(dim=1)
test_samples_tensor = torch.tensor(test_samples).cuda().type(torch.complex64)

# Make predictions
outputs = network(test_samples_tensor)
logits, distances, tan_loss = outputs
_, primary_prediction = torch.min(distances, 1)

# Find top 3 predictions
predictions = [torch.kthvalue(distances, rank).indices for rank in range(1, 4)]
predicted_classes = [known_classes[int(prediction)] for prediction in predictions]

# Calculate scores
soft_min_distances = softmax(-distances)
inverse_scores = 1 - soft_min_distances
scores = distances * inverse_scores
scores_array = scores.cpu().detach().numpy()

# Normalize the scores
mean_score, std_score = np.mean(scores_array), np.std(scores_array)
z_scores = (scores_array - mean_score) / std_score
print('min z_score: ', z_scores.min())

# Adjust scores based on thickness
adjusted_scores = scores_array * thickness_vector
thickness_indices = [i[0] for i, x in np.ndenumerate(thickness_vector.flatten()) if x == 1]
adjusted_scores = adjusted_scores[:, thickness_indices]

# Calculate the loss tangent
loss_tangent = tan_loss_scaler.inverse_transform(tan_loss.cpu().detach().numpy())
print('loss tangent: ', loss_tangent)

# Gather thickness-permittivity values for each prediction
thickness_permittivity_values = [labels[predicted_value] for predicted_value in clusters[predicted_classes[0]][0]]
thickness_permittivity_values_array = np.array(thickness_permittivity_values).reshape((-1, 2))

# Define material model
def materialModel(freq, thickness, rel_permittivity, loss_tangent):
        c0 = 299792458
        epsilon_r1 = 1 - 0j
        epsilon_r2 = rel_permittivity * (1 - 1j * loss_tangent)
        R12 = (np.sqrt(epsilon_r1) - np.sqrt(epsilon_r2)) / (np.sqrt(epsilon_r1) + np.sqrt(epsilon_r2))
        gamma = 1j * 2 * np.pi * freq * np.sqrt(epsilon_r2) / c0
        R = ((1 - np.exp(-2 * gamma * thickness)) * R12) / (1 - R12**2 * np.exp(-2 * gamma * thickness))
        T = ((1 - R12**2) * np.exp(-gamma * thickness)) / (1 - R12**2 * np.exp(-2 * gamma * thickness))

        return {'R': R, 'T': T}

# Apply material model to predict thickness and permittivity
if input_thickness is None:
    rand_thickness, rand_permittivity = random.choice(thickness_permittivity_values_array)
    X_k = materialModel(frequency_axis, rand_thickness, rand_permittivity, loss_tangent)['R']
    X_t = materialModel(frequency_axis, rand_thickness, rand_permittivity, loss_tangent)['T']
else:
    permittivity_values_for_thickness = thickness_permittivity_values_array[np.argwhere(np.around(thickness_permittivity_values_array[:,0], decimals=5) == input_thickness), 1]
    if permittivity_values_for_thickness.size==0:
        print("Error: No predictions could be generated for the provided input thickness.")
        sys.exit()
    mse_eps = []
    for permittivity_value in permittivity_values_for_thickness:
        X_k = materialModel(frequency_axis, input_thickness, permittivity_value, loss_tangent)['R']
        mse_eps.append(mean_squared_error(np.abs(X_k.flatten()), np.abs(test_samples.flatten())))
    predicted_permittivity = permittivity_values_for_thickness[np.argmin(mse_eps)]
    X_k = materialModel(frequency_axis, input_thickness, predicted_permittivity, loss_tangent)['R']
    X_t = materialModel(frequency_axis, input_thickness, predicted_permittivity, loss_tangent)['T']
    print('eps_r: ', predicted_permittivity)

#
X_meas_pred_sample = test_samples.flatten()
predicted_S11_sample = X_k.flatten()
predicted_S21_sample = X_t.flatten()

############### Plotting
# Create plot for visualizing results
fig = plt.figure(figsize=(6,6), dpi=120)

# Plot Magnitude
magnitude_subplot = plt.subplot(2,1,1)

# Plot reference S11 (VNA) data
magnitude_subplot.plot(reference_data["frequency"].flatten()/1e9, 20*np.log10(np.abs(reference_data["S"][:,0].flatten())),
                       label="S11 VNA", color='#bcbd22', linestyle='solid')

# Plot predicted S11 data
magnitude_subplot.plot(frequency_axis/1e9, 20*np.log10(abs(predicted_S11_sample)),
                       label="S11 Radar+NN", color='tab:red', linestyle=(0, (5, 1)))

# Plot reference S21 (VNA) data
magnitude_subplot.plot(reference_data["frequency"].flatten()/1e9, 20*np.log10(np.abs(reference_data["S"][:,1].flatten())),
                       label="S21 VNA", color='tab:orange', linestyle=(0, (4,1,1,1,1,1)))

# Plot predicted S21 data
magnitude_subplot.plot(frequency_axis/1e9, 20*np.log10(abs(predicted_S21_sample)),
                       label="S21 Radar+NN", color='#1f77b4', linestyle=(0, (5,1,1,1)))

# Add labels, legend, and format the subplot
magnitude_subplot.axvline(x=170, color='black', linewidth=2)
magnitude_subplot.text(x=170+1, y=-20, s="MCK band limit", rotation=90, verticalalignment='center')
magnitude_subplot.set(xlim=[110, 182], ylim=[-50, 2], title="Measured S-Parameter \nMaterial: "+ material_string,
                      xlabel="Frequency (GHz)", ylabel="Magnitude (dB)")
magnitude_subplot.grid(True)
magnitude_subplot.legend(loc=3, prop={'size': 8})
plt.tight_layout()

# Plot Phase
phase_subplot = plt.subplot(2,1,2, sharex=magnitude_subplot)

# Calculate phase for reference S11 (VNA) data
refl_phase_vna = reference_data["S"][:,0].flatten() * np.exp(1j*np.pi)

# Plot phase for reference S11 (VNA) data
phase_subplot.plot(reference_data["frequency"].flatten()/1e9, np.rad2deg(np.angle(refl_phase_vna)),
                   label="S11 VNA", color='#bcbd22', linestyle='solid')

# Calculate phase for predicted S11 and S21 data
reflection_phase = np.angle(predicted_S11_sample)
transmission_phase = np.angle(predicted_S21_sample)

# Plot phases for predicted S11 and S21 data
phase_subplot.plot(frequency_axis/1e9, np.rad2deg(reflection_phase), label="S11 Radar+NN", color='tab:red', linestyle=(0, (5, 1)))
phase_subplot.plot(reference_data["frequency"].flatten()/1e9, np.rad2deg(np.angle(reference_data["S"][:,1].flatten())),
                   label="S21 VNA", color='tab:orange', linestyle=(0, (4,1,1,1,1,1)))
phase_subplot.plot(frequency_axis/1e9, np.rad2deg(transmission_phase), label="S21 Radar+NN", color='#1f77b4', linestyle=(0, (5,1,1,1)))

# Add labels, legend, and format the subplot
phase_subplot.axvline(x=170, color='black', linewidth=2)
phase_subplot.text(x=170+1, y=0, s="MCK band limit", rotation=90, verticalalignment='center')
phase_subplot.set(xlabel="Frequency (GHz)", ylabel='Phase (deg)', ylim=[-180, 180])
phase_subplot.grid(True)
phase_subplot.legend(loc=3, prop={'size': 8})

# Show the final plot
plt.tight_layout()
plt.show()

# Save the plot
fig.subplots_adjust(hspace=0.24)
fig.savefig(filename_fig, format='jpg')
