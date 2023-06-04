"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import numpy as np
import scipy.io as io

# Constants
FREQ_MIN = 126e9
FREQ_MAX = 182e9
NUM_FREQ = 256
THICKNESS_RANGE = np.arange(0.5, 5.01, 0.01) * 1e-3
PERMITTIVITY_RANGE = np.arange(2, 5.01, 0.01)
LOSS_TANGENT = 0.0001

# Radar frequency array
radar_freq = np.linspace(FREQ_MAX, FREQ_MIN, NUM_FREQ)  # Down-chirp

# Create a meshgrid for thickness and permittivity ranges
thicknesses, permittivity = np.meshgrid(THICKNESS_RANGE, PERMITTIVITY_RANGE)
material_params = np.vstack((thicknesses.flatten(), permittivity.flatten()))
total_samples = material_params.shape[1]

# Function to calculate reflection coefficient
def materialModel(freq, thickness, rel_permittivity, loss_tangent):
    c0 = 299792458
    epsilon_r1 = 1 - 0j
    epsilon_r2 = rel_permittivity * (1 - 1j * loss_tangent)
    R12 = (np.sqrt(epsilon_r1) - np.sqrt(epsilon_r2)) / (np.sqrt(epsilon_r1) + np.sqrt(epsilon_r2))
    gamma = 1j * 2 * np.pi * freq * np.sqrt(epsilon_r2) / c0
    R = ((1 - np.exp(-2 * gamma * thickness)) * R12) / (1 - R12 ** 2 * np.exp(-2 * gamma * thickness))
    T = ((1 - R12 ** 2) * np.exp(-gamma * thickness)) / (1 - R12 ** 2 * np.exp(-2 * gamma * thickness))
    return {'R': R, 'T': T}

# Initialize arrays
calibrated_s_if = np.zeros((total_samples, NUM_FREQ))
labels = np.zeros((total_samples, 2))

# Calculate s_if for each set of material parameters
for sample in range(total_samples):
    Thickness = material_params[0, sample]
    permittivity = material_params[1, sample]
    reflection_coefficient = materialModel(radar_freq, thickness=Thickness, rel_permittivity=permittivity, loss_tangent=LOSS_TANGENT)['R']
    calibrated_s_if[sample, :] = np.abs(reflection_coefficient)
    labels[sample, :] = [Thickness, permittivity]

# Save results
io.savemat("utils_generated/Dataset_eps_d_ndgrid_135751.mat", {"s_if_cal": calibrated_s_if, "labels": labels})
io.savemat("utils_generated/labels_135751.mat", {"labels": labels})