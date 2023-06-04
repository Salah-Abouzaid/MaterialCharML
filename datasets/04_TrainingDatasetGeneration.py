"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import numpy as np
import scipy.io as io
from scipy import signal
import os

# Frequency settings
freq_min = 126e9
freq_max = 182e9
freq_points = 5121
frequency = np.linspace(freq_max, freq_min, freq_points) # Down-chirp

# Waveguide and reflection settings
waveguideSize = 2.54e-3 / 2
reflectionWaveguideLength = 25e-2 / 4
reflectionRange = (0.12, 0.32)
caliperThickness = 3030e-6
FFT_size = 2 ** 16

# Calibration configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
short_path = os.path.join(current_dir, '..', 'measurements', 'radar_short.mat')
match_path = os.path.join(current_dir, '..', 'measurements', 'radar_match.mat')

matchSet = io.loadmat(match_path)
match = matchSet["data"].flatten()
shortSet = io.loadmat(short_path)
short = shortSet["data"].flatten()

#Calibrating MUT and SHORT
# Step 1: Subtract the MATCH and scale by 2**15/5 for ADC Bit to voltage conversion
calibration_data = (short-match).reshape(-1, 1) / 2**16 * 5

# Step 2: Compensate waveguide dispersion
kCut = np.pi / waveguideSize  # cutoff wavemnumber of the feeding WR10
kFS = 2 * np.pi * frequency / 3e8  # free space wavenumber
k = np.sqrt(kFS ** 2 - kCut ** 2)  # wavenumber within the setup

# Transfer function of the dispersive waveguide (25cm including 16cm feeding)
HWG = np.exp(1j * k * reflectionWaveguideLength)

# Step 3: Apply windowing using Hamming/Hann window
window = signal.windows.hamming(freq_points)
calibration_data = calibration_data.flatten() * HWG * window

# Step 4: Transform to radar echo domain
df = np.abs(np.mean(np.diff(frequency)))  # Frequency resolution
Rmax = 1 / df * 3e8  # Maximum unambiguous roundtrip distance
distance = np.arange(FFT_size) / FFT_size * Rmax
echoes = np.fft.fft(calibration_data, FFT_size, axis=0) / freq_points  # radar echoes

# Step 5: Remove signal contributions not belonging to the material reflections
valid_range = np.logical_or(distance > reflectionRange[1], distance < reflectionRange[0])
echoes[valid_range] = 0
valid_range_not = np.logical_not(valid_range)
window_hann = signal.windows.hann(valid_range_not.sum())
echoes[valid_range_not] = echoes[valid_range_not] * window_hann

# Step 6: Transform back to IF signal domain (implicit Hilbert transform)
radar_response = np.fft.ifft(echoes, axis=0)

# Step 7: Crop to original range and save as calibrated short data
calibrated_short_data = radar_response[:freq_points]

########################
# Load labels and cluster data
labels_clusters = io.loadmat('utils_generated/labels_135751.mat')['labels']
cluster_data = io.loadmat('utils_generated/clusters_k1000_d2_n.mat')['FrameStack_k1000'].reshape(-1,1)

######################## Dataset Generation
# Initialize constants for dataset generation
N_clusters = 1000
samples_per_cluster = 100
total_samples = N_clusters * samples_per_cluster
complexReflectionCoef_uncalib = np.empty((total_samples, 256), dtype=np.complex64)

parameters = []
label_clusters = []

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

def select_random_parameters(clusterData, labelsClusters, idx):
    rand_s = np.random.choice(clusterData[idx][0].flatten())
    d = np.around(labelsClusters[rand_s][0], decimals=5)
    eps_r = np.around(labelsClusters[rand_s][1], decimals=2)
    tan_loss_min = 0.0001
    tan_loss_max = 0.01
    tan_loss = np.around(tan_loss_min + (tan_loss_max - tan_loss_min) * np.random.rand(1), 4)[0]
    Ref_Coe = materialModel(frequency, thickness=d, rel_permittivity=eps_r, loss_tangent=tan_loss)['R']

    return [d, eps_r, tan_loss], Ref_Coe

####### Note: noise can be added in different ways
def generate_noise(s_uncalib, Ref_Coe, win, F, NFFT):
    target_snr_db = np.random.randint(30, 60)                                                   ## method 1
    #target_snr_db = np.random.randint(110,140)                                                 ## method 2
    sig_avg_watts = np.mean(abs(Ref_Coe))
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(s_uncalib))
    noise_volts = noise_volts * win
    noise_f = np.fft.fft(noise_volts, NFFT) / F
    return noise_volts, noise_f

def apply_gibbs(Echoes_i_sim_noise, dist, reflection_range):
    vm = np.logical_or(dist > reflection_range[1], dist < reflection_range[0])
    Echoes_i_sim_noise[vm] = 0
    return Echoes_i_sim_noise

for i in range(N_clusters):
    for jj in range(samples_per_cluster):
        params, Ref_coe = select_random_parameters(cluster_data, labels_clusters, i)
        parameters.append(params)
        label_clusters.append(i)

        s_uncal = -Ref_coe * calibrated_short_data
        Rrzp_i_sim = np.pad(s_uncal, (0, 2 ** 16 - 5121), mode='constant')                      ## method 1
        echoes_i_sim = np.fft.fft(Rrzp_i_sim)                                                   ## method 1

        noiseVolts, noiseF = generate_noise(s_uncal, Ref_coe, window, freq_points, FFT_size)
        #s_uncal = s_uncal + noiseVolts                                                         ## method 2
        #Rrzp_i_sim = np.pad(s_uncal, (0, 2**16-5121), mode='constant')                         ## method 2
        #echoes_i_sim = np.fft.fft(Rrzp_i_sim)                                                  ## method 2

        reflectionRangem = (np.random.uniform(0.01, 0.12), np.random.uniform(0.33, 0.37))
        echoes_i_sim_noise = apply_gibbs(noiseF + echoes_i_sim, distance, reflectionRangem)     ## method 1
        #echoes_i_sim_noise = apply_gibbs(echoes_i_sim, distance, reflectionRangem)             ## method 2

        Rrzp_i_sim_gb = np.fft.ifft(echoes_i_sim_noise)
        Rr_i_sim_gb = Rrzp_i_sim_gb[:freq_points]

        s_if_uncal_gb = -Rr_i_sim_gb / calibrated_short_data
        complexReflectionCoef_uncalib[jj + i * samples_per_cluster, :] = s_if_uncal_gb[0:5120:20].astype(np.complex64)

#
label_clusters = np.array(label_clusters).reshape(-1, 1)
parameters = np.array(parameters).reshape(-1, 3)

# Save generated dataset
mdic = {"s_crf_uncal": complexReflectionCoef_uncalib, "labels": label_clusters, "labels2": parameters}
io.savemat("utils_generated/Dataset_Complex_k1000_d2_v1.mat", mdic)