"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import numpy as np
import scipy.io as io
from scipy import signal

# Load MUT data
data = io.loadmat("MUT/radar_PE300_4900um.mat")
# MUT saving name
MUTfileName ="MUT/generated/calibrated_radar_PE300_4900um.mat"

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
matchSet = io.loadmat("radar_match.mat")
match = matchSet["data"].flatten()
shortSet = io.loadmat("radar_short.mat")
short = shortSet["data"].flatten()

mut = data["data"].flatten()

#Calibrating MUT and SHORT
# Step 1: Subtract the MATCH and scale by 2**15/5 for ADC Bit to voltage conversion
calibration_data = np.asarray([short-match, mut-match]).transpose()/2**16*5

# Step 2: Compensate waveguide dispersion
kCut = np.pi / waveguideSize  # cutoff wavemnumber of the feeding WR10
kFS = 2 * np.pi * frequency / 3e8  # free space wavenumber
k = np.sqrt(kFS ** 2 - kCut ** 2)  # wavenumber within the setup

# Transfer function of the dispersive waveguide (25cm including 16cm feeding)
HWG = np.exp(1j * k * reflectionWaveguideLength)

# Step 3: Apply windowing using Hamming/Hann window
window = signal.windows.hamming(freq_points)
calibration_data = calibration_data * np.tile(HWG, (2, 1)).T * np.tile(window, (2, 1)).T

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
echoes[valid_range_not, :] = echoes[valid_range_not, :] * np.tile(window_hann, (2, 1)).T

# Step 6: Transform back to IF signal domain (implicit Hilbert transform)
radar_response = np.fft.ifft(echoes, axis=0)
radar_response = radar_response[:freq_points, :]

# Step 7: Normalize MUT measurement by short and save calibrated radar signal
calibrated_radar_signal = -radar_response[:, 1] / radar_response[:, 0]

# Save data
calibrated_data_dict = {"s_crf_cal_mut": calibrated_radar_signal[:5120]}
io.savemat(MUTfileName, calibrated_data_dict)