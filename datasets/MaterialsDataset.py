"""
    If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset


class MaterialsDataset(Dataset):
    """Materials Dataset class to load the materials data"""

    # List of data files corresponding to each split
    split_data_files = {
        "train": "train_256x1.mat",
        "test": "test_256x1.mat",
    }

    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None):
        """
        Initialize the MaterialsDataset instance

        Args:
            root (str): Path to the directory containing the data files
            split (str): The data split to load. Options are "train" and "test"
            transform: A function/transform to apply to the input data
            target_transform: A function/transform to apply to the target labels
        """
        # Construct the full path to the data file for the given split
        data_file = self.split_data_files[split]
        data_file_path = os.path.join(root, data_file)

        # Load the MAT file
        loaded_mat = sio.loadmat(data_file_path)

        # Extract the data and labels from the loaded MAT file
        self.features = np.expand_dims(loaded_mat["X"], axis=(1, -1))  # shape: (num_samples, 1, length, 1)
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()  # shape: (num_samples,)
        self.labels2 = loaded_mat["y2"].astype(np.float32)  # shape: (num_samples,)

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.features)

    def __getitem__(self, idx: int):
        """
        Fetch the data sample and its corresponding labels at the given index

        Args:
            idx (int): Index of the data sample to fetch

        Returns:
            tuple: (data_sample, [primary_label, secondary_label])
        """
        # Fetch the data sample and labels at the given index
        data_sample = self.features[idx]
        primary_label = int(self.labels[idx])
        secondary_label = self.labels2[idx].astype(np.float32)

        # Apply the transforms if specified
        if self.transform is not None:
            data_sample = self.transform(data_sample)
        if self.target_transform is not None:
            primary_label = self.target_transform(primary_label)

        return data_sample, [primary_label, secondary_label]
