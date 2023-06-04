"""
    Functions useful for creating experiment datasets and dataloaders.
    Dimity Miller, 2020
    modified by Salah Abouzaid, 2023
"""

import torch
import json
import random
random.seed(1000)
from datasets.MaterialsDataset import MaterialsDataset

def get_train_loaders(datasetName, trial_num, cfg):
    """
        Create training dataloaders.
        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file
        returns trainloader, evalloader, testloader, mapping - changes labels from original to known class label
    """
    trainSet, valSet, testSet, _ = load_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
        trainValIdxs = json.load(f)
        train_idxs = trainValIdxs['Train']
        val_idxs = trainValIdxs['Val']

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']

    trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
    valSubset = create_dataSubsets(valSet, known_classes, val_idxs)
    testSubset = create_dataSubsets(testSet, known_classes)

    #create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, cfg['num_classes'])

    batch_size = cfg['batch_size']

    trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True, num_workers = cfg['dataloader_workers'])
    valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader, mapping


def get_eval_loaders(datasetName, trial_num, cfg):
    """
        Create evaluation dataloaders.
        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file
        returns knownloader, unknownloader, mapping - changes labels from original to known class label
    """
    _, _, testSet, _ = load_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']
        unknown_classes = class_splits['Unknown']

    testSubset = create_dataSubsets(testSet, known_classes)

    unknownSubset = create_dataSubsets(testSet, unknown_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, cfg['num_classes'])

    batch_size = cfg['batch_size']
    knownloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=False)
    unknownloader = torch.utils.data.DataLoader(unknownSubset, batch_size=batch_size, shuffle=False)

    return knownloader, unknownloader, mapping

def load_datasets(datasetName, cfg, trial_num):
    """
        Load all datasets for training/evaluation.
        datasetName: name of dataset
        cfg: config file
        trial_num: trial number dictating known/unknown class split
        returns trainset, valset, knownset, unknownset
    """

    unknownSet = None

    if datasetName == "Materials":
        trainSet = MaterialsDataset('datasets/data/Materials/train/', split = "train")
        valSet = MaterialsDataset('datasets/data/Materials/train/', split = "train")
        testSet = MaterialsDataset('datasets/data/Materials/val/', split = "test")
    else:
        print("Sorry, that dataset has not been implemented.")
        exit()

    return trainSet, valSet, testSet, unknownSet


def create_dataSubsets(dataset, classes_to_use, idxs_to_use=None):
    """
        Returns dataset subset that satisfies class and idx restraints.
        dataset: torchvision dataset
        classes_to_use: classes that are allowed in the subset (known vs unknown)
        idxs_to_use: image indexes that are allowed in the subset (train vs val, not relevant for test)
        returns torch Subset
    """
    import torch

    # get class label for dataset.
    targets = dataset.labels

    subset_idxs = []
    if idxs_to_use == None:
        for i, lbl in enumerate(targets):
            if lbl in classes_to_use:
                subset_idxs += [i]
    else:
        for class_num in idxs_to_use.keys():
            if int(class_num) in classes_to_use:
                subset_idxs += idxs_to_use[class_num]

    dataSubset = torch.utils.data.Subset(dataset, subset_idxs)
    return dataSubset


def create_target_map(known_classes, num_classes):
    """
        Creates a mapping from original dataset labels to new 'known class' training label
        known_classes: classes that will be trained with
        num_classes: number of classes the dataset typically has

        returns mapping - a dictionary where mapping[original_class_label] = known_class_label
    """
    mapping = [None for i in range(num_classes)]
    known_classes.sort()
    for i, num in enumerate(known_classes):
        mapping[num] = i
    return mapping

############################
def get_anchor_loaders(datasetName, trial_num, cfg):
	"""
		Supply trainloaders for calculating anchor class centres.
		datasetName: name of dataset
		trial_num: trial number dictating known/unknown class split
		cfg: config file
		returns trainloader
	"""
	trainSet= load_anchor_datasets(datasetName, cfg, trial_num)

	with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']
	trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size= cfg['batch_size'])

	return trainloader

def load_anchor_datasets(datasetName, cfg, trial_num):
	"""
		Load train datasets for calculating anchor class centres.
		datasetName: name of dataset
		cfg: config file
		trial_num: trial number dictating known/unknown class split
		returns trainset
	"""
	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']

	if datasetName == "Materials":
		trainSet = MaterialsDataset('datasets/data/Materials/train/', split = "train")
	else:
		print("Sorry, that dataset has not been implemented.")
		exit()

	return trainSet