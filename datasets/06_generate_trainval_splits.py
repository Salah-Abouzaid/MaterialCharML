"""
	Randomly select train and validation subsets from training datasets.
	Dimity Miller, 2020
	modified by Salah Abouzaid, 2023
"""

import json
import random
import numpy as np
from MaterialsDataset import MaterialsDataset
random.seed(1000)

def save_trainval_split(dataset, train_idxs, val_idxs):
	print("Saving {} Train/Val split to {}/trainval_idxs.json".format(dataset, dataset))
	file = open('{}/trainval_idxs.json'.format(dataset), 'w')
	file.write(json.dumps({'Train': train_idxs, 'Val': val_idxs}))
	file.close()

MaterialsLoaded = MaterialsDataset('data/Materials/train/')

datasets = {'Materials': MaterialsLoaded}
split = {'Materials': 0.75}

for datasetName in datasets.keys():
	dataset = datasets[datasetName]
	#get class label for each signal.

	targets = dataset.labels
	num_classes = len(np.unique(targets))

	#save signal idxs per class
	class_idxs = [[] for i in range(num_classes)]
	for i, lbl in enumerate(targets):
		class_idxs[lbl] += [i]

	#determine size of train subset
	class_size = [len(x) for x in class_idxs]
	class_train_size = [int(split[datasetName]*x) for x in class_size]

	#subset per class into train and val subsets randomly
	train_idxs = {}
	val_idxs = {}
	for class_num in range(num_classes):
		train_size = class_train_size[class_num]
		idxs = class_idxs[class_num]
		random.shuffle(idxs)
		train_idxs[class_num] = idxs[:train_size]
		val_idxs[class_num] = idxs[train_size:]

	save_trainval_split(datasetName, train_idxs, val_idxs)