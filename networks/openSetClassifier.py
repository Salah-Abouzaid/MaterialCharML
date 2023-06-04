"""
	A large portion of the code was originally sourced from:
	https://github.com/dimitymiller/cac-openset#class-anchor-clustering-a-distance-based-loss-for-training-open-set-classifiers

	If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout2d
from complexPyTorch.complexFunctions import complex_relu


class openSetClassifier(nn.Module):
	def __init__(self, num_classes = 1000, num_channels = 1, sig_length=256, init_weights = False, dropout = 0.3, **kwargs):
		super(openSetClassifier, self).__init__()

		self.num_classes = num_classes
		self.encoder = BaseEncoder(num_channels, dropout)
		
		if sig_length == 128:
			self.classify = ComplexLinear(128*8, num_classes)
			self.regress = ComplexLinear(128 * 8, 1)
		elif sig_length == 256:
			self.classify = ComplexLinear(128*16, num_classes)
			self.regress = ComplexLinear(128 * 16, 1)
		else:
			print('That signal size has not been implemented, sorry.')
			exit()

		self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad = False)

		self.cuda()


	def forward(self, x, skip_distance = False):
		batch_size = len(x)

		x = self.encoder(x)
		x = x.view(batch_size, -1)

		outLinear = self.classify(x)
		outLinear = outLinear.abs()

		outRegress = self.regress(x)
		outRegress = outRegress.abs()

		if skip_distance:
			return outLinear, None, outRegress

		outDistance = self.distance_classifier(outLinear)

		return outLinear, outDistance, outRegress

	def set_anchors(self, means):
		self.anchors = nn.Parameter(means.double(), requires_grad = False)
		self.cuda()

	def distance_classifier(self, x):
		''' Calculates euclidean distance from x to each class anchor
			Returns n x m array of distance from input of batch_size n to anchors of size m
		'''

		n = x.size(0)
		m = self.num_classes
		#'''
		d = self.num_classes
		x = x.unsqueeze(1).expand(n, m, d).double()
		anchors = self.anchors.unsqueeze(0).expand(n, m, d)
		dists = torch.norm(x-anchors, 2, 2)
		#'''
		# larger batch_size but slower:
		'''
		dists = torch.tensor([[torch.norm(x[i] - self.anchors[j], 2) for j in range(m)] for i in range(n)],
							 dtype=torch.float64,
							 device = torch.device('cuda:0'),
							 requires_grad = True
							 )
		'''
		return dists

class BaseEncoder(nn.Module):
	def __init__(self, num_channels, dropout=0.3, **kwargs):
		super().__init__()
		# self.dropout = ComplexDropout2d(dropout)

		self.bn1 = ComplexBatchNorm2d(64)
		self.bn2 = ComplexBatchNorm2d(64)
		self.bn3 = ComplexBatchNorm2d(128)

		self.conv1 = ComplexConv2d(num_channels, 64, (8, 1), (2, 1), (3, 0), bias=False)
		self.conv2 = ComplexConv2d(64, 64, (8, 1), (2, 1), (3, 0), bias=False)
		self.conv3 = ComplexConv2d(64, 128, (8, 1), (4, 1), (3, 0), bias=False)

		self.cuda()

	def forward(self, x):
		x1 = self.conv1(x)
		x1 = self.bn1(x1)
		x1 = complex_relu(x1)
		x1 = self.conv2(x1)
		x1 = self.bn2(x1)
		x1 = complex_relu(x1)
		x1 = self.conv3(x1)
		x1 = self.bn3(x1)
		x1 = complex_relu(x1)
		return x1




