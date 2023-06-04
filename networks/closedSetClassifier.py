"""
	A large portion of the code was originally sourced from:
	https://github.com/dimitymiller/cac-openset#class-anchor-clustering-a-distance-based-loss-for-training-open-set-classifiers

	If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout2d
from complexPyTorch.complexFunctions import complex_relu

class closedSetClassifier(nn.Module):
    def __init__(self, num_classes=1000, num_channels=1, sig_length=256, init_weights=False, dropout=0.3, **kwargs):
        super(closedSetClassifier, self).__init__()

        self.num_classes = num_classes
        self.encoder = BaseEncoder(num_channels, dropout)

        if sig_length == 128:
            self.classify = ComplexLinear(128 * 8, num_classes)
            self.regress = ComplexLinear(128 * 8, 1)
        elif sig_length == 256:
            self.classify = ComplexLinear(128 * 16, num_classes)
            self.regress = ComplexLinear(128 * 16, 1)
        else:
            print('That signal size has not been implemented, sorry.')
            exit()

        self.cuda()

    def forward(self, x):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)
        outLinear = outLinear.abs()

        outRegress = self.regress(x)
        outRegress = outRegress.abs()

        return outLinear, outRegress

class BaseEncoder(nn.Module):
    def __init__(self, num_channels, dropout=0.3, **kwargs):
        super().__init__()
        #self.dropout = ComplexDropout2d(dropout)

        self.bn1 = ComplexBatchNorm2d(64)
        self.bn2 = ComplexBatchNorm2d(64)
        self.bn3 = ComplexBatchNorm2d(128)

        self.conv1 = ComplexConv2d(num_channels, 64, (8,1), (2,1), (3,0), bias=False)
        self.conv2 = ComplexConv2d(64, 64, (8,1), (2,1), (3,0), bias=False)
        self.conv3 = ComplexConv2d(64, 128, (8,1), (4,1), (3,0), bias=False)

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