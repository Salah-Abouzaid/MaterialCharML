"""
	A large portion of the code was originally sourced from:
	https://github.com/dimitymiller/cac-openset#class-anchor-clustering-a-distance-based-loss-for-training-open-set-classifiers

	If you use this code in your research, please consider citing our paper:
    Deep Learning-Based Material Characterization Using FMCW Radar With Open-Set Recognition Technique
    https://doi.org/10.1109/TMTT.2023.3276053
	Salah Abouzaid, 2023
"""

import argparse
import json
import torch
from sklearn.preprocessing import MinMaxScaler
from networks import openSetClassifier
import datasets.utils as dataHelper
from utils import find_anchor_means, gather_outputs
import metrics
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Open Set Classifier Evaluation')
parser.add_argument('--dataset', default="Materials", type=str, help='Dataset for prediction', choices=['Materials'])
parser.add_argument('--model', required=True, type=int, help='Classifier selection (0: Model-A, 1:Model-B)', choices=[0, 1])
parser.add_argument('--name', default='', type=str, help='Name of training script')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_num = args.model

print('==> Preparing data for model {}..'.format(model_num))
with open('datasets/config.json') as config_file:
    cfg = json.load(config_file)[args.dataset]

#Create dataloaders for evaluation
knownloader, unknownloader, mapping = dataHelper.get_eval_loaders(args.dataset, model_num, cfg)

###############################Open Set Network Evaluation###############################

print('==> Building open set network for model {}..'.format(model_num))
net = openSetClassifier.openSetClassifier(cfg['num_known_classes'][model_num], cfg['sig_channels'], cfg['sig_length'], dropout = cfg['dropout'])
checkpoint = torch.load('networks/weights/{}/{}_{}_{}CACclassifierTotalLoss.pth'.format(args.dataset, args.dataset, model_num, args.name))

net = net.to(device)
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
if 'anchors' not in pretrained_dict.keys():
    pretrained_dict['anchors'] = checkpoint['net']['means']
net.load_state_dict(pretrained_dict)
net.eval()

#find mean anchors for each class. [[Check this]]: Commented?
anchor_means = find_anchor_means(net, mapping, args.dataset, model_num, cfg, only_correct = True)
anchor_means = np.array(anchor_means)
net.set_anchors(torch.Tensor(anchor_means))

print('==> Evaluating open set network accuracy for model {}..'.format(model_num))
x, y, y2_pred, y2_tg = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)

tan_loss_scaler = MinMaxScaler()
tan_loss_scaler.fit(np.array([0.0001, 0.01]).reshape(-1, 1))
y2_pred = tan_loss_scaler.inverse_transform(y2_pred)
y2_tg = tan_loss_scaler.inverse_transform(y2_tg)
MSE = metrics.MeanSquaredError(y2_pred, y2_tg) # returns MAE

accuracy = metrics.accuracy(x, y)

print('accuracy: ',accuracy)
print('MAE: ', MSE)


print('==> Evaluating open set network AUROC for model {}..'.format(model_num))
xK, yK, y2_predK, y2_tgK = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)
xU, yU, y2_predU, y2_tgU = gather_outputs(net, mapping, unknownloader, data_idx = 1, calculate_scores = True, unknown = True)

auroc = metrics.auroc(xK, xU)

print('AUROC: ',auroc)

fpr, tpr, thresholds = metrics.roc(xK, xU)

fig = plt.figure(figsize=(6,5), dpi=120)
lw = 2
plt.plot(
    fpr,
    tpr,
    color="#1f77b4",
    lw=lw,
    label="ROC curve (AUROC = {mean_auroc:2.2f} %)".format(mean_auroc=auroc*100),
)
plt.plot([0, 1], [0, 1], color="tab:red", lw=lw, linestyle="--", label = "random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right", prop={'size': 8})
plt.tight_layout()
plt.show()


