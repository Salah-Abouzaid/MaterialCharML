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
from networks import closedSetClassifier
import datasets.utils as dataHelper
import metrics
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Closed Set Classifier Evaluation')
parser.add_argument('--dataset', default="Materials", type=str, help='Dataset for prediction', choices=['Materials'])
parser.add_argument('--start_trial', required=True, type=int, help='Classifier selection (0: Model-A, 1:Model-B)', choices=[0, 1])
parser.add_argument('--name', default='', type=str, help='Name of training script')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trial_num = args.start_trial

print('==> Preparing data for trial {}..'.format(trial_num))
with open('datasets/config.json') as config_file:
    cfg = json.load(config_file)[args.dataset]

# Create dataloaders for evaluation
knownloader, unknownloader, mapping = dataHelper.get_eval_loaders(args.dataset, trial_num, cfg)

###############################Closed Set Network Evaluation###############################
print('==> Building closed set network for trial {}..'.format(trial_num))
net = closedSetClassifier.closedSetClassifier(cfg['num_known_classes'][trial_num], cfg['sig_channels'], cfg['sig_length'],
                                              dropout=cfg['dropout'])
checkpoint = torch.load(
    'networks/weights/{}/{}_{}_{}closedSetClassifierTotalLoss.pth'.format(args.dataset, args.dataset, trial_num, args.name))

net = net.to(device)
net.load_state_dict(checkpoint['net'])
net.eval()

X = []
y = []
y2_pred = []
y2_tg = []

softmax = torch.nn.Softmax(dim=1)
for i, data in enumerate(knownloader):
    signals, labels = data
    targets = torch.Tensor([mapping[x] for x in labels[0]]).long().cuda()
    tanloss_tg = labels[1]

    signals = signals.cuda().type(torch.complex64)
    logits, tanloss = net(signals)
    scores = softmax(logits)

    X += scores.cpu().detach().tolist()
    y += targets.cpu().tolist()
    y2_pred += tanloss.cpu().tolist()
    y2_tg += tanloss_tg.cpu().tolist()

X = -np.asarray(X)
y = np.asarray(y)
y2_pred = np.asarray(y2_pred)
y2_tg = np.asarray(y2_tg)

tan_loss_scaler = MinMaxScaler()
tan_loss_scaler.fit(np.array([0.0001, 0.01]).reshape(-1, 1))
y2_pred = tan_loss_scaler.inverse_transform(y2_pred)
y2_tg = tan_loss_scaler.inverse_transform(y2_tg)
MSE = metrics.MeanSquaredError(y2_pred, y2_tg) # returns MAE

accuracy = metrics.accuracy(X, y)

print('accuracy: ',accuracy)
print('MAE: ', MSE)

XU = []
for i, data in enumerate(unknownloader):
    signals, labels = data

    signals = signals.cuda().type(torch.complex64)
    logits, tanloss = net(signals)
    scores = softmax(logits)
    XU += scores.cpu().detach().tolist()

XU = -np.asarray(XU)
auroc = metrics.auroc(X, XU)

print('AUROC: ', auroc)

fpr, tpr, thresholds = metrics.roc(X, XU)

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