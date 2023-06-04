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
import torch.optim as optim
import json
import argparse
import datasets.utils as dataHelper
from networks import openSetClassifier
from utils import progress_bar
import os
import numpy as np

parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--dataset', default="Materials", type=str, help='Dataset for prediction', choices=['Materials'])
parser.add_argument('--model', required=True, type=int, help='Classifier selection (0: Model-A, 1:Model-B)', choices=[0, 1])
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--alpha', default = 10, type = int, help='Magnitude of the anchor point')
parser.add_argument('--lbda', default = 0.1, type = float, help='Weighting of Anchor loss component')
parser.add_argument('--lbda2', default = 0.5, type = float, help='Weighting of MSE loss component')
parser.add_argument('--name', default='', type=str, help='Name of training script')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters useful when resuming and finetuning
best_acc = 0
best_mse = 10000
best_tot_loss = 10000
best_cac = 10000
start_epoch = 0

#Create dataloaders for training
print('==> Preparing data..')
with open('datasets/config.json') as config_file:
	cfg = json.load(config_file)[args.dataset]

trainloader, valloader, _, mapping = dataHelper.get_train_loaders(args.dataset, args.model, cfg)

###############################Open Set Network training###############################

print('==> Building network..')
net = openSetClassifier.openSetClassifier(cfg['num_known_classes'][args.model], cfg['sig_channels'], cfg['sig_length'],
										init_weights = not args.resume, dropout = cfg['dropout'])

# initialising with anchors
anchors = torch.diag(torch.Tensor([args.alpha for i in range(cfg['num_known_classes'][args.model])]))
net.set_anchors(anchors)

net = net.to(device)
training_iter = int(args.resume)

if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('networks/weights'), 'Error: no checkpoint directory found!'
	
	checkpoint = torch.load('networks/weights/{}/{}_{}_{}CACclassifierTotalLoss.pth'.format(args.dataset, args.dataset, args.model, args.name))

	start_epoch = checkpoint['epoch']

	net_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
	net.load_state_dict(pretrained_dict)

net.train()
optimizer = optim.SGD(net.parameters(), lr = cfg['openset_training']['learning_rate'][training_iter],
							momentum = 0.9, weight_decay = cfg['openset_training']['weight_decay'])
#optimizer = optim.Adam(net.parameters(), lr=cfg['openset_training']['learning_rate'][training_iter], weight_decay = cfg['openset_training']['weight_decay'])

loss_mse = nn.L1Loss() #used mean absolute error (MAE)

def CACLoss(distances, ground_truth):
	"""
    Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualization.

    Args:
        distances (torch.Tensor): The distances matrix.
        ground_truth (torch.Tensor): Ground truth labels.

    Returns:
        tuple: The total loss, anchor loss, and tuplet loss.
    """

	# Get the distances corresponding to the ground truth labels.
	true_distances = torch.gather(distances, 1, ground_truth.view(-1, 1)).view(-1)

	# Create a tensor where each row corresponds to the non-ground truth classes for each instance.
	non_gt_classes = torch.Tensor([[i for i in range(cfg['num_known_classes'][args.model])
									if ground_truth[x] != i] for x in range(len(distances))]).long().cuda()

	# Gather the distances for the non-ground truth classes.
	non_gt_distances = torch.gather(distances, 1, non_gt_classes)

	# Select the top 10 smallest distances.
	smallest_distances = torch.topk(non_gt_distances, k=10, dim=1, largest=False).values

	# Calculate the mean true distance (the "anchor" loss).
	anchor_loss = torch.mean(true_distances)

	# Calculate the tuplet loss.
	tuplet_loss = torch.exp(-smallest_distances + true_distances.unsqueeze(1))
	tuplet_loss = torch.mean(torch.log(1 + torch.sum(tuplet_loss, dim=1)))

	# Calculate the total loss.
	total_loss = args.lbda * anchor_loss + tuplet_loss

	return total_loss, anchor_loss, tuplet_loss


# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correctDist = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets0, targets1 = inputs.to(device).type(torch.complex64), targets[0].to(device), targets[1].to(device)
		#convert from original dataset label to known class label
		targets0 = torch.Tensor([mapping[x] for x in targets0]).long().to(device)

		optimizer.zero_grad()

		outputs = net(inputs) # openSetClassifier returns: outLinear, outDistance
		cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets0) # outputs[1]: outDistance
		loss2 = loss_mse(outputs[2], targets1)
		loss_total = cacLoss + args.lbda2 * loss2

		loss_total.backward()

		optimizer.step()

		train_loss += loss_total.item()

		_, predicted = outputs[1].min(1)

		total += targets0.size(0)
		correctDist += predicted.eq(targets0).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %3.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correctDist/total, correctDist, total))

def val(epoch):
	global best_acc
	global best_mse
	global best_tot_loss
	global best_cac
	net.eval()
	mse_loss = 0
	cac_loss = 0
	tot_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			inputs = inputs.to(device).type(torch.complex64)
			targets0 = torch.Tensor([mapping[x] for x in targets[0]]).long().to(device)
			targets1 = targets[1].to(device)

			outputs = net(inputs)

			cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets0)
			loss2 = loss_mse(outputs[2], targets1)
			loss3 = cacLoss + args.lbda2 * loss2

			mse_loss += loss2
			cac_loss += cacLoss
			tot_loss += loss3

			_, predicted = outputs[1].min(1)
			
			total += targets0.size(0)

			correct += predicted.eq(targets0).sum().item()

			progress_bar(batch_idx, len(valloader), 'MSE Loss: %3.3f | Acc: %.3f%% (%d/%d)'
						 % (mse_loss / (batch_idx + 1), 100. * correct / total, correct, total))
   
	mse_loss /= len(valloader)
	cac_loss /= len(valloader)
	tot_loss /= len(valloader)
	acc = 100.*correct/total

	# Save checkpoint.
	state = {
		'net': net.state_dict(),
		'acc': acc,
		'mse': mse_loss,
		'tot_loss': tot_loss,
		'epoch': epoch,
	}

	if not os.path.isdir('networks/weights/{}'.format(args.dataset)):
		os.mkdir('networks/weights/{}'.format(args.dataset))

	save_name = '{}_{}_{}CACclassifier'.format(args.dataset, args.model, args.name)

	if tot_loss <= best_tot_loss:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(args.dataset)+save_name+'TotalLoss.pth')
		best_tot_loss = tot_loss

	if acc >= best_acc:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(args.dataset)+save_name+'Accuracy.pth')
		best_acc = acc

if __name__ == '__main__':
	max_epoch = cfg['openset_training']['max_epoch'][training_iter]+start_epoch
	for epoch in range(start_epoch, max_epoch):
		train(epoch)
		val(epoch)

