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
from utils import progress_bar
from networks import closedSetClassifier
import os

parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default="Materials", type=str, help='Dataset for prediction', choices=['Materials'])
parser.add_argument('--lbda2', default = 5.0, type = float, help='Weighting of MAE loss component')
parser.add_argument('--model', required=True, type=int, help='Classifier selection (0: Model-A, 1:Model-B)', choices=[0, 1])
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--name', default='', type=str, help='Name of training script')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters useful when resuming and finetuning
best_acc = 0
best_mse = 10000
best_tot_loss = 10000
start_epoch = 0

# Create dataloader for training
print('==> Preparing data..')
with open('datasets/config.json') as config_file:
    cfg = json.load(config_file)[args.dataset]

trainloader, valloader, _, mapping = dataHelper.get_train_loaders(args.dataset, args.model, cfg)

###############################Closed Set Network training###############################
print('==> Building network..')
net = closedSetClassifier.closedSetClassifier(cfg['num_known_classes'][args.model], cfg['sig_channels'], cfg['sig_length'],
                                              init_weights=not args.resume, dropout=cfg['dropout'])
net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('networks/weights/{}'.format(args.dataset)), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(
        'networks/weights/{}/{}_{}_{}closedSetClassifierTotalLoss.pth'.format(args.dataset, args.dataset, args.model, args.name))

    best_acc = checkpoint['acc']
    best_mse = checkpoint['mse']
    best_tot_loss = checkpoint['tot_loss']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.L1Loss()

training_iter = int(args.resume)
#optimizer = optim.SGD(net.parameters(), lr=cfg['closedset_training']['learning_rate'][training_iter],
#                      momentum=0.9, weight_decay=cfg['closedset_training']['weight_decay'])
optimizer = optim.Adam(net.parameters(), lr=cfg['closedset_training']['learning_rate'][training_iter])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets[0], targets[1] = inputs.to(device).type(torch.complex64), targets[0].to(device), targets[1].to(device)
        targets[0] = torch.Tensor([mapping[x] for x in targets[0]]).long().to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss1 = loss_ce(outputs[0], targets[0])
        loss2= loss_mse(outputs[1], targets[1])

        loss_total = loss1 + args.lbda2 * loss2

        loss_total.backward()

        optimizer.step()

        train_loss += loss_total.item()
        _, predicted = outputs[0].max(1)

        total += targets[0].size(0)
        correct += predicted.eq(targets[0]).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Tot Loss: %3.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def val(epoch):
    global best_acc
    global best_mse
    global best_tot_loss
    net.eval()
    mse_loss = 0
    tot_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = inputs.to(device).type(torch.complex64)
            targets[0] = torch.Tensor([mapping[x] for x in targets[0]]).long().to(device)
            targets[1] = targets[1].to(device)

            outputs = net(inputs)

            _, predicted = outputs[0].max(1)

            loss1 = loss_ce(outputs[0], targets[0])
            loss2 = loss_mse(outputs[1], targets[1])

            mse_loss += loss2
            tot_loss += loss1 + args.lbda2 * loss2

            total += targets[0].size(0)

            correct += predicted.eq(targets[0]).sum().item()

            progress_bar(batch_idx, len(valloader), 'MSE Loss: %3.3f | Acc: %.3f%% (%d/%d)'
                         % (mse_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    mse_loss /= len(valloader)
    tot_loss /= len(valloader)

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
    save_name = '{}_{}_{}closedSetClassifier'.format(args.dataset, args.model, args.name)

    if tot_loss <= best_tot_loss:
        print('Saving..')
        torch.save(state, 'networks/weights/{}/'.format(args.dataset) + save_name + 'TotalLoss.pth')
        best_tot_loss = tot_loss

    if acc >= best_acc:
        print('Saving..')
        torch.save(state, 'networks/weights/{}/'.format(args.dataset) + save_name + 'Accuracy.pth')
        best_acc = acc

if __name__ == '__main__':
    max_epoch = cfg['closedset_training']['max_epoch'][training_iter] + start_epoch
    for epoch in range(start_epoch, start_epoch + max_epoch):
        train(epoch)
        val(epoch)