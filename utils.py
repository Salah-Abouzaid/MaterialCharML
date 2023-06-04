"""
    Helper functions for training and evaluation.
    progress_bar and format_time function was taken from https://github.com/kuangliu/pytorch-cifar which mimics xlua.progress
    Dimity Miller, 2020
    modified by Salah Abouzaid, 2023
"""

import sys
import time
import numpy as np
import torch
import datasets.utils as dataHelper

term_width = 84

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def find_anchor_means(net, mapping, datasetName, trial_num, cfg, only_correct = False):
    ''' Tests data and fits a multivariate gaussian to each class' logits.
        Returns means and covariances for each class. '''
    loader = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
    logits, labels, _, _ = gather_outputs(net, mapping, loader, only_correct = only_correct)

    num_classes = cfg['num_known_classes'][trial_num]
    means = [None for i in range(num_classes)]
    for cl in range(num_classes):
        x = logits[labels == cl]
        means[cl] = np.mean(x, axis = 0)
    return means

def SoftmaxTemp(logits, T = 1):
    num = torch.exp(logits/T)
    denom = torch.sum(torch.exp(logits/T), 1).unsqueeze(1)
    return num/denom

def gather_outputs(net, mapping, dataloader, data_idx = 0, calculate_scores = False, unknown = False, only_correct = False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
        only_correct    True to filter for correct classifications as per logits
    '''
    X = []
    y = []
    y2_pred = []
    y2_tg = []

    if calculate_scores:
        softmax = torch.nn.Softmax(dim = 1)

    for i, data in enumerate(dataloader):
        signals, labels = data
        signals = signals.cuda().type(torch.complex64)

        if unknown:
            targets = labels[0]
        else:
            targets = torch.Tensor([mapping[x] for x in labels[0]]).long().cuda()

        tanloss_tg = labels[1]

        outputs = net(signals)
        logits = outputs[0]
        distances = outputs[1]
        tanloss = outputs[2]

        if only_correct:
            if data_idx == 0:
                _, predicted = torch.max(logits, 1)
            else:
                _, predicted = torch.min(distances, 1)
            mask = predicted == targets
            logits = logits[mask]
            distances = distances[mask]
            targets = targets[mask]
            tanloss = tanloss[mask]

        if calculate_scores:
            softmin = softmax(-distances)
            invScores = 1-softmin
            scores = distances*invScores
        else:
            if data_idx == 0:
                scores = logits
            if data_idx == 1:
                scores = distances

        X += scores.cpu().detach().tolist()
        y += targets.cpu().tolist()
        y2_pred += tanloss.cpu().tolist()
        y2_tg += tanloss_tg.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)
    y2_pred = np.asarray(y2_pred)
    y2_tg = np.asarray(y2_tg)

    return X, y, y2_pred, y2_tg