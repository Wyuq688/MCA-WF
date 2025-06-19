import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.multiclass import unique_labels
import ipdb


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)  # batch_size

        _, pred = output.topk(maxk, 1, True, True)  # pred: (batch_size, maxk)
        #_, pred = output[0].topk(maxk, 1, True, True)  # pred: (batch_size, maxk)
        pred = pred.t()  # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def precision_recall_matrix(predicts, labels, n=10):
    # labels: (64, 81)
    # predicts: (64, 81)
    final_pre = torch.zeros([0]).cuda()
    final_recall = torch.zeros([0]).cuda()
    for cur in range(n):
        pre_topk = predicts.topk(cur + 1)[1]
        pre_topk_onehot = torch.zeros(labels.shape).cuda()
        pre_topk_onehot = pre_topk_onehot.scatter_(1, pre_topk, 1)

        hit = torch.sum(pre_topk_onehot * labels, dim=1)
        # precision
        precision = hit / (cur + 1)
        recall = hit * (1. / torch.sum(labels, dim=1))
        final_pre = torch.cat((final_pre, precision.unsqueeze(1)), dim=1)
        final_recall = torch.cat((final_recall, recall.unsqueeze(1)), dim=1)

    return final_pre, final_recall


def precision_recall(predicts, labels, n=10):
    predicts = predicts.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # from top to bottom
    sorted_predicts = (-predicts).argsort()
    top_n_inds = sorted_predicts[:, :n]

    # compute top-n hits for each sample
    hit = np.zeros([len(labels), n])
    for i in range(len(labels)):  # for each sample in [0,batch_size-1]
        # calculate the performance from top-1 to top-n
        for j in range(1, n + 1):  # for each value of n in [1,n]
            for k in range(j):  # for each rank of j
                if labels[i, top_n_inds[i, k]] - 1 == 0:
                    hit[i, j - 1] += 1  # j-1 since hit is 0-indexed
    # compute precision
    denominator = np.arange(n) + 1  # 10
    denominator = np.tile(denominator, [len(labels), 1])
    # get precision
    precision = hit / denominator  # (128,10)

    # compute recall
    # get denominator, the sum of the number of ingre in this recipe

    denominator = np.sum(labels, axis=1)  # (128)

    denominator = np.tile(np.expand_dims(denominator, axis=1), [1, n])  # (128,10)
    # get recall
    recall = hit / denominator  # (128,10)

    return precision, recall


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def para_name(args):
    if args.stage == 1:
        name_para = 'datset={}~stage={}~img_net={}~frozen_blks={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            #args.method,
            args.frozen_blks,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
    elif args.stage == 2:
        name_para = 'datset={}~stage={}~img_net={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            #args.word_net,
            args.img_net,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
        if args.method == 'recon':
            name_para += '~lr_finetune={}~lrd_finetune={}'.format(args.lr_finetune, args.lrd_rate_finetune)
            name_para += '~mseloss'
    elif args.stage ==3:
        name_para = 'datset={}~stage={}~img_net={}~frozen_blks={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            args.frozen_blks,
            #args.method,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
            #args.beta_align,
            #args.type_align
        )
    elif args.stage in [4,5,6,7]:
        name_para = 'datset={}~stage={}~img_net={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            #args.method,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
#         if args.method != 'img2word':
#             name_para += '~beta_loss_word={}'.format(args.beta_loss_word)
#         if args.method == 'clip':
#             name_para += '~lr_finetune={}~lrd_finetune={}'.format(args.lr_finetune, args.lrd_rate_finetune)
#         if args.method == 'img2word':
#             name_para += '~lr_finetune={}~lrd_finetune={}'.format(args.lr_finetune, args.lrd_rate_finetune)
#             name_para += 'MSE-sum'

#     elif args.stage in [4,5]:
#         name_para = 'datset={}~img_net={}~stage={}~method={}~bs={}~lr={}~adj={}~topk={}~soft={}'.format(
#             args.dataset,
#             args.img_net,
#             args.stage,
#             args.method,
#             args.batch_size,
#             args.lr,
#             args.adj,
#             args.topk,
#             args.beta_soft
#         )
#         if 'gcn_multi' in args.adj:
#             name_para += '~time={}~w2={}~w3={}'.format(args.gcn_time, args.gcn_w2, args.gcn_w3)
#         elif 'gcn_class' in args.adj:
#             name_para += '~relation={}~threshold={}'.format(args.gcn_relation, args.gcn_threshold)
#     if args.stage == 6:
#         name_para = 'datset={}~img_net={}~bs={}~lr={}~feature_fusion'.format(
#             args.dataset,
#             args.img_net,
#             args.batch_size,
#             args.lr
#         )
#     if args.dataset == 'wide':
#         name_para += '~pos_weight={}'.format(args.pos_weight)
    return name_para


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)