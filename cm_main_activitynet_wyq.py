# -*- coding: utf-8 -*-
import io
import os
import os.path
import time
import random
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from cm_build_dataset_wyq import build_dataset
from cm_utils import *
import itertools
from torch import Tensor
import math
# load environmental settings
import cm_opts_wyq

#增加mm21论文当中的模块

opt = cm_opts_wyq.opt_algorithm()

if opt.stage==1 and opt.img_net=='vilt':
    from vilt_model.model_s3d_wyq import ViLTransformerSS, VilT_Classification
elif opt.stage==2 and opt.img_net=='vilt':
    from vilt_model.model_vgg_wyq import ViLTransformerSS, VilT_Classification
elif opt.stage==3 and opt.img_net=='vilt':
    from vilt_model.model_full_wyq import ViLTransformerSS, VilT_Classification
# elif opt.stage==4 and opt.img_net=='vilt':
#     from vilt_model.model_causal_vilt import ViLTransformerSS, VilT_Classification
    
elif opt.stage==1 and opt.img_net=='gru':
    from vilt_model.model_s3d_gru import gru_encoder_t,VilT_Classification #取相同的名字但是用的是GRU网络
elif opt.stage==2 and opt.img_net=='gru':
    from vilt_model.model_vgg_gru import gru_encoder_t,VilT_Classification
elif opt.stage==3 and opt.img_net=='gru':
    from vilt_model.model_full_gru import gru_encoder_t,VilT_Classification



# -----------------------------------------------------------------dataset information--------------------------------------------------------------------

opt.dataset = 'activitynet'
opt.root_path = './activitynet_train_test/'  # path to root folder
opt.data_path ='./activitynet_train_test/'
opt.img_path = '/vireo172/ready_chinese_food'  # path to image folder



# --------------------------------------------------------------------settings----------------------------------------------------------------------------

# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1  #随机数种子
measure_best = 0  # best measurement
torch.manual_seed(SEED)
kwargs = {'num_workers': 5, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    
# log and model paths
result_path = os.path.join(opt.result_path, para_name(opt))

if not os.path.exists(result_path):
    os.makedirs(result_path)

# train settings
mode = opt.mode

EPOCHS = opt.lr_decay * 3 + 1


# -------------------------------------------------------------dataset & dataloader-----------------------------------------------------------------------

transform_img_train=None
transform_img_test=None  


# create dataset
# ipdb.set_trace()
dataset_train=build_dataset(opt.stage,opt.img_path,opt.data_path,transform_img_train,mode,opt.dataset, 'train')
dataset_test=build_dataset(opt.stage,opt.img_path,opt.data_path,transform_img_test,mode,opt.dataset, 'test')

# dataloader


# ipdb.set_trace()
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=200, **kwargs)


# -----------------------------------------------------------------Model/optimizer------------------------------------------------------------------------

def get_updateModel(model, path):
#     ipdb.set_trace()
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(shared_dict)

    model.load_state_dict(model_dict)

    return model




from vilt_model.config import get_config

# #ipdb.set_trace()
vilt_config = get_config()
model = VilT_Classification(vilt_config)
model=model.cuda()




if opt.stage in [1,2,3,4] and opt.img_net=='vilt':

     optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr_m1)

elif opt.stage in [1,2,3,4] and opt.img_net=='gru':
     optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay, lr=opt.lr_m1)
    #ipdb.set_trace()
#     optimizer_m1=optim.Adam(model.module1.parameters(),weight_decay=opt.weight_decay, lr=opt.lr_m1)
#     optimizer_m2=optim.Adam(model.module2.parameters(),weight_decay=opt.weight_decay, lr=opt.lr_m1)
#     optimizer=[optimizer_m1,optimizer_m2]

#------------------------------------------------------------对比学习-------------------------------------------
from torch import Tensor
import math

#增加对比学习模块
#加入对比学习损失
class NTXentLoss(nn.Module):#不加标签信息的loss
    def __init__(self, temperature=0.1, eps=1e-6):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.dense1=nn.Linear(60,1).cuda()
        self.dense2=nn.Linear(128,768).cuda()
        
    def forward(self, vgg_features, pos_vgg_features):

        
        vgg_features=self.dense2(vgg_features)
#         print(vgg_features)
        pos_vgg_features=self.dense2(pos_vgg_features)
#         print(pos_vgg_features)
        
        vgg_features = vgg_features.transpose(1,2)
        vgg_features=torch.squeeze(self.dense1(vgg_features))
        vgg_features = nn.functional.normalize(vgg_features, dim=-1)
        
        pos_vgg_features=pos_vgg_features.transpose(1,2)
        pos_vgg_features=torch.squeeze(self.dense1(pos_vgg_features))
        pos_vgg_features = nn.functional.normalize(pos_vgg_features, dim=-1)
        
        out_1=vgg_features
        out_2=pos_vgg_features
              
                
        out = torch.cat([out_1, out_2], dim=0)

        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature).sum(dim=-1)
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)

        neg = torch.clamp(neg - row_sub, min=self.eps)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + self.eps)).mean()




# ----------------------------------------------------------------Train----------------------------------------------------------------------------------


def train_epoch(epoch, decay, optimizer, train_stage, train_log):
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    losses = AverageMeter('Loss', ':.4e')
    lossAB = AverageMeter('LossAB', ':.4e')
    losses_con = AverageMeter('Loss', ':.4e')
    losses_cls1 = AverageMeter('Loss', ':.4e')
    losses_cls2 = AverageMeter('Loss', ':.4e')

    model.train()
    total_time = time.time()
    #ipdb.set_trace()
    
    #  n
# s 
# c 一直往下执行，直到遇到ipdb、断点
# b 17 在17行设置断点，然后c，运行到17行就停下
# r 一直运行到return
    
    
    for batch_idx, (data,label) in enumerate(train_loader):
        print("- - -- - - - - - -batch_idx : {}".format(batch_idx))
        start_time = time.time()
        
        # load data
        #msrvtt data 
        if train_stage==1:
            batch_size_cur=data.size(0)
            if CUDA:
                s3d_features=data.cuda()
                label=label.cuda()
        elif train_stage==2:
            batch_size_cur=data.size(0)
            if CUDA:
                vgg_features=data.cuda()
                label=label.cuda()
    
        elif train_stage==3:
            batch_size_cur=data[0].size(0)
            if CUDA:
                [s3d_features,vgg_features]=data


                s3d_features=s3d_features.cuda()
                vgg_features=vgg_features.cuda()
                label=label.cuda()

        elif train_stage==4:
            batch_size_cur = data[0][0].size(0)*2 #当前是32

            if CUDA:
                #取出A的正样本为B
                s3d_features=data[0][0]
                vgg_features=data[0][1]
                index=data[0][2]
                label_A=label[0]
                
                pos_s3d_features=data[1][0]
                pos_vgg_features=data[1][1]
                pos_index=data[1][2]
                pos_label_A=label[1]
                
                s3d_features=s3d_features.cuda()
                vgg_features=vgg_features.cuda()
                label_A=label_A.cuda()
                
                pos_s3d_features=pos_s3d_features.cuda()
                pos_vgg_features=pos_vgg_features.cuda()
                pos_label_A=pos_label_A.cuda()
                
                label=torch.cat([label_A,pos_label_A],dim=0)
                label=label.cuda()
                
                
        else:
            assert 1 < 0, 'Please fill train_stage!'

        # prediction and loss
        
        if train_stage==1:
            #ipdb.set_trace()
            output=model(s3d_features)
            criterion = nn.CrossEntropyLoss()
            loss_cls = criterion(output, label)
            final_loss = loss_cls
        elif train_stage==2:
            output=model(vgg_features)
            criterion = nn.CrossEntropyLoss()
            loss_cls = criterion(output, label)
            final_loss = loss_cls
        elif train_stage==3:

            output=model(s3d_features,vgg_features)
            criterion = nn.CrossEntropyLoss()
            loss_cls = criterion(output, label) #多模态预测的损失
            final_loss = loss_cls
            
#
        elif train_stage==4:
            
            if opt.contrastive==True:
#                 ipdb.set_trace()
                #增加对比学习 64*60*1024->bs*维度
#                 vgg_features=vgg_features.cuda()
#                 pos_vgg_features=pos_vgg_features.cuda()
                criterion_con = NTXentLoss()
                loss_s2s = criterion_con(vgg_features, pos_vgg_features) #计算语义的对比学习
                image_feats,text_feats,output,lossAB = model(s3d_features,vgg_features,pos_s3d_features,pos_vgg_features,batch_size_cur)
                #输入正样本
                criterion_align_l2 = nn.MSELoss()
                loss_align_l2 = criterion_align_l2(image_feats,text_feats)            
                criterion_align_kl = nn.KLDivLoss()
                image_feats = F.log_softmax(image_feats, dim=1)
                text_feats = F.softmax(text_feats.detach(), dim=1)
                loss_align_kl = criterion_align_kl(image_feats,text_feats)
                loss_align = 0.01*loss_align_l2 + loss_align_kl
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                final_loss = loss_cls+lossAB+loss_s2s+loss_align  #计算整个的loss

               
            else:
                output,lossAB = model(s3d_features,vgg_features,pos_s3d_features,pos_vgg_features,batch_size_cur) #输入正样本
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                final_loss = loss_cls+lossAB  #计算整个的loss
            

        # optimization
        losses.update(final_loss.item(), batch_size_cur)
        # frozen/no frozen
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        # divided /  部分微调
#         [optimizer_m1, optimizer_m2] = optimizer
#         optimizer_m1.zero_grad()
#         optimizer_m2.zero_grad()
#         final_loss.backward()
#         optimizer_m1.step()
#         optimizer_m2.step()

        
        if train_stage in [1,2,3,4]:
            #ipdb.set_trace()
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1.update(acc1[0], batch_size_cur)
            top5.update(acc5[0], batch_size_cur)
            
        if train_stage in [1,2,3,4]:
            if opt.img_net=='gru':
                optimizer_cur = optimizer
            elif opt.img_net=='vilt':
                #frozen /no frozen
                optimizer_cur = optimizer
            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                           'Time {data_time:.3f}\t'
                           'Loss {loss.val:.4f}({loss.avg:.4f})\t'
#                            'LossAB{lossAB:.4f}\t'
                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,
                    top1=top1, top5=top5, lr=optimizer_cur.param_groups[-1]['lr']))  
            train_log.write(log_out + '\n')
            train_log.flush()
#         elif train_stage==3:
#             if opt.img_net=='gru':
#                 optimizer_cur = optimizer
#             elif opt.img_net=='vilt':
#                 #frozen /no frozen
#                 optimizer_cur = optimizer
#             log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
#                            'Time {data_time:.3f}\t'
#                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,
#                     top1=top1, top5=top5, lr=optimizer_cur.param_groups[-1]['lr']))  
            
            # divided  /部分微调
#             log_out = ('Epoch: [{0}][{1}/{2}], lr_m1: {lr_m1:.5f}\t lr_m2: {lr_m2:.5f}\t'
#                             'Time {data_time:.3f}\t'
#                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses, top1=top1, top5=top5, lr_m1=optimizer_m1.param_groups[-1]['lr'],lr_m2=optimizer_m2.param_groups[-1]['lr']))
#             train_log.write(log_out + '\n')
#             train_log.flush()

# ----------------------------------------------------------------Test----------------------------------------------------------------------------------
def test_epoch(epoch, stage, test_log):
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            #             ipdb.set_trace()
            print("- - -- - - - - - -test_batch_idx : {}".format(batch_idx))
            # load data
            if stage==1:
                batch_size_cur = data.size(0)
                if CUDA:
                    s3d_features=data.cuda()
                    label = label.cuda()
                # s3d_features
                output = model(s3d_features)
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)
                losses.update(loss_cls.item(), batch_size_cur)
            elif stage==2:
                batch_size_cur = data.size(0)
                if CUDA:
                    vgg_features=data.cuda()
                    label = label.cuda()
                # s3d_features
                output = model(vgg_features)
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)
                losses.update(loss_cls.item(), batch_size_cur)
            elif stage==3:
                batch_size_cur = data[0].size(0)
                [s3d_features,vgg_features] = data
                if CUDA:

#                     s3d_features=res #随机特征选择的帧
                    s3d_features= s3d_features.cuda()
                    vgg_features = vgg_features.cuda()
                    label = label.cuda()
                # full
                output = model(s3d_features, vgg_features)
#                 image_feats,text_feats,output = model(s3d_features, vgg_features)
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)
                losses.update(loss_cls.item(), batch_size_cur)
            elif stage==4:
                batch_size_cur = data[0][0].size(0)*2 #当前是32
                if CUDA:
                #取出A的正样本为B
                    s3d_features=data[0][0]
                    vgg_features=data[0][1]
                    index=data[0][2]
                    label_A=label[0]

                    pos_s3d_features=data[1][0]
                    pos_vgg_features=data[1][1]
                    pos_index=data[1][2]
                    pos_label_A=label[1]


                    s3d_features=s3d_features.cuda()
                    vgg_features=vgg_features.cuda()
                    label_A=label_A.cuda()

                    pos_s3d_features=pos_s3d_features.cuda()
                    pos_vgg_features=pos_vgg_features.cuda()
                    pos_label_A=pos_label_A.cuda()

                    label=torch.cat([label_A,pos_label_A],dim=0)
                    label=label.cuda()
                    
                criterion_con = NTXentLoss()
                loss_s2s = criterion_con(vgg_features, pos_vgg_features) #计算语义的对比学习
#                 output,lossAB = model(s3d_features,vgg_features,pos_s3d_features,pos_vgg_features,batch_size_cur) #输入正样本
                image_feats,text_feats,output,lossAB = model(s3d_features,vgg_features,pos_s3d_features,pos_vgg_features,batch_size_cur)
                # compute loss
                criterion = nn.CrossEntropyLoss()
                loss_cls = criterion(output, label)
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], batch_size_cur)
                top5.update(acc5[0], batch_size_cur)
                losses.update(loss_cls.item(), batch_size_cur)
#
           
                
        if stage in [1,2,3,4]:
            log_out = (
                    'Epoch: {epoch} Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f} Time {time:.3f}'
                    .format(epoch=epoch, top1=top1, top5=top5, loss=losses, time=round((time.time() - start_time), 4)))
            print(log_out)
            test_log.write(log_out + '\n')
            test_log.flush()
            return top1.avg

def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate


if __name__ == '__main__':
    log_training = open(os.path.join(result_path, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(result_path, 'log_test.csv'), 'w')

    for epoch in range(1, EPOCHS + 1):
        #ipdb.set_trace()
        # frozen/no frozen
        if opt.img_net=='vilt':
            lr_scheduler(epoch, optimizer, opt.lr_decay, opt.lrd_rate)
              # divided
#             lr_scheduler(epoch, optimizer_m1, opt.lr_decay, opt.lrd_rate)
#             lr_scheduler(epoch, optimizer_m2, opt.lr_decay, opt.lrd_rate)
#             optimizer = [optimizer_m1,optimizer_m2]
        else:
            lr_scheduler(epoch, optimizer, opt.lr_decay, opt.lrd_rate)
       

        
        if opt.test_only == False:
            train_epoch(epoch, opt.lr_decay, optimizer, opt.stage, log_training)
        measure_cur = test_epoch(epoch, opt.stage, log_testing)
        # save current model
        if measure_cur > measure_best:
            torch.save(model.state_dict(), result_path + '/model_best.pt')
            measure_best = measure_cur
            torch.save(model.state_dict(), result_path + '/model_{}.pt'.format(epoch))

        if epoch == EPOCHS:
            if opt.test_only == False:
                torch.save(model.state_dict(), result_path + '/model_final.pt')
