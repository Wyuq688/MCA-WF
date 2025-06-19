import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig
from packaging import version
from vilt_model.modules import heads, objectives, vilt_utils
#torch.nn.parameter.Parameter
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch import optim
from cm_utils import *
from torch import Tensor
import math

# #增加复现模型的代码，mm21年
# from vilt_models import 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  
#[64,30,1024->64,30,768]
class model_Linear(nn.Module):
    def __init__(self):
        super(model_Linear, self).__init__()
        self.linear1 = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(128, 768)
        
#     def forward(self, img,text):
#         return self.linear1(img),self.linear2(text)
    def forward(self, img):
        return self.linear1(img)
#不加对比约束
class VilT_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module0 = model_Linear()
        self.module1 = BaseboneModel(config)

        self.module2 = classifier(config["hidden_size"])
#         self.module3 = classifier1(config["hidden_size"])
#         self.module3 = gru_encoder_t_text(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
#         self.module3 = ViLTransformerText(config)

        self.dense1 = nn.Linear(30, 1) #第一个加权操作得到的linear层
        self.dense_t = nn.Linear(60, 1) #S 这里要改动
        self.dense2 = nn.Linear(1, 768)
        self.dense3 = nn.Linear(1536,1)
        self.dense4 = nn.Linear(768,768)
        
        self.sigmoid= nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, s3d_features,text_features):
#=============================================attentionV:基于视觉注意力机制的视觉增强算法============================================================
#---------------------------------------------gru-w1-activitynet---------------------------------------------------------------------------------------
      #attention V ---w1 
#         s3d_features,text_features=self.module0(s3d_features,text_features) #32,30,768
#         raw_img=s3d_features
        
#         context_vector=torch.mean(s3d_features,dim=1,keepdim=True) #取特征的平均值也即context vector 64*1*768
#         context_vector=context_vector.transpose(1, 2) #64,768,1
#         weight=torch.bmm(s3d_features,context_vector) #64,30*1
#         weight=self.dense2(weight)  #bs*30*768 可以去学到一个最好的权重
#         enhence_img=raw_img*weight   #视觉增强的视频帧
#         x=torch.cat([enhence_img,text_features],dim=1)
#         x=self.module1(x)
#         pred = self.module2(x)
#         return pred  
#---------------------------------------gru-（w2+sigmoid)-msrvtt-----------------------------------------------------------------------------------
        s3d_features=self.module0(s3d_features) #32,30,768
        raw_img=s3d_features
#         ipdb.set_trace()
        context_vector=torch.mean(s3d_features,dim=1,keepdim=True)#30*768
        context_vector=context_vector.repeat(1,30,1)
        mixcat=torch.cat([s3d_features,context_vector],dim=2)
        weight=self.dense3(mixcat)
        weight=self.sigmoid(weight)
        weight=torch.softmax(weight,1)
        enhence_img=raw_img*weight
        x=torch.cat([enhence_img,text_features],dim=1)
        x=self.module1(x)
        pred = self.module2(x)
        return pred



class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(103, 1)  #msrvtt 64,x,1 activitynet 64,
        self.dense2 = nn.Linear(768,20)  # 64 1 768->64 1 20个
        self.relu = nn.LeakyReLU()
        self.dense1_i = nn.Linear(60, 1)
        self.dense1_t = nn.Linear(60, 1)
        
    def forward(self, embeds):
#         ipdb.set_trace()
        image_embeds = embeds.transpose(1, 2)  #16,768,160  #16，768，30
        feats = torch.squeeze(self.dense1(image_embeds).transpose(1, 2)) #64*768;#64,1,768
        feats = self.relu(feats) #64*768
        output = self.dense2(feats) #64*20 #预测
        return output
    
#GRU_model
class BaseboneModel(nn.Module):
    def __init__(self,config,CUDA=1,num_hidden=768, num_key_ingre=5):
        super(BaseboneModel, self).__init__()
        
        self.CUDA = CUDA
        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre

        self.gru = nn.GRU(768, self.num_hidden)
        self.relu = nn.LeakyReLU()
        

    def forward(self,img):
        # compute latent vectors
        #ipdb.set_trace()
        img_t_embeds = img.permute(1, 0, 2) #64,30,768 ->30,64,768
        # obtain gru output of hidden vectors
        h0_en = Parameter(torch.zeros((1, img.shape[0], self.num_hidden), requires_grad=True)) #(1，batchsize，768)
        if self.CUDA:
            h0_en = h0_en.cuda()    
        gru_embed_img , _ = self.gru(img_t_embeds, h0_en) #30,batchsize,768 ;1,batchsize,768  
        x = gru_embed_img.permute(1, 0, 2)# batchsize,30,768
        
        return x
    

