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


#[64,30,1024->64,30,768]
class model_Linear(nn.Module):
    def __init__(self):
        super(model_Linear, self).__init__()
        self.linear = nn.Linear(1024, 768)

    def forward(self, img):
        return self.linear(img)

    
class VilT_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.module0 = model_Linear()
        self.module1 = gru_encoder_t(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
        self.module2 = classifier(config["hidden_size"],config["se_dim"])
        #self.module2 = classifier(144,768)

    def forward(self, img):
        #ipdb.set_trace()
        #img=self.module0(img) #64,x,768
        embed = self.module1(img)  # img=64,30,768
        pred = self.module2(embed)

        return pred #64*20维度的向量


class classifier(nn.Module):
    def __init__(self, hidden_size,se_dim):
        super().__init__()
        self.dense1 = nn.Linear(se_dim, 1)  #msrvtt 64,x,1 activitynet 64,
#         # 64 768 x->64 768 1  error :1920*768and 30*1
        self.dense2 = nn.Linear(hidden_size, 20)  # 64 1 768->64 1 20个类
        #self.dense1 = nn.Linear(144, 1)  #msrvtt 64,x,1 activitynet 64,
        #self.dense2 = nn.Linear(768, 172)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, embeds):
        #ipdb.set_trace()
        #embeds=64,30,768 
        image_embeds = embeds.transpose(1, 2)  #16,768,160  #16，768，30
        feats = torch.squeeze(self.dense1(image_embeds).transpose(1, 2)) #64*768;#64,1,768
        feats = self.relu(feats) #64*768
        output = self.dense2(feats) #64*20 #预测
        feats = nn.functional.normalize(feats, dim=-1) #64*768 特征
        return output

    
class gru_encoder_t(nn.Module):
    #def __init__(self, CUDA, gloveVector, num_hidden, num_key_ingre=5):
    #def __init__(self, CUDA, gloveVector, num_hidden=768, num_key_ingre=5):
    def __init__(self,config,CUDA=1,num_hidden=768, num_key_ingre=5):
        super(gru_encoder_t, self).__init__()
        
        self.CUDA = CUDA
        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre

        self.gru = nn.GRU(768, self.num_hidden)
        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    def forward(self, y):
        # compute latent vectors
        #ipdb.set_trace()
        encoder_t_embeds = y
        encoder_t_embeds = encoder_t_embeds.permute(1, 0, 2) #64,30,768 ->30,64,768

        # obtain gru output of hidden vectors
        h0_en = Parameter(torch.zeros((1, y.shape[0], self.num_hidden), requires_grad=True)) #(1，batchsize，768)

        if self.CUDA:
            h0_en = h0_en.cuda()
        gru_embed , _ = self.gru(encoder_t_embeds, h0_en) #30,batchsize,768 ;1,batchsize,768  
        #30,batchsize,768
        gru_embed = gru_embed.permute(1, 0, 2)# batchsize,30,768
        
        return gru_embed

        # the last non-zero item
#         pos_embeds = []

#         for i in y.tolist():
#             if 0 in i:
#                 pos_embeds.append(i.index(0)-1)
#             else:
#                 pos_embeds.append(len(i)-1)
                
#         ipdb.set_trace()
#         pos_embeds = np.array(pos_embeds)
#         pos_y = np.array(range(y.shape[0]))
#         att_embed = gru_embed[pos_y, pos_embeds, :] #30,batchsize,768
        
#         att_embed = att_embed.permute(1,0,2) #batchsize,30,768
#         return att_embed  #batchsize，768

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    

