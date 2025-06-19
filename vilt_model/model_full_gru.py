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
        self.linear1 = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(128, 768)
        
    def forward(self, img,text):
        return self.linear1(img),self.linear2(text)
#     def forward(self, img):
#         return self.linear1(img)

    
class VilT_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module0 = model_Linear()
        self.module1 = gru_encoder_t(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
        self.module2 = classifier(config["hidden_size"])
        self.module3 = gru_encoder_t_text(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
#         self.dense1 = nn.Linear(73, 1) #S
        self.dense1 = nn.Linear(30, 1) #V
        self.dense2 = nn.Linear(1, 768)

    def forward(self, img, text):
#         ipdb.set_trace()
        img,text=self.module0(img,text) #64,60,768
#         img=self.module0(img)
        raw_img=img
        #method_1
#         text_embed=self.module3(text)#64,73,768
#         text_embed=text_embed.transpose(1, 2) #转置 64，768，73
#         text_embed=self.dense1(text_embed) #64，768，1
#         context_vector=torch.bmm(img,text_embed) #64,30,1
#         context_vector=self.dense2(context_vector) #64,30,768
        
#         att_img=raw_img*context_vector
        
        #img=self.module0(img) #64,x,768
        
       #method_2 
        img_vector=torch.mean(img,dim=1,keepdim=True) #取特征的平均值也即context vector 64*1*768
        img_vector=img_vector.transpose(1, 2) #64,768,1
        context_vector=torch.bmm(raw_img,img_vector) #64,30*1
        context_vector=self.dense2(context_vector) #context vector 64*seq*768
        att_img=raw_img*context_vector
        
        embed_full = torch.cat([att_img,text],dim=1)  #前者特征融合1early fusion
        #embed = self.module1(img,text)  # img=64,30,768  后者特征融合
        embed = self.module1(embed_full)
        pred = self.module2(embed)
        return pred
        
        #KLosll
#         ipdb.set_trace()
#         embed_full = torch.cat([att_img,text],dim=1)  #前者特征融合1early fusion
#         image_embeds_jh,text_embeds_jh,embeds = self.module1(embed_full)
#         image_feats,text_feats,output = self.module2(image_embeds_jh,text_embeds_jh,embeds)
#         return image_feats,text_feats,output #64*20维度的向量


class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(120, 1)  #msrvtt 64,x,1 activitynet 64,
#         # 64 768 x->64 768 1  error :1920*768and 30*1
        self.dense1_i = nn.Linear(60, 1)
        self.dense1_t = nn.Linear(60, 1)
        self.dense2 = nn.Linear(hidden_size, 200)  # 64 1 768->64 1 20个
#         self.dense2 = nn.Linear(256, 200)  # 64 1 768->64 1 20个
#         self.dense3 = nn.Linear(256, 128) 
        self.relu = nn.LeakyReLU()

#     def forward(self, image_embeds_jh,text_embeds_jh,embeds):
    def forward(self,embeds):
#         ipdb.set_trace()
        #embeds=64,30,768 
        image_embeds = embeds.transpose(1, 2)  #16,768,160  #16，768，30
        feats = torch.squeeze(self.dense1(image_embeds).transpose(1, 2)) #64*768;#64,1,768
        feats = self.relu(feats) #64*768
        output = self.dense2(feats) #64*20 #预测
        return output
#         

#         image_embeds_jh =image_embeds_jh.transpose(1,2)#64*768*73
#         text_embeds_jh = text_embeds_jh.transpose(1,2)
#         image_feats = torch.squeeze(self.dense1_i(image_embeds_jh)) #64*768
#         text_feats = torch.squeeze(self.dense1_t(text_embeds_jh))#64*768
        
#         image_feats = image_feats[:,:128] #bs*128
#         text_feats = text_feats[:,:128]#bs*128
        

        
#         #cat-1
#         fusion_feats=torch.cat((image_feats, text_feats),dim=1) #->bs*256
# #         fusion_feats=fusion_feats=self.dense3(fusion_feats)
# #         fusion_feats=self.relu(fusion_feats)
#         output = self.dense2(fusion_feats)
        
# #         #max/min
# #         fusion_feats=torch.max(image_feats, text_feats) #->bs*256
# # #         output = self.dense2(fusion_feats)
        
# #         #mul
# #         fusion_feats=image_feats*text_feats#->bs*256
        
# # # # #         fusion_feats = self.relu(fusion_feats)  #no-rule2
# #         output = self.dense2(fusion_feats)
# # # # #         output = torch.squeeze(self.dense2(fusion_feats)) #16,20 全部特征损失  #256
        
#         return image_feats,text_feats,output

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

    def forward(self,img):
        #ipdb.set_trace()
        #encoder_t_embeds = img
        #KLoss增加异构对齐部分
        
        
        img_t_embeds = img.permute(1, 0, 2) #64,30,768 ->30,64,768
        #text_t_embeds = text.permute(1,0,2) 

        # obtain gru output of hidden vectors
        h0_en = Parameter(torch.zeros((1, img.shape[0], self.num_hidden), requires_grad=True)) #(1，batchsize，768)
        
        #h1_en = Parameter(torch.zeros((1, text.shape[0], self.num_hidden), requires_grad=True)) #(1，batchsize，768)
        

        if self.CUDA:
            h0_en = h0_en.cuda()
            #h1_en = h1_en.cuda()
        gru_embed_img , _ = self.gru(img_t_embeds, h0_en) #30,batchsize,768 ;1,batchsize,768  
        #30,batchsize,768
        
        x = gru_embed_img.permute(1, 0, 2)# batchsize,30,768
        
#         gru_embed_text , _ = self.gru(text_t_embeds, h1_en) #30,batchsize,768 ;1,batchsize,768  
#         #30,batchsize,768
#         gru_embed_text = gru_embed_text.permute(1, 0, 2)# batchsize,30,768
        
#         #indexVectors = torch.cat([cls_token, indexVectors], dim=1).long()
#         gru_embed_full = torch.cat([gru_embed_img,gru_embed_text],dim=1)  #特征融合1
#         image_embeds_jh = x[:,0:60]
#         text_embeds_jh = x[:,-60:]
        
#         return image_embeds_jh,text_embeds_jh,x
         
#         return gru_embed_img
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    

class gru_encoder_t_text(nn.Module):
    #def __init__(self, CUDA, gloveVector, num_hidden, num_key_ingre=5):
    #def __init__(self, CUDA, gloveVector, num_hidden=768, num_key_ingre=5):
    def __init__(self,config,CUDA=1,num_hidden=768, num_key_ingre=5):
        super(gru_encoder_t_text, self).__init__()
        
        self.CUDA = CUDA
        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre

        self.gru = nn.GRU(768, self.num_hidden)
        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    def forward(self,text):
        # compute latent vectors
        #ipdb.set_trace()
        #encoder_t_embeds = img
        #img_t_embeds = img.permute(1, 0, 2) #64,30,768 ->30,64,768
        text_t_embeds = text.permute(1,0,2) 
        
        h1_en = Parameter(torch.zeros((1, text.shape[0], self.num_hidden), requires_grad=True)) #(1，batchsize，768)

        if self.CUDA:
            h1_en = h1_en.cuda()
        gru_embed_text , _ = self.gru(text_t_embeds, h1_en) #30,batchsize,768 ;1,batchsize,768  
        #30,batchsize,768
        gru_embed_text = gru_embed_text.permute(1, 0, 2)# batchsize,30,768
        
        
        return gru_embed_text #64,73,768

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    