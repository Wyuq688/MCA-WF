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

import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from packaging import version
from vilt_model.modules import heads, objectives, vilt_utils


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


    
class VilT_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module0 = model_Linear()
#         self.module1 = BaseboneModel(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
        self.module1 = BaseboneModel(config)

        self.module2 = classifier(config["hidden_size"])
        self.module3 = gru_encoder_t_text(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
#         self.module3 = ViLTransformerText(config)
        
        self.dense1 = nn.Linear(30, 1) #
        self.dense_t = nn.Linear(73, 1) #S 这里要改动
        self.dense2 = nn.Linear(1, 768)
        self.dense3 = nn.Linear(1536,1)
        self.dense4 = nn.Linear(768,768)
        
        self.sigmoid= nn.Sigmoid()
         
    
    def forward(self, s3d_features,text_features,pos_s3d_features,pos_text_features,batch_size_cur):
        
#         ipdb.set_trace()
        s3d_features=self.module0(s3d_features) #32,30,768
        pos_s3d_features=self.module0(pos_s3d_features)#32,30,768
        
        A=s3d_features  #16*30*768
        B=pos_s3d_features
        
        raw_imgA=A
        raw_imgB=B
        
        raw_img=torch.cat([raw_imgA,raw_imgB],dim=0)
        
        
        #attention S
        all_text=torch.cat([text_features,pos_text_features],dim=0)
        text_embed=self.module3(all_text)#64,73,768
        text_embed=text_embed.transpose(1, 2) #转置 64，768，73
        text_embed=self.dense_t(text_embed) #64，768，1
        context_vector_S=torch.bmm(raw_img,text_embed) #64,30,1
        context_vector_S=torch.softmax(context_vector_S,1)
#         context_vector_S=self.dense2(context_vector_S) #64,30,768
        
        
        #attentionV
        contextV=torch.mean(raw_img,dim=1,keepdim=True) #1*768
        contextV=contextV.transpose(1, 2) #768*1 30*768
        contextV=torch.bmm(raw_img,contextV) #64,30*1
#         context_vectorA=self.dense2(context_vectorA) #context vector 64*seq*768
        contextV=torch.softmax(contextV,1) 
    
#         contextV=torch.mean(raw_img,dim=1,keepdim=True) #1*768
#         contextV=contextV.transpose(1, 2) #768*1 30*768
#         contextV=torch.bmm(raw_img,contextV) #64,30*1
# #         context_vectorA=self.dense2(context_vectorA) #context vector 64*seq*768
#         contextV=torch.softmax(contextV,1)      
      
       #正样本对比注意力关键帧选择  attentionC
#         ipdb.set_trace()
        meanB=torch.mean(B,dim=1,keepdim=True) #16*1*768
        BB=meanB.repeat(1,30,1)
        AA=torch.cat([A,BB],dim=2) #A:16*30*1536
        AA=self.dense3(AA)  #AA=16*30*1
        causal_A=torch.softmax(AA,1)# A的因果帧 16*30*1
            
        meanA=torch.mean(A,dim=1,keepdim=True)
        AA=meanA.repeat(1,30,1)
        cont=torch.cat([B,AA],dim=2)
        BB=self.dense3(cont)  #AA=16*30*1
        causal_B=torch.softmax(BB,1) #B的因果帧 30*1
        
        causalP=torch.cat([causal_A,causal_B],dim=0)
        
        causal=causalP+context_vector_S+contextV
        
        causal=self.dense2(causal)
#         causal=self.sigmoid(causal)
        full_en_img=raw_img*causal
        
        full_text_features=torch.cat([text_features,pos_text_features],dim=0)
        
        x=torch.cat([full_en_img,full_text_features],dim=1)
        x=self.module1(x)  
        pred = self.module2(x)

        
        
        
        
        
        #需要计算loss来进行约束正样本和样本之间进行关键帧选择
        
        value_A,indices_A=causal_A.topk(10,dim=1,largest=True,sorted=True) #top10A  16*10*1
        value_B,indices_B=causal_B.topk(10,dim=1,largest=True,sorted=True) #top10B  16*10*1
        
        #存储因果帧对应的帧的特征
        TA_features=torch.zeros(len(A),10,768)
        TB_features=torch.zeros(len(B),10,768) 
        for i in range(len(A)):
            TA_features[i][:]=A[i, indices_A[i].squeeze(1),:]
        for i in range(len(B)):
            TB_features[i][:]=B[i, indices_B[i].squeeze(1),:]
                
#         ipdb.set_trace()

        TA_features=TA_features.cuda()
        TB_features=TB_features.cuda()
        
        TA_features=self.dense4(TA_features)
        TB_features=self.dense4(TB_features) #因果特征考虑次序问题，先经过一层linear，然后取mean，计算mseloss
        
        topmean_A=torch.mean(TA_features,dim=1,keepdim=True) #变成BS*1*768
        topmean_B=torch.mean(TB_features,dim=1,keepdim=True)
        
        criterion =torch.nn.MSELoss(reduction='mean') #True是返回向量形式的loss，False是标量形式；第二个True返回loss.mean()，loss.sum()
        lossAB = criterion(topmean_A, topmean_B)
        
        
        return pred,lossAB
#         ipdb.set_trace()

        

    

class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(103, 1)  #msrvtt 64,x,1 activitynet 64,
#         # 64 768 x->64 768 1  error :1920*768and 30*1
        self.dense2 = nn.Linear(768, 20)  # 64 1 768->64 1 20个
        self.relu = nn.LeakyReLU()
        self.dense1_i = nn.Linear(30, 1)
        self.dense1_t = nn.Linear(73, 1)

#     def forward(self, image_embeds_jh,text_embeds_jh,x):
    def forward(self, embeds):
#         ipdb.set_trace()
#         embeds=64,30,768 
        image_embeds = embeds.transpose(1, 2)  #16,768,160  #16，768，30
        feats = torch.squeeze(self.dense1(image_embeds).transpose(1, 2)) #64*768;#64,1,768
        feats = self.relu(feats) #64*768
        output = self.dense2(feats) #64*20 #预测
        return output


    
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
        #30,batchsize,768
        x = gru_embed_img.permute(1, 0, 2)# batchsize,30,768
        return x


    
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
    
if __name__ == "__main__":
    # 定义配置字典
    config = {
        "hidden_size": 768,
        "num_hidden": 768,
        "num_key_ingre": 5,
        "CUDA": 1,
    }

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() and config["CUDA"] else "cpu")

    # 初始化模型
    model = VilT_Classification(config).to(device)

    # 生成随机输入数据
    batch_size = 32
    seq_length_img = 30
    seq_length_text = 73
    feature_dim_img = 1024
    feature_dim_text = 768

    s3d_features = torch.randn(batch_size, seq_length_img, feature_dim_img).to(device)
    text_features = torch.randn(batch_size, seq_length_text, feature_dim_text).to(device)
    pos_s3d_features = torch.randn(batch_size, seq_length_img, feature_dim_img).to(device)
    pos_text_features = torch.randn(batch_size, seq_length_text, feature_dim_text).to(device)
    batch_size_cur = batch_size

    # 前向传播
    pred, lossAB = model(s3d_features, text_features, pos_s3d_features, pos_text_features, batch_size_cur)

    # 输出结果形状
    print("预测结果形状:", pred.shape)
    print("损失值形状:", lossAB.shape)

    # 打印模型结构（可选）
    print("\n模型结构:")
    print(model)