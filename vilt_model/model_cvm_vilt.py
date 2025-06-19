import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig,BertEmbeddings
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
        self.module3 = ViLTransformerText(config)

        self.dense1 = nn.Linear(30, 1) #第一个加权操作得到的linear层
        self.dense_t = nn.Linear(73, 1) #S 这里要改动
        self.dense2 = nn.Linear(1, 768)
        self.dense3 = nn.Linear(1536,1)
        self.dense4 = nn.Linear(768,768)
        
        self.sigmoid= nn.Sigmoid()
        self.relu = nn.ReLU()
       
        
    def forward(self, s3d_features,text_features):
#=============================================attentionV:基于视觉注意力机制的视觉增强算法============================================================
#------------------------------------------vilt-(w1+sigmoid)-activitynet------------------------------------------------------------------------------ 
#         s3d_features,text_features=self.module0(s3d_features,text_features) #32,30,768
#         raw_img=s3d_features
        
#         context_vector=torch.mean(s3d_features,dim=1,keepdim=True) #取特征的平均值也即context vector 64*1*768
#         context_vector=context_vector.transpose(1, 2) #64,768,1
#         weight=torch.bmm(s3d_features,context_vector) #64,30*1
#         weight=self.sigmoid(weight)
#         weight=torch.softmax(weight,1)
#         enhence_img=raw_img*weight   #视觉增强的视频帧
#         x=self.module1(enhence_img, text_features) #vilt-backbone
#         pred = self.module2(x)
#         return pred
   
#-------------------------------------vilt-(w2+sigmoid)-msrvtt-------------------------------------------------------------------------------------
#         s3d_features=self.module0(s3d_features) #32,30,768
#         raw_img=s3d_features
#         context_vector=torch.mean(s3d_features,dim=1,keepdim=True)
#         context_vector=context_vector.repeat(1,30,1)
#         mixcat=torch.cat([s3d_features,context_vector],dim=2)
#         weight=self.dense3(mixcat)
#         weight=self.sigmoid(weight)
#         weight=torch.softmax(weight,1)
#         enhence_img=raw_img*weight
#         x=self.module1(enhence_img,text_features)
#         pred = self.module2(x)
#         return pred

#--------------------------------attentionS+attentionV-------------------------------------------------------------------------------------------
        s3d_features=self.module0(s3d_features) #32,30,768
        raw_img=s3d_features
        context_vector=torch.mean(s3d_features,dim=1,keepdim=True)
        context_vector=context_vector.repeat(1,30,1)
        mixcat=torch.cat([s3d_features,context_vector],dim=2)
        weightV=self.dense3(mixcat)
        weightV=self.sigmoid(weightV)
        weightV=torch.softmax(weightV,1)
        
#         ipdb.set_trace()
        text_embed=self.module3(text_features)#64,73,768
        text_embed=text_embed.transpose(1, 2) #转置 64，768，73
        text_embed=self.dense_t(text_embed) #64，768，1
        context_vector_S=torch.bmm(raw_img,text_embed) #64,30,1
        context_vector_S=torch.softmax(context_vector_S,1)
        
        weight=weightV+context_vector_S
        
        enhence_img=raw_img*weight
        embeds_att=enhence_img
        t_embeds_att=text_features

        embeds_jh,t_embeds_jh,x=self.module1(enhence_img,text_features)
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
    

#Vilt model

class BaseboneModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])  # 建立一个长度2的字典
        self.token_type_embeddings.apply(objectives.init_weights)
     
        # is it  pretrained vit?
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        #ipdb.set_trace()
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # load pretrained vilt
#         if (
#                 self.hparams.config["load_path"].split('.')[-1] == "ckpt"
#                 and not self.hparams.config["test_only"]
#         ):
#             #             ipdb.set_trace()
#             ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
#             state_dict = ckpt["state_dict"]
#             self.load_state_dict(state_dict, strict=False)
#             print("- - - - - - - -\n ckpt : {} has been loaded in model \n - - - - - - - -".format(
#                 self.hparams.config["load_path"].split('/')[-1]))

#         hs = self.hparams.config["hidden_size"]

#         vilt_utils.set_metrics(self)
#         self.current_tasks = list()

#         # ===================== load downstream (test_only) ======================
#         #ipdb.set_trace()

#         if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
#             ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
#             state_dict = ckpt["state_dict"]
#             self.load_state_dict(state_dict, strict=False)


####==========================load best model===========================================#####
        if self.hparams.config["load_path"] != "":
#             ipdb.set_trace()
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt
          
            self.load_state_dict(state_dict, strict=False)
#------------------------------------------------------------------------------------------------#
    def infer(
            self,
            img,
            text,
    ):
#      
        co_embed=torch.cat([img, text], dim=1)#先进行特征融合，再进行分类的结果
        x=co_embed
        
       
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)
        x = self.transformer.norm(x)
        embeds_jh=x[:,:30]
        t_embeds_jh=x[:,30:]
        return embeds_jh,t_embeds_jh,x


        
    def forward(self, img, text):
        embeds_jh,t_embeds_jh,x = self.infer(img, text)  #64,103,768
        return embeds_jh,t_embeds_jh,x
    
    
    
    
class ViLTransformerText(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])  # 建立一个长度2的字典
        self.token_type_embeddings.apply(objectives.init_weights)
     
        # is it  pretrained vit?
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        #ipdb.set_trace()
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

#         # load pretrained vilt
#         if (
#                 self.hparams.config["load_path"].split('.')[-1] == "ckpt"
#                 and not self.hparams.config["test_only"]
#         ):
#             #             ipdb.set_trace()
#             ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
#             state_dict = ckpt["state_dict"]
#             self.load_state_dict(state_dict, strict=False)
#             print("- - - - - - - -\n ckpt : {} has been loaded in model \n - - - - - - - -".format(
#                 self.hparams.config["load_path"].split('/')[-1]))

        hs = self.hparams.config["hidden_size"]

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        #ipdb.set_trace()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(self,text):
   
        #co_embed=torch.cat([img, text], dim=1)
        #x=co_embed
        x=text

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)
        x = self.transformer.norm(x)
        return x

    def forward(self,text):

        text_embeds = self.infer(text)  #64,103,768
        return text_embeds
