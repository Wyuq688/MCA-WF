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
        
    def forward(self, img,text):
        return self.linear1(img),self.linear2(text)
#     def forward(self, img):
#         return self.linear1(img)


    
class VilT_Classification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module0 = model_Linear()
#         self.module1 = BaseboneModel(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
        self.module1 = BaseboneModel(config)
        self.module2 = classifier(config["hidden_size"])
#         self.module3 = gru_encoder_t_text(config,config["CUDA"],config["num_hidden"],config["num_key_ingre"])
        self.module3 = ViLTransformerText(config)
        
        self.dense1 = nn.Linear(30, 1) #
        self.dense_t = nn.Linear(60, 1) #S 这里要改动
        self.dense2 = nn.Linear(1, 768)
        self.dense3 = nn.Linear(1536,1)
        self.dense4 = nn.Linear(768,768)
        
        self.sigmoid= nn.Sigmoid()

    def forward(self, s3d_features, text_features, pos_s3d_features, pos_text_features, batch_size_cur):

            # ipdb.set_trace()
            s3d_features = self.module0(s3d_features)  # 32,30,768
            pos_s3d_features = self.module0(pos_s3d_features)  # 32,30,768

            A = s3d_features  # 16*30*768
            B = pos_s3d_features

            raw_imgA = A
            raw_imgB = B

            raw_img = torch.cat([raw_imgA, raw_imgB], dim=0)

            # attention S
            all_text = torch.cat([text_features, pos_text_features], dim=0)
            text_embed = self.module3(all_text)  # 64,73,768
            text_embed = text_embed.transpose(1, 2)  # 转置 64，768，73
            text_embed = self.dense_t(text_embed)  # 64，768，1
            context_vector_S = torch.bmm(raw_img, text_embed)  # 64,30,1
            context_vector_S = torch.softmax(context_vector_S, 1)
            #         context_vector_S=self.dense2(context_vector_S) #64,30,768

            # attentionV
            contextV = torch.mean(raw_img, dim=1, keepdim=True)  # 1*768
            contextV = contextV.transpose(1, 2)  # 768*1 30*768
            contextV = torch.bmm(raw_img, contextV)  # 64,30*1
            #         context_vectorA=self.dense2(context_vectorA) #context vector 64*seq*768
            contextV = torch.softmax(contextV, 1)

            # 正样本对比注意力关键帧选择  attentionC
            #         ipdb.set_trace()
            meanB = torch.mean(B, dim=1, keepdim=True)  # 16*1*768
            BB = meanB.repeat(1, 30, 1)
            AA = torch.cat([A, BB], dim=2)  # A:16*30*1536
            AA = self.dense3(AA)  # AA=16*30*1
            causal_A = torch.softmax(AA, 1)  # A的因果帧 16*30*1

            meanA = torch.mean(A, dim=1, keepdim=True)
            AA = meanA.repeat(1, 30, 1)
            cont = torch.cat([B, AA], dim=2)
            BB = self.dense3(cont)  # AA=16*30*1
            causal_B = torch.softmax(BB, 1)  # B的因果帧 30*1

            causalP = torch.cat([causal_A, causal_B], dim=0)

            causal = causalP + context_vector_S + contextV

            causal = self.dense2(causal)
            #         causal=self.sigmoid(causal)
            full_en_img = raw_img * causal

            full_text_features = torch.cat([text_features, pos_text_features], dim=0)

            x = torch.cat([full_en_img, full_text_features], dim=1)
            x = self.module1(x)
            pred = self.module2(x)

            # 需要计算loss来进行约束正样本和样本之间进行关键帧选择

            value_A, indices_A = causal_A.topk(10, dim=1, largest=True, sorted=True)  # top10A  16*10*1
            value_B, indices_B = causal_B.topk(10, dim=1, largest=True, sorted=True)  # top10B  16*10*1

            # 存储因果帧对应的帧的特征
            TA_features = torch.zeros(len(A), 10, 768)
            TB_features = torch.zeros(len(B), 10, 768)
            for i in range(len(A)):
                TA_features[i][:] = A[i, indices_A[i].squeeze(1), :]
            for i in range(len(B)):
                TB_features[i][:] = B[i, indices_B[i].squeeze(1), :]

            #         ipdb.set_trace()

            TA_features = TA_features.cuda()
            TB_features = TB_features.cuda()

            TA_features = self.dense4(TA_features)
            TB_features = self.dense4(TB_features)  # 因果特征考虑次序问题，先经过一层linear，然后取mean，计算mseloss

            topmean_A = torch.mean(TA_features, dim=1, keepdim=True)  # 变成BS*1*768
            topmean_B = torch.mean(TB_features, dim=1, keepdim=True)

            criterion = torch.nn.MSELoss(
                reduction='mean')  # True是返回向量形式的loss，False是标量形式；第二个True返回loss.mean()，loss.sum()
            lossAB = criterion(topmean_A, topmean_B)

            return pred, lossAB

    #         ipdb.set_trace()


#
class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(120,1)  #msrvtt 64,x,1 activitynet 64,
#         # 64 768 x->64 768 1  error :1920*768and 30*1
        self.dense2 = nn.Linear(768, 200)  # 64 1 768->64 1 20个
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

#         feats = nn.functional.normalize(feats, dim=-1) #64*768 特征


#         image_embeds_jh =image_embeds_jh.transpose(1,2)#64*768*73
#         text_embeds_jh = text_embeds_jh.transpose(1,2)
#         image_feats = torch.squeeze(self.dense1_i(image_embeds_jh)) #64*768
#         text_feats = torch.squeeze(self.dense1_t(text_embeds_jh))#64*768
        
#         image_feats = image_feats[:,:128] #bs*128
#         text_feats = text_feats[:,:128]#bs*128
        

        
#         #cat-1
#         fusion_feats=torch.cat((image_feats, text_feats),dim=1) #->bs*256
# #         usion_feats=self.dense3(fusion_feats)
# # #         fusion_feats=self.relu(fusion_feats)
#         output = self.dense2(fusion_feats)
        
# # #         #max/min
# # #         fusion_feats=torch.max(image_feats, text_feats) #->bs*256
# # # #         output = self.dense2(fusion_feats)
        
# # #         #mul
# # #         fusion_feats=image_feats*text_feats#->bs*256
        
# # # # # #         fusion_feats = self.relu(fusion_feats)  #no-rule2
# # #         output = self.dense2(fusion_feats)
# # # # # #         output = torch.squeeze(self.dense2(fusion_feats)) #16,20 全部特征损失  #256
        
#         return image_feats,text_feats,output

#         return output



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

#         # load pretrained vilt
#         if (
#                 self.hparams.config["load_path"].split('.')[-1] == "pt"
#                 and not self.hparams.config["test_only"]
#         ):
#             #             ipdb.set_trace()
#             ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
#             state_dict = ckpt
# #             state_dict = ckpt["state_dict"]
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

    def infer(
            self,
            img,
            text
          
    ):
#      
        co_embed=torch.cat([img, text], dim=1)#先进行特征融合，再进行分类的结果
        x=co_embed
#         x=img
       
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)
        x = self.transformer.norm(x)
        embed=x #128*120*768
#         ipdb.set_trace()
        A=embed[:200,:] #60,60
        B=embed[200:,:]
        embeds_jh_pt=A[:,:60]
        t_embeds_jh_pt=A[:,60:]
        return embeds_jh_pt,t_embeds_jh_pt,x
#         return x

      
#     def forward(self, img):
# #         x = self.infer(img, text)  #64,103,768
# #         image_embeds_jh = x[:,0:30]
# #         text_embeds_jh = x[:,-73:]
#         x=self.infer(img) 
# #         return image_embeds_jh,text_embeds_jh,x
#         return x
        
    def forward(self, img, text):
#         x = self.infer(img, text)  #64,103,768
#         image_embeds_jh = x[:,0:30]
#         text_embeds_jh = x[:,-73:]
        embeds_jh_pt,t_embeds_jh_pt,x=self.infer(img,text) 
#         return image_embeds_jh,text_embeds_jh,x
        return embeds_jh_pt,t_embeds_jh_pt,x
        

        
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
if __name__ == "__main__":
    # 定义配置字典
    config = {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "max_text_len": 60,
        "drop_rate": 0.1,
        "vocab_size": 30522,
        "load_path": "",
        "vit": "vit_base_patch16_224",
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
    seq_length_text = 60
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
    #
    # 以上代码实现了：模型定义 ：定义了一个基于PyTorch Lightning的视频分类模型VilT_Classification，它包含多个子模块，如
    # model_Linear、BaseboneModel、classifier和ViLTransformerText等。配置字典 ：在
    # if __name__ == "__main__": 部分，定义了一个包含模型超参数的配置字典
    # config，如隐藏层大小、文本最大长度、dropout概率等。
    # 设备设置 ：根据配置和系统实际情况确定使用GPU还是CPU。
    # 模型初始化 ：利用配置字典初始化VilT_Classification模型，并将其移动到相应的设备上。
    # 随机数据生成 ：生成了符合模型输入要求的随机视觉特征、文本特征及其对应的正样本特征，用于模拟实际输入数据。
    # 前向传播测试 ：将生成的随机数据输入到模型中，进行前向传播，得到预测结果和损失值。
    # 结果输出 ：打印预测结果和损失值的形状，以验证模型的输出是否符合预期，同时可选择打印模型结构以便进一步确认模型构建的正确性。
    # 执行上述代码即可对VilT_Classification模型进行基本的功能测试和验证。