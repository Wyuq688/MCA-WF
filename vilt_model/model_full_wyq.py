import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from packaging import version
from vilt_model.modules import heads, objectives, vilt_utils


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
        self.module1 = ViLTransformerSS(config)
        self.module2 = classifier(config["hidden_size"])
        self.module3 = ViLTransformerText(config)
#         self.dense1 = nn.Linear(30, 1) #V
        self.dense1 = nn.Linear(73, 1) #S
        self.dense2 = nn.Linear(1, 768)

    def forward(self, img, text):
#         ipdb.set_trace()
        img=self.module0(img) #64,30,768 #浅层特征
#         img,text=self.module0(img,text) #64,x,768
        raw_img=img
# image_embeds_pt, embeds_att_pt, embeds_jh_pt ,t_embeds_pt,t_embeds_att_pt,t_embeds_jh_pt
#         img=self.module0(img) #64,30,768 #浅层特征
    
#         image_embeds=img
#         t_embeds=text
        
        
        #image_embeds,embeds_jh,t_embeds,t_embeds_jh=self.module1(img,text)
        
#         raw_img=img #将图像特征进行保存起来进行运算
# #     #attention+S
        text_embed=self.module3(text) #64，73，768
        text_embed=text_embed.transpose(1, 2) #转置 64，768，73
        text_embed=self.dense1(text_embed) #64，768，1
        context_vector=torch.bmm(img,text_embed) #64,30,1
        weight=self.dense2(context_vector) #64,30,768
        context_vector=torch.softmax(weight,1)
        att_img=raw_img*context_vector
        
        embed=self.module1(att_img)
        
    #attention+V
#         img_vector=torch.mean(img,dim=1,keepdim=True) #取特征的平均值也即context vector 64*1*768
#         img_vector=img_vector.transpose(1, 2) #64,768,1
#         context_vector=torch.bmm(raw_img,img_vector) #64,30*1
#         context_vector=self.dense2(context_vector) #context vector 64*seq*768
#         att_img=raw_img*context_vector
#         embed_full = torch.cat([att_img,text],dim=1)  #前者特征融合1early fusion
#         embed = self.module1(att_img,text)  # img=64,30,768  后者特征融合
#         embed = self.module1(img,text)
#         embed = self.module1(embed_full)
        pred = self.module2(embed)
        return pred
#         
        
#         embeds_att=att_img #经过注意力机制的视觉和语义特征
#         t_embeds_att=text
        
#         image_embeds_jh,text_embeds_jh,embed = self.module1(att_img, text)#64,103,768  #融合之后送入backone的

# #         image_embeds_ll,text_embeds_ll,image_embeds_dv,text_embeds_dv,image_embeds,text_embeds = self.module1(att_img, text)
# #         image_feats,text_feats,output = self.module2( image_embeds,text_embeds) #视觉和语义分别预测加在一起的损失
# #         return  image_embeds_ll,text_embeds_ll,image_embeds_dv,text_embeds_dv,image_feats,text_feats,output
    
# #         embeds_jh=embed[:,:30]
# #         t_embeds_jh=embed[:,30:]
        
#         #embed=torch.cat([embeds_jh,t_embeds_jh],dim=1)
#         image_feats,text_feats,output = self.module2(image_embeds_jh,text_embeds_jh,embed ) #多模态预测
        
#         return image_feats,text_feats,output

#         return pre #16,172
#         return image_embeds, embeds_att, embeds_jh,pre,t_embeds,t_embeds_att,t_embeds_jh


class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(30, 1)  # 64 768 T ->64 768 1///145 16 40
        self.dense2 = nn.Linear(768, 20)  # 64 1 768->64 1 172/81  #vireo172/nus-wide
#         self.dense3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()


#加上异构对齐  
#     def forward(self,image_embeds_jh,text_embeds_jh,embeds):
    def forward(self,embeds):
#         ipdb.set_trace()
        all_embeds = embeds.transpose(1, 2) #16,768,160
        feats = self.dense1(all_embeds).transpose(1, 2)#16,1,768
        feats = self.relu(feats) 
        output = torch.squeeze(self.dense2(feats)) #16,20 全部特征损失
        return output
        
        #异构对齐损失
#         image_embeds_jh =image_embeds_jh.transpose(1,2)
#         text_embeds_jh = text_embeds_jh.transpose(1,2)
#         image_feats = torch.squeeze(self.dense1_i(image_embeds_jh)) #64*768
#         text_feats = torch.squeeze(self.dense1_t(text_embeds_jh))#64*768
        
# #         image_feats = self.relu(image_feats)#64*768 #no-rule1
# #         text_feats = self.relu(text_feats)#64*768
        
#         image_feats = image_feats[:,:128] #bs*128
#         text_feats = text_feats[:,:128]#bs*128
        
# #         #cat-1
# #         fusion_feats=torch.cat((image_feats, text_feats),dim=1) #->bs*256
# # # #cat-2    
        
# # #         fusion_feats=self.dense3(fusion_feats) #256->128
# # #         fusion_feats=self.relu(fusion_feats) #加一层激活函数
        
# # #         #max/min
# #         fusion_feats=torch.min(image_feats, text_feats) #->bs*256
        
# # #         #mul
#         fusion_feats=image_feats*text_feats#->bs*256
        
# # # # #         fusion_feats = self.relu(fusion_feats)  #no-rule2
#         output = self.dense2(fusion_feats)  #128->20
# # # #         output = torch.squeeze(self.dense2(fusion_feats)) #16,20 全部特征损失  #256
        
#         return image_feats,text_feats,output

      

class ViLTransformerSS(pl.LightningModule):
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

        hs = self.hparams.config["hidden_size"]

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        #ipdb.set_trace()
        #test_only时候用的load权重
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(self,img): 
#      
#         co_embed=torch.cat([img, text], dim=1)#先进行特征融合，再进行分类的结果
#         x=co_embed
        x=img
        
        #异构特征对齐
        
#         x = torch.cat( [text_embeds, image_embeds], dim=1)    
#         image_embeds_ll = x[:,-144:]
#         text_embeds_ll = x[:,1:text_embeds.size(1)]
#         image_embeds_dv = image_embeds_ll
#         text_embeds_dv = text_embeds_ll
# #         ipdb.set_trace()
# # 直接微调
        
#         for i, blk in enumerate(self.transformer.blocks):
#             if i<12:
#                 image_embeds_ll, _attn = blk(image_embeds_ll)
#         image_embeds = image_embeds_ll
#         image_embeds = self.transformer.norm(image_embeds)
#         text_embeds = self.transformer.norm(text_embeds_dv)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)
        x = self.transformer.norm(x)
        return x

#         for i, blk in enumerate(self.transformer.blocks):
#             x, _attn = blk(x)
#         x = self.transformer.norm(x)
#         image_embeds_ll = x[:,1:img.size(1)]
#         text_embeds_ll = x[:,-73:]
        
#         image_embeds_dv = image_embeds_ll  #多模态交互后的视觉特征
#         text_embeds_dv = text_embeds_ll #多模态交互后的文本特征
        
#         image_embeds_ll = torch.squeeze(torch.mean(image_embeds_ll, dim = 1))
#         text_embeds_ll = torch.squeeze(torch.mean(text_embeds_ll, dim = 1))
#         image_embeds_dv = torch.squeeze(torch.mean(image_embeds_dv, dim = 1))
#         text_embeds_dv = torch.squeeze(torch.mean(text_embeds_dv, dim = 1))
   
#         return x


    def forward(self,img):
        x = self.infer(img)  #64,103,768        
        return x
        
#     def forward(self, img, text):
#         x = self.infer(img, text)  #64,103,768
# #         image_embeds_jh = x[:,0:60]
# #         text_embeds_jh = x[:,-60:]
        
#         return x
        


#得到文本的交互特征
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
