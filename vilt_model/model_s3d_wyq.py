import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig
from packaging import version
from vilt_model.modules import heads, objectives, vilt_utils


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
        self.module0 = model_Linear()
        self.module1 = ViLTransformerSS(config)
        self.module2 = classifier(config["hidden_size"],config["se_dim"])
        #self.module2 = classifier(144,768)
        self.sigmoid=nn.Sigmoid()

    def forward(self, img):
#         ipdb.set_trace()
        
        img=self.module0(img) #64,x,768
#         context_vector=torch.mean(img,dim=1,keepdim=True)
#         context_vector=context_vector.transpose(1, 2) #64,768,1
#         weight=torch.bmm(img,context_vector) #64,30*1
#         weight=self.sigmoid(weight)
#         weight=torch.softmax(weight,1)
#         enimg=img*weight
        embed=self.module1(img)
#         embed = self.module1(enimg)  ##[16,3,384,384] #clo_token[16,x,768] jump to 196 line
        pred = self.module2(embed)

        return pred #64*20维度的向量


class classifier(nn.Module):
    def __init__(self, hidden_size,se_dim):
        super().__init__()
        self.dense1 = nn.Linear(30, 1)  #msrvtt 64,x,1 activitynet 64,
#         # 64 768 x->64 768 1  error :1920*768and 30*1
        self.dense2 = nn.Linear(hidden_size, 20)  # 64 1 768->64 1 20个类
        #self.dense1 = nn.Linear(144, 1)  #msrvtt 64,x,1 activitynet 64,
        #self.dense2 = nn.Linear(768, 172)
        self.relu = nn.ReLU()

    def forward(self, embeds):
        #ipdb.set_trace()
        #embeds=64,30,768
        
        #embeds = embeds[:, 1:, :]  #16,160,768  #16，30，768
        image_embeds = embeds.transpose(1, 2)  #16,768,160  #16，768，30
        feats = torch.squeeze(self.dense1(image_embeds).transpose(1, 2)) #64*768;#64,1,768
        feats = self.relu(feats) #64*768
        output = self.dense2(feats) #64*20 #预测
        feats = nn.functional.normalize(feats, dim=-1) #64*768 特征
        return output


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

        #self.text_embeddings = BertEmbeddings(bert_config)
        #self.text_embeddings.apply(objectives.init_weights)

        #self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])  # 建立一个长度2的字典
        #self.token_type_embeddings.apply(objectives.init_weights)

        # is it  pretrained vit?
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        #         ipdb.set_trace()
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)


# ==============================================================Downstream =========================================================================== #
        # load pretrain and downstream tesk head
        # exp1:   image Classification
        # 172 class for food
        # exp2:  text classification
        # 353 class for ingredients
#         ipdb.set_trace()
        # load pretrained vilt
        if (
                self.hparams.config["load_path"].split('.')[-1] == "pth"
                and not self.hparams.config["test_only"]
        ):
            #             ipdb.set_trace()
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("- - - - - - - -\n ckpt : {} has been loaded in model \n - - - - - - - -".format(
                self.hparams.config["load_path"].split('/')[-1]))

        hs = self.hparams.config["hidden_size"]


        vilt_utils.set_metrics(self)
        self.current_tasks = list()

# ========================================================load downstream (test_only) ===================================================================
        # ipdb.set_trace()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
            self,
            img,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):

        x=img  #64,30,768维度的向量
        
        for i, blk in enumerate(self.transformer.blocks):
            #ipdb.set_trace()
            x, _attn = blk(x)

        x = self.transformer.norm(x)
        return x  #
 


    def forward(self,img): #输出的为img特征

        x = self.infer(img)
        return x  
    

       
    def training_step(self, img, batch_idx):
        vilt_utils.set_task(self)
        output = self(img)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, img, batch_idx):
        vilt_utils.set_task(self)
        output = self(img)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, img, batch_idx):
        vilt_utils.set_task(self)
        output = self(img)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, img, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
