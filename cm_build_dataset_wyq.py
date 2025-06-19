import io
import scipy.io as matio
import numpy as np
from PIL import Image
import ipdb
import torch.utils.data
from torchvision import transforms
import torch

 
def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


class dataset_stage1(torch.utils.data.Dataset):#msrvtt and activitynet
    def __init__(self, dataset_indicator, data_path=None, transform=None,flag='train'):
        """Method to initilaize variables."""
        if dataset_indicator=='msrvtt':
            if flag=='train':
                self.s3d_features=np.load(data_path+'msrvtt_train_s3d_features.npy') #(7010,30,1024)
                #self.text_features=np.load(data_path+'msrvtt_train_txt_features.npy') #(7010,383)
                with io.open(data_path+'id_train_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'msrvtt_test_s3d_features.npy') #(2990,30,1024)
                #self.text_features=np.load(data_path+'msrvtt_test_txt_features.npy')#(2990,363)
                with io.open(data_path+'id_test_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
        
        elif dataset_indicator=='activitynet':
            if flag=='train':
                self.s3d_features=np.load(data_path+'news3d_activitynet_train.npy') #(10009,60,1024)
                with io.open(data_path+'activitynet_train_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'news3d_activitynet_test.npy') #(4515,60,,1024)
                with io.open(data_path+'activitynet_test_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
            
        self.labels=np.array(labels,dtype=int)
        self.dataset_indicator = dataset_indicator
        self.transform = transform
        #self.text_features=text_features
        

    #取出一次训练所需要的数据
    def __getitem__(self, index):
        #ipdb.set_trace()
        if self.dataset_indicator=='msrvtt':
            labels=self.labels[index]
            s3d_features=self.s3d_features[index]
            #text_features=self.text_features[index]
            #return s3d_features,text_features,labels
            #return text_features,labels
        
        elif self.dataset_indicator=='activitynet':
            labels=self.labels[index]
            s3d_features=self.s3d_features[index]
            #vgg_features=self.text_features[index]
        return s3d_features,labels
            #return s3d_features,vgg_features,labels
      
    def __len__(self):
        
        return len(self.s3d_features)
    
    
class dataset_stage2(torch.utils.data.Dataset):#msrvtt 的文本特征和activitynet的vgg的音频特征
    def __init__(self, dataset_indicator, data_path=None, transform=None,flag='train'):
        """Method to initilaize variables."""
        if dataset_indicator=='msrvtt':
            if flag=='train':
                self.text_features=np.load(data_path+'train_text_features.npy') #(7010,73,768)
                with io.open(data_path+'id_train_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.text_features=np.load(data_path+'samedim_test_text_features.npy') #(2990,73,768)
                with io.open(data_path+'id_test_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
        
        elif dataset_indicator=='activitynet':
            if flag=='train':
                self.vgg_features=np.load(data_path+'newvggish_activitynet_train.npy') #(10009,60,128)
                with io.open(data_path+'activitynet_train_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.vgg_features=np.load(data_path+'newvggish_activitynet_test.npy')#(4515,60,128)
                with io.open(data_path+'activitynet_test_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
            
        self.labels=np.array(labels,dtype=int)
        self.dataset_indicator = dataset_indicator
        self.transform = transform
        

    #取出一次训练所需要的数据
    def __getitem__(self, index):
        #ipdb.set_trace()
        if self.dataset_indicator=='msrvtt':
            labels=self.labels[index]
            text_features=self.text_features[index]
            return text_features,labels
        
        elif self.dataset_indicator=='activitynet':
            labels=self.labels[index]
            vgg_features=self.vgg_features[index]
            return vgg_features,labels
      
    def __len__(self):
        if self.dataset_indicator=='msrvtt':
            return len(self.text_features)
        elif self.dataset_indicator=='activitynet':
            return len(self.vgg_features)
    
    
    
class dataset_stage3(torch.utils.data.Dataset):#msrvtt and activitynet
    def __init__(self, dataset_indicator, data_path=None, transform=None,flag='train'):
        """Method to initilaize variables."""
        if dataset_indicator=='msrvtt':
            if flag=='train':
                self.s3d_features=np.load(data_path+'msrvtt_train_s3d_features.npy') #(7010,30,1024)
                self.text_features=np.load(data_path+'train_text_features.npy') #(7010,73,768)
                with io.open(data_path+'id_train_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'msrvtt_test_s3d_features.npy') #(2990,30,1024)
                self.text_features=np.load(data_path+'samedim_test_text_features.npy')#(2990,73,768)
                with io.open(data_path+'id_test_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
        
        elif dataset_indicator=='activitynet':
            if flag=='train':
                self.s3d_features=np.load(data_path+'news3d_activitynet_train.npy') #(10009,60,1024)
                self.vgg_features=np.load(data_path+'newvggish_activitynet_train.npy') #(10009,60,128)
                with io.open(data_path+'activitynet_train_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'news3d_activitynet_test.npy') #(4515,60,1024)
                self.vgg_features=np.load(data_path+'newvggish_activitynet_test.npy')#(4515,60,128)
                with io.open(data_path+'activitynet_test_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
            
        self.labels=np.array(labels,dtype=int)
        self.dataset_indicator = dataset_indicator
        self.transform = transform
        

    #取出一次训练所需要的数据
    def __getitem__(self, index):
        #ipdb.set_trace()
        if self.dataset_indicator=='msrvtt':
            labels=self.labels[index]
            s3d_features=self.s3d_features[index]
            text_features=self.text_features[index]
            return [s3d_features,text_features],labels
        
        elif self.dataset_indicator=='activitynet':
            labels=self.labels[index]
            s3d_features=self.s3d_features[index]
            vgg_features=self.vgg_features[index]
            return [s3d_features,vgg_features],labels
      
    def __len__(self):
        return len(self.s3d_features)
# labels=[]
# ipdb.set_trace()
    
class dataset_stage4(torch.utils.data.Dataset):#msrvtt and activitynet
    def __init__(self, dataset_indicator, data_path=None, transform=None,flag='train'):
        """Method to initilaize variables."""
        if dataset_indicator=='msrvtt':
            if flag=='train':
                self.s3d_features=np.load(data_path+'msrvtt_train_s3d_features.npy') #(7010,30,1024)
                self.text_features=np.load(data_path+'train_text_features.npy') #(7010,73,768)
                with io.open(data_path+'id_train_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'msrvtt_test_s3d_features.npy') #(2990,30,1024)
                self.text_features=np.load(data_path+'samedim_test_text_features.npy')#(2990,73,768)
                with io.open(data_path+'id_test_msrvtt_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
        
        elif dataset_indicator=='activitynet':
            if flag=='train':
                self.s3d_features=np.load(data_path+'news3d_activitynet_train.npy') #(10009,60,1024)
                self.vgg_features=np.load(data_path+'newvggish_activitynet_train.npy') #(10009,60,128)
                with io.open(data_path+'activitynet_train_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]

            elif flag=='test':
                self.s3d_features=np.load(data_path+'news3d_activitynet_test.npy') #(4515,60,1024)
                self.vgg_features=np.load(data_path+'newvggish_activitynet_test.npy')#(4515,60,128)
                with io.open(data_path+'activitynet_test_id_labels.txt',encoding='utf-8') as file:
                    labels=file.read().split('\n')[:-1]
            
        self.labels=np.array(labels,dtype=int)
        self.dataset_indicator = dataset_indicator
        self.transform = transform
        

    #取出一次训练所需要的数据
    def __getitem__(self, index):
        #ipdb.set_trace()
        if self.dataset_indicator=='msrvtt':
            label=self.labels[index]
            
            pos_list = np.where(np.isin(self.labels,label))[0]
#             pos_list = np.delete(pos_list,index)
            pos_index=np.random.choice(pos_list)
    
            while True:
                if pos_index==index:
                    pos_index=np.random.choice(pos_list)
                else:
                    break
                    
            pos_label=self.labels[pos_index]
            
            s3d_features=self.s3d_features[index]
            text_features=self.text_features[index]
            pos_s3d_features=self.s3d_features[pos_index]
            pos_text_features=self.text_features[pos_index]
            
            return [[s3d_features,text_features,index],[pos_s3d_features,pos_text_features,pos_index]],[label,pos_label]
        
        elif self.dataset_indicator=='activitynet':
            label=self.labels[index]
            pos_list = np.where(np.isin(self.labels,label))[0]
#             pos_list = np.delete(pos_list,index)
            pos_index=np.random.choice(pos_list)
    
            while True:
                if pos_index==index:
                    pos_index=np.random.choice(pos_list)
                else:
                    break
                    
            pos_label=self.labels[pos_index]
            
            s3d_features=self.s3d_features[index]
            vgg_features=self.vgg_features[index]
            pos_s3d_features=self.s3d_features[pos_index]
            pos_vgg_features=self.vgg_features[pos_index]
            return [[s3d_features,vgg_features,index],[pos_s3d_features,pos_vgg_features,pos_index]],[label,pos_label]
      
    def __len__(self):
        return len(self.s3d_features*2)
        
def build_dataset(train_stage, image_path, data_path, transform, mode, dataset_indicator, flag):
    if train_stage==1: #download msrvtt/activitynet dataset
        dataset=dataset_stage1(dataset_indicator,data_path=data_path,transform=transform,flag=flag)
    elif train_stage==2: #download msrvtt text features or activitynet vgg features
        dataset=dataset_stage2(dataset_indicator,data_path=data_path,transform=transform,flag=flag)
    elif train_stage in [3,6]:  #download msrvtt text and s3d features 
        dataset=dataset_stage3(dataset_indicator,data_path=data_path,transform=transform,flag=flag)
    elif train_stage in [4,5,7]: #download msrvtt text and activitynet positive features
        dataset = dataset_stage4(dataset_indicator,data_path=data_path,transform=transform,flag=flag)
    else:
        
        assert 1 < 0, 'Please fill the correct train stage!'

    return dataset

