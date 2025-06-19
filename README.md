# MCA-WF （IJCNN2023 项目代码说明）
## 一、项目概述
本项目基于 IJCNN2023 会议的相关研究，聚焦视频分类任务，提出了融合多模态信息的分类模型，旨在提升分类准确性与鲁棒性。
相关dataset数据集数据在实验室云服务器fuxiancode中，论文链接如：[Multi-channel Attentive Weighting of Visual Frames for Multimodal Video Classification](#https://ieeexplore.ieee.org/abstract/document/10192036)
## 二、核心模型架构
### 1. 基础模型
基于 GRU 的模型：利用门控循环单元处理时序特征。
ViLT 模型：基于 Vision-Language Transformer 的跨模态基础架构。
### 2. 单模态模型
针对不同模态独立建模：
- 视频 s3d 特征：提取视频时空特征的 3D 卷积特征。
- 文本特征：处理文本语义信息。
- vgg 特征：基于 VGG 网络的视觉特征提取。
### 3. 多模态融合
通过设计融合机制，整合不同模态的互补信息，增强模型对视频内容的理解能力。
### 4. backbone_pp 框架
整合上述模型优势，通过优化设计实现高效的视频分类建模。
## 三、代码功能说明
### 1. 模型定义
VilT_Classification 模型：核心分类模型，包含以下子模块：
model_Linear：线性变换层。
BaseboneModel：基础特征提取骨干网络。
classifier：分类器模块。
ViLTransformerText：文本模态的 Transformer 处理单元。
### 2. 数据处理
实现视频与文本数据的预处理流程，为训练和测试提供标准化输入。
### 3. 训练与测试
训练逻辑：集成优化算法（如 Adam），支持参数调优。
测试评估：在测试集上计算准确率、召回率等指标，验证模型性能。
## 四、使用指南
### 1. 环境配置
需安装以下依赖：
PyTorch：深度学习框架。
PyTorch Lightning：基于 PyTorch 的训练框架。
transformers：自然语言处理工具库。
### 2. 数据准备
按代码要求整理视频分类数据集（如格式、标注规范）。
### 3. 模型运行
主程序：运行 demo.py 启动训练 / 测试。
参数调整：可修改配置项（如 batch_size、learning_rate 等）。

