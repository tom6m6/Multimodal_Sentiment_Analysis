# 当代人工智能实验五：多模态情感分析

10211900416 郭夏辉

**给定配对的文本和图像，预测对应的情感标签。**三分类任务：positive, neutral, negative。我要设计一个多模态融合模型，这个模型能够整合输入图片和文本数据输出其情感标签。在此模型设计出来之后，我要自行从训练集中划分验证集、调整超参数，然后在测试集上预测情感标签。

## 部署

项目运行环境：

- Ubuntu 22.04 64位
- NVIDIA驱动版本：535.104.05 
- CUDA Version: 12.2 
- Python 3.11.5
- GPU: V100S

你需要安装如下依赖：

- torch==2.1.2
- torchaudio==2.1.2
- torchvision==0.16.2
- ekphrasis==0.5.4
- ftfy==6.1.3
- grpcio==1.60.0
- accelerate==0.25.0
- transformers==4.32.1
- tqdm==4.65.0
- ftfy==6.1.3
- regex
- clip 安装clip需要到github中下载 即` pip install git+https://github.com/openai/CLIP.git`
- Pillow==9.4.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- tensorboard==2.15.1

当然你也可以，进入目录，输入以下命令快速安装依赖：

```
pip install --upgrade pip 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

## 项目结构

```bash
├── data #实验数据集
│   ├── pairs #实验五数据.zip解压后的图像-文本pairs
│   │   ├── 1.jpg
│   │   ├── 1.txt
│   │   ├── 2.jpg
│   │   ├── 2.txt
│   │   └── ......
│   ├── test_without_label.txt #测试集数据，只有标签
│   ├── train1.txt # 随机cut后的新训练数据
│   └── train.txt # 训练数据，带有标签
├── download.py # 下载预训练模型到本地
├── model_roberta # roberta模型，需要download
│   ├── config.json
│   └── model.safetensors
├── output # 我的实验输出
│   ├── answer1.txt # fine tuning resnet+roberta的结果，仅供参考
│   ├── answer.txt # linear probe CLIP的结果，作为最终答案
│   └── tmp_no_open_check.txt # 检查模型误判时生成的文件
├── play.ipynb # 用于展现训练过程
├── README.md # 介绍
├── report.pdf # 实验报告
├── requirements.txt # 依赖库
├── runs # tensorboard训练数据日志文件
│   ├── board.ipynb # 方便在云主机打开tensorborad
|   ├──Jan31_13-45-12_10-23-167-72model_training
|   |  └──events.out.tfevents.1706679912.10-23-167-72.2694.0
│   └── ......
├── saved # 保存的模型及特征
│   ├── clip_test.json # clip对测试集提取特征
│   ├── clip_train.json # clip对训练数据提取特征
│   ├── resnet_test.json # resnet对测试集提取特征
│   ├── resnet_train.json # resnet对训练数据提取特征
│   ├── roberta_test.json # roberta对测试集提取特征
│   ├── roberta_train.json # roberta对训练数据提取特征
│   └── saved_models # 训练后保存的模型
│       ├── clip.pth # clip
│       └── resnet.pth # resnet
├── tokenizer_roberta # roberta tokenizer，需要download
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
└──	utils # 项目主体目录
    ├── check.py # 误判检验实验
    ├── ClipDataset.py 
    ├── func.py # 一些常用的功能
    ├── __init__.py
    ├── model1 # linear probe类模型
    │   ├── fusion.py # 多模态 直接拼接
    │   ├── fusionWithCrossAttention.py # 多模态 交叉注意力
    │   └── single.py # 单模态
    ├── model2 # fine tuning类模型
    │   ├── fusion.py # 多模态
    │   └── single.py # 单模态
    ├── predict_fine_tuning.py # fine_tuning的预测部分
    ├── predict_linear_probe.py # linear_probe的预测部分
    ├── preprocess1.py # cut实验数据预处理+特征提取部分
    ├── preprocess.py # 主实验数据预处理+特征提取部分
    ├── ResNetDataset.py
    ├── RobertaDataset.py
    ├── train_fine_tuning.py # fine_tuning的训练部分
    ├── train_linear_probe1.py # cut实验linear_probe的训练部分
    └── train_linear_probe.py # linear_probe的训练部分
```

我知道你很纳闷为什么我主体的功能都放在了util目录下面而不是单独在外面搞一个main.py，这样确实显得有点头重脚轻，但我懒得再单独抽出来了，功能到位即可。

## 项目运行

1. 进入项目文件夹，下载Huggingface来源的预训练模型，这里需要保证你能顺利连接国际互联网：

```
python download.py
```

当然如果你之前已经下载好了对应的model和tokenizer可以跳过这一步

2. 数据预处理及生成文本和图像的embedding

```
python utils/preprocess.py 
```

3. linear probe方式进行训练

```
python utils/train_linear_probe.py --vtype resnet --ttype roberta --D 128 --lr 1e-4 --batchsize 128 --epochs 30
```

参数介绍：

- --vtype	图像embedding	默认为clip	clip,resnet可选
- --ttype	 文本embedding	默认为clip	clip,roberta可选
- --batchsize	批大小	默认为64
- --lr	学习率	默认为1e-4
- --epochs	训练轮数	默认为30
- --D	分类头大小	默认为128
- --fused	消融实验	默认为0（多模态）可以为1（text only）或2（image only）
- --testsize	验证集的比例，默认为0.15

经过一番训练，若是多模态融合模型，最后保存到了saved/saved_models


3. fine tuning方式进行训练

```
python utils/train_fine_tuning.py --D 128 --lr 1e-4 --batchsize 128 --epochs 30
```

参数介绍：

- --batchsize	批大小	默认为64
- --lr	学习率	默认为1e-4
- --epochs	训练轮数	默认为30
- --D	分类头大小	默认为128
- --fused	消融实验	默认为0（多模态）可以为1（text only）或2（image only）
- --testsize	验证集的比例，默认为0.15


4. linear head方式进行预测

```
python utils/predict_linear_probe.py --vtype resnet --ttype roberta --batchsize 128 --model_loc saved/saved_models/clip.pth
```

参数介绍：

- --vtype	图像embedding	默认为clip	clip,resnet可选
- --ttype	 文本embedding	默认为clip	clip,roberta可选
- --batchsize	批大小	默认为1024
- --model_loc	用于预测的模型的位置，默认为saved/saved_models/clip.pth

5. fine tune方式进行预测

```
python utils/predict_linear_probe.py --batchsize 128
```

参数介绍：

- --batchsize	批大小	默认为1024
- --model_loc	用于预测的模型的位置，默认为saved/saved_models/resnet.pth

## 实验结果

详见报告

## 引用&参考

[1]: Transformer架构的整体指南: https://luxiangdong.com/2023/09/10/trans/#/

[2]: Zi-Yi Dou, et al. An Empirical Study of Training End-to-End Vision-and-Language Transformers [J]. arXiv preprint arXiv: 2111.02387

[3]: Dosovitskiy, Alexey, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[4]: 狗都能看懂的Vision Transformer的讲解和代码实现vision transformer代码-CSDN博客: https://blog.csdn.net/weixin_42392454/article/details/122667271

[5]: ViT论文逐段精读【论文精读】: https://www.youtube.com/watch?v=FRFt3x0bO94

[6]: Liu, Yinhan, et al. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

[7]: Kaiming He, et al: Deep residual learning for image recognition, 2016. arXiv:1512.03385

[8]: 论文精读: https://www.youtube.com/playlist?list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I

[9]: CLIP 论文逐段精读【论文精读】: https://www.youtube.com/watch?v=OZF1t_Hieq8

[10]:  Alec Radford, et al. Learning transferable visual models from natural language supervision. International Conference on Machine Learning. PMLR, 2021. arXiv:2103.00020

[11]: Why Pytorch officially use mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] to normalize images?：  https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

[12]: Cheema, Gullal S., et al. A fair and comprehensive comparison of multimodal tweet sentiment analysis methods [J]. Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding. 2021.

[13]: 深度学习基础系列（九）| Dropout VS Batch Normalization? 是时候放弃Dropout了 - 可可心心 - 博客园: https://www.cnblogs.com/hutao722/p/9946047.html

[14]: Sergey Ioffe, et al. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv:1502.03167

[15]: Sheng Shen, et al. PowerNorm: Rethinking Batch Normalization in Transformers. arXiv: 2003.07845

[16]: https://github.com/cleopatra-itn/fair_multimodal_sentiment
