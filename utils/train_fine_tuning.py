import argparse
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import RobertaTokenizer
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from model2.single import single
from model2.fusion import fusion
from preprocess import *
from func import cat2int, set_seed
import warnings
warnings.filterwarnings('ignore')

# 混合精度是否启用
use_amp = True
# 写入一些日志数据，可视化训练情况
writer = SummaryWriter(comment='model_training')
# 可复现性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RANDOM_SEED = 42


class MyDataset2(Dataset):
    def __init__(self, dir, img_transform, txt_transform, tokenizer, labels):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform
        self.tokenizer = tokenizer
        self.txt_transform = txt_transform
        self.labels = np.array(labels).astype(np.int64)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pairs', fname + '.jpg'))
        text = open(os.path.join('data/pairs', fname + '.txt'), 'r', encoding='utf-8', errors='ignore').read().strip()

        if self.img_transform:
            img = self.img_transform(img)

        if self.txt_transform:
            text = self.txt_transform(text)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        label = self.labels[idx]

        return img, encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), torch.tensor(label)


def train(hyper_dict, dloc, save_path="saved/saved_models/"):
    vdim, tdim = 0, 0
    print(hyper_dict)
    print("using {} device.".format(device))

    # 读取数据集与特征信息
    labels = pd.read_csv("data/train.txt")["tag"].map(cat2int).to_numpy().flatten()
    # 构建数据集与dataloader
    tokenizer = RobertaTokenizer.from_pretrained('./tokenizer_roberta')
    train_data = MyDataset2(dloc, img_transforms, text_transform, tokenizer, labels)

    train_size = int((1-hyper_dict.testsize) * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(dataset=train_data, batch_size=hyper_dict.batchsize, num_workers=2, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=hyper_dict.batchsize, num_workers=2)

    # vtype 只能为resnet
    vdim = 1000
    # ttype 只能为roberta
    tdim = 768

    D = hyper_dict.D

    if hyper_dict.fused == 1:
        net = single(tdim, D).to(device)
    elif hyper_dict.fused == 2:
        net = single(vdim, D).to(device)
    else:
        net = fusion(vdim, tdim, D).to(device)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=hyper_dict.lr)
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义训练与验证过程
    epochs = hyper_dict.epochs
    best_acc = 0.0
    train_steps = len(train_loader)
    # fp16设置
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, input_ids, attention_mask, labels = data

            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                if hyper_dict.fused == 1:
                    outputs = net(1, input_ids, attention_mask)
                elif hyper_dict.fused == 2:
                    outputs = net(2, images)
                else:
                    outputs = net(images, input_ids, attention_mask)

                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            writer.add_scalar('training loss', loss, epoch * train_steps + step)

        net.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_input_ids, val_attention_mask, val_labels = val_data

                val_images = val_images.to(device)
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                if hyper_dict.fused == 1:
                    outputs = net(1, val_input_ids, val_attention_mask)
                elif hyper_dict.fused == 2:
                    outputs = net(2, val_images)
                else:
                    outputs = net(val_images, val_input_ids, val_attention_mask)

                preds = torch.argmax(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(val_labels.cpu().numpy().flatten())

        val_accurate = metrics.accuracy_score(all_labels, all_preds)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate

        # 即将结束训练，而且必须是融合模型才能保存
        if hyper_dict.epochs == epoch + 1 and hyper_dict.fused == 0:
            save_path += hyper_dict.vtype + '.pth'
            torch.save(net, save_path)

    print('Finished Training')
    print('Best Accuracy:\t %.3f' % (best_acc))

if __name__ == '__main__':
    # 读取参数设置
    parser = argparse.ArgumentParser(description="training")
    # 这两个不要管
    parser.add_argument('--vtype', type=str, default='resnet', help='only resnet')
    parser.add_argument('--ttype', type=str, default='roberta', help='only roberta')

    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default='1e-4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--fused', type=int, default=0, help='1: no image, 2: no text')
    parser.add_argument('--testsize', type=float, default='0.15')
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    train(args, dloc='data/train.txt')