import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from model1.single import single
from model1.fusion import fusion
from model1.fusionWithCrossAttention import fusionWithCrossAttention
from func import cat2int, set_seed

# 混合精度是否启用
use_amp = True
# 写入一些日志数据，可视化训练情况
writer = SummaryWriter(comment='model_training')
# 可复现性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
RANDOM_SEED = 42

class MyDataset1(Dataset):
    def __init__(self, vfeats, tfeats, labels):
        self.vfeats = vfeats
        self.tfeats = tfeats
        self.labels = np.array(labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vfeat = self.vfeats[idx]
        tfeat = self.tfeats[idx]
        label = self.labels[idx]
        return torch.FloatTensor(vfeat), torch.FloatTensor(tfeat), torch.tensor(label)


def train(hyper_dict, train_loader, val_loader, save_path = "saved/saved_models/"):
    vdim, tdim = 0, 0
    print(hyper_dict)
    print("using {} device.".format(device))

    if hyper_dict.vtype == "clip":
        vdim = 512
    elif hyper_dict.vtype == "resnet":
        vdim = 1000

    if hyper_dict.ttype == "clip":
        tdim = 512
    elif hyper_dict.ttype == "roberta":
        tdim = 768

    D = hyper_dict.D

    if hyper_dict.fused == 1:
        net = single(tdim, D).to(device)
    elif hyper_dict.fused == 2:
        net = single(vdim, D).to(device)
    else:
        #net = fusion(vdim, tdim, D).to(device)
        net = fusionWithCrossAttention(vdim, tdim, D).to(device)

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
            images, texts, labels = data
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                if hyper_dict.fused == 1:
                    outputs = net(texts.to(device))
                elif hyper_dict.fused == 2:
                    outputs = net(images.to(device))
                else:
                    outputs = net(images.to(device), texts.to(device))

                loss = loss_function(outputs, labels.to(device))

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
                val_images, val_text, val_labels = val_data

                if hyper_dict.fused == 1:
                    outputs = net(val_text.to(device))
                elif hyper_dict.fused == 2:
                    outputs = net(val_images.to(device))
                else:
                    outputs = net(val_images.to(device), val_text.to(device))

                preds = torch.argmax(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(val_labels.cpu().numpy().flatten())

        val_accurate = metrics.accuracy_score(all_labels, all_preds)
        print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

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
    parser.add_argument('--vtype', type=str, default='clip', help='resnet or clip')
    parser.add_argument('--ttype', type=str, default='clip', help='roberta or clip')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default='1e-4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--fused', type=int, default=0, help='1: no image, 2: no text')
    parser.add_argument('--testsize', type=float, default='0.15')
    args = parser.parse_args()
    set_seed(RANDOM_SEED)

    labels = pd.read_csv("data/train.txt")["tag"].map(cat2int).to_numpy().flatten()
    feats_text = json.load(open('saved/%s_train.json' % args.ttype, 'r'))
    feats_img = json.load(open('saved/%s_train.json' % args.vtype, 'r'))

    feats_text = feats_text['txt_feats']
    feats_img = feats_img['img_feats']
    feats_text = np.array(feats_text)
    feats_img = np.array(feats_img)
    print(len(feats_text), len(feats_img))

    # 获取训练集/验证集/测试集分割
    label_train, label_val, feats_train_txt, feats_val_txt, feats_train_img, feats_val_img = train_test_split(labels, feats_text, feats_img, test_size=args.testsize, random_state=RANDOM_SEED, shuffle=True)
    label_train = list(map(int, label_train))
    label_val = list(map(int, label_val))

    # 构建数据集与dataloader
    train_data = MyDataset1(feats_train_img, feats_train_txt, label_train)
    val_data = MyDataset1(feats_val_img, feats_val_txt, label_val)
    #all_data = MyDataset1(feats_img, feats_text, labels)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batchsize)
    
    #all_loader = DataLoader(dataset=all_data, batch_size=args.batchsize)

    train(args, train_loader, val_loader)