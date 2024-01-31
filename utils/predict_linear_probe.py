import argparse
import json
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from model1.single import single
from model1.fusion import fusion
from model1.fusionWithCrossAttention import fusionWithCrossAttention
from func import int2cat, set_seed

class MyDataset3(Dataset):
    def __init__(self, vfeats, tfeats):
        self.vfeats = vfeats
        self.tfeats = tfeats

    def __len__(self):
        return len(self.vfeats)

    def __getitem__(self, idx):
        vfeat = self.vfeats[idx]
        tfeat = self.tfeats[idx]
        return torch.FloatTensor(vfeat), torch.FloatTensor(tfeat)

def predict(hyper_dict, test_loader):
    print(hyper_dict)
    print("using {} device.".format(device))
    # 直接加载
    net = torch.load(hyper_dict.model_loc)
    net.eval()
    all_preds = []

    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)

        for val_data in val_bar:
            val_images, val_text = val_data
            outputs = net(val_images.to(device), val_text.to(device))
            preds = torch.argmax(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy().flatten())

    return all_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--vtype', type=str, default='clip', help='resnet or clip')
    parser.add_argument('--ttype', type=str, default='clip', help='roberta or clip')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--model_loc', type=str, default='saved/saved_models/clip.pth')
    args = parser.parse_args()
    # 读取数据集与特征信息
    feats_text = json.load(open('saved/%s_test.json' % args.ttype, 'r'))
    feats_img = json.load(open('saved/%s_test.json' % args.vtype, 'r'))

    feats_text = feats_text['txt_feats']
    feats_img = feats_img['img_feats']

    feats_text = np.array(feats_text)
    feats_img = np.array(feats_img)
    print(len(feats_text), len(feats_img))

    # 构建数据集与dataloader
    test_data = MyDataset3(feats_img, feats_text)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize)

    # 推理
    res = predict(args, test_loader)
    df = pd.read_csv("data/test_without_label.txt")
    df["tag"] = res
    df["tag"] = df["tag"].map(int2cat)
    df.to_csv("output/answer.txt", index=False)