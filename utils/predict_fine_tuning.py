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
from func import int2cat, set_seed
import warnings
warnings.filterwarnings('ignore')

class MyDataset4(Dataset):
    def __init__(self, dir, img_transform, txt_transform, tokenizer):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform
        self.tokenizer = tokenizer
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pairs', fname + '.jpg'))
        text = open(os.path.join('data/pairs', fname + '.txt'), 'r', encoding='utf-8',
                    errors='ignore').read().strip()

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


        return img, encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()


def predict(hyper_dict, dloc):
    print(hyper_dict)
    print("using {} device.".format(device))
    net = torch.load(hyper_dict.model_loc)

    # 构建数据集与dataloader
    tokenizer = RobertaTokenizer.from_pretrained('./tokenizer_roberta')
    test_data = MyDataset4(dloc, img_transforms, text_transform, tokenizer)
    test_loader = DataLoader(dataset=test_data, batch_size=hyper_dict.batchsize, num_workers=2)

    net.eval()
    all_preds = []
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_input_ids, val_attention_mask = val_data

            val_images = val_images.to(device)
            val_input_ids = val_input_ids.to(device)
            val_attention_mask = val_attention_mask.to(device)
            outputs = net(val_images, val_input_ids, val_attention_mask)
            preds = torch.argmax(outputs.data, 1)

            all_preds.extend(preds.cpu().numpy().flatten())
    
    return all_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="predict")
    # 这两项不用管
    parser.add_argument('--vtype', type=str, default='resnet')
    parser.add_argument('--ttype', type=str, default='roberta')

    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--model_loc', type=str, default='saved/saved_models/resnet.pth', help='saved models\' location')
    args = parser.parse_args()

    # 推理
    res = predict(args, dloc='data/test_without_label.txt')
    df = pd.read_csv("data/test_without_label.txt")
    df["tag"] = res
    df["tag"] = df["tag"].map(int2cat)
    df.to_csv("output/answer1.txt", index=False)