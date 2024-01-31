import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ClipDataset(Dataset):
    def __init__(self, dir, img_transform=None, txt_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pairs', fname + '.jpg'))
        text = open(os.path.join('data/pairs', fname + '.txt'), 'r', encoding='utf-8', errors='ignore').read().strip()
        tag = str(self.file_names.iloc[idx, 1])

        if self.img_transform:
            img = self.img_transform(img)

        if self.txt_transform:
            text = self.txt_transform(text)

        return img, text, tag