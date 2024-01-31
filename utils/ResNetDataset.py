import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ResNetDataset(Dataset):
    def __init__(self, dir, img_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.img_transform = img_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join('data/pairs', fname + '.jpg'))

        if self.img_transform:
            img = self.img_transform(img)

        return img, fname
