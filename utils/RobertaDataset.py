import os
import pandas as pd
from torch.utils.data import Dataset

class RobertaDataset(Dataset):
    def __init__(self, dir, txt_transform=None):
        self.file_names = pd.read_csv(dir)
        self.dir = dir
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])

        text = open(os.path.join('data/pairs', fname + '.txt'), 'r', encoding='utf-8', errors='ignore').read().strip()

        if self.txt_transform:
            text = self.txt_transform(text)

        return text, fname