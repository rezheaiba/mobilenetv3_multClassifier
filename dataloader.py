"""
# @Time    : 2023/2/8 13:43
# @File    : dataloader.py
# @Author  : rezheaiba
"""
import csv
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Loader(Dataset):
    def __init__(self, csv_path, root, transforms_=None, mode="train"):
        self.name = []
        self.label1 = []
        self.label2 = []

        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                self.name.append(row[0])
                self.label1.append(int(row[1]))
                self.label2.append(int(row[2]))

        self.root = root
        self.mode = mode
        if transforms_ is not None:
            self.transform = transforms_

    def __getitem__(self, index):
        dir = self.name[index].split('_')[0]
        path = os.path.join(self.root, self.mode, dir, self.name[index])
        img = Image.open(path).convert("RGB")
        label1 = self.label1[index]
        label2 = self.label2[index]


        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "label1": torch.tensor(label1, dtype=torch.int64),
                "label2": torch.tensor(label2, dtype=torch.int64),

                }

    def __len__(self):
        return len(self.name)
