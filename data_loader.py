import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from __settings__ import *


class SFEWDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_src = 'images/' + self.df.iloc[index, 0][:-4] + '.png'

        image = Image.open(image_src)
        image = np.array(image)[:, :, :3]

        label = self.df.iloc[index, 1] - 1

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        return image, torch.LongTensor([label])


df = pd.read_excel('SFEW.xlsx')
df = df.sample(frac=1.0, random_state=random_seed)


transforms_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.9, 1.0), p=1.0),
    A.Equalize(p=0.5),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_eval = A.Compose([
    A.Resize(height=img_size, width=img_size, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])


train_index = int(len(df) * train_portion)
valid_index = train_index + int(len(df) * valid_portion)
train = df.iloc[:train_index]
valid = df.iloc[train_index:valid_index]
test = df.iloc[valid_index:]

dataset_train = SFEWDataset(train, transforms=transforms_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_valid = SFEWDataset(valid, transforms=transforms_eval)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

dataset_test = SFEWDataset(valid, transforms=transforms_eval)
dataloader_test = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

dataset = (dataloader_train, dataloader_valid, dataloader_test)
num_classes = 7