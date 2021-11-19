# encoding: utf-8
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class AnimeData(Dataset):

    def __init__(self, dataRoot, subFold, transform):
        super(AnimeData, self).__init__()

        if not os.path.exists(os.path.join(dataRoot, subFold)):
            raise FileNotFoundError('路径不存在！！！！！')

        self.imgFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.jpg"))
        self.transform = transform
        self.len = len(self.imgFiles)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len

        img = Image.open(self.imgFiles[index]).convert("RGB")
        img = self.transform(img)

        return img
