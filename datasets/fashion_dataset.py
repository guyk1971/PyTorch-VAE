import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile
from torchvision.io import read_image
from glob import glob as Glob
import numpy as np


class FashionDataset(Dataset):
    '''
    FashionDataset dataset
    URL: 
    '''
    
    def __init__(self, dataset_path, split='train',transform=None,**kwargs):
        self.files=[i for i in Glob(f'{dataset_path}/Img/**/*.jpg',recursive=True)] 
        # read item attributes:
        # the attribute can be the cloth attribute or it can be viewing angle or type of item (based on the folder)
        # in the following example, we're taking the cloth attributes (463 attributes)
        anno_filename = os.path.join(dataset_path,'Anno','attributes','list_attr_items.txt')
        with open(anno_filename,'r') as f:
            attr_items=f.read().splitlines()
        attr_items=attr_items[2:]
        attr_items=[i.split() for i in attr_items]
        self.attr_items = {i[0]:[int(int(a)>0) for a in i[1:]] for i in attr_items}
        # get attributes names
        anno_filename = os.path.join(dataset_path,'Anno','attributes','list_attr_cloth.txt')
        with open(anno_filename,'r') as f:
            attr_cloth=f.read().splitlines()
        self.attr_names = attr_cloth[2:]

        # split the dataset and throw the irrelevant part
        train_fraction=0.8
        train_size = int(train_fraction* len(self.attr_items.keys()))

        if split=='train':
            self.files = self.files[:train_size]
        else:
            self.files = self.files[train_size:]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        fpath = self.files[ix]
        lbl=self.attr_items[fpath.split('/')[-2]]
        img = read_image(fpath)/255.0
        return img,lbl

    @classmethod
    def get_transform(cls,split,patch_size):
        '''
        create the transform to be performed on the dataset samples
        arguments:
            - split : whether 'train' or 'val'
            - patch_size : as dictated by the VAEDataset class
        '''
        if split=='train':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(patch_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
                #                     std=[0.2332, 0.2500, 0.2564]),
                                                        ])
        elif (split=='val'):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(patch_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
                #                     std=[0.2332, 0.2500, 0.2564]),
                                                        ])
        elif (split=='test'):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(patch_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
                #                     std=[0.2332, 0.2500, 0.2564]),
                                                        ])
        else: 
            transform=None 
        return transform


    def choose(self):
        return self[np.random.randint(len(self))]


    def collate_fn(self, batch):
        imgs, attrs = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in imgs]
        else:
            imgs = [img[None] for img in imgs]
        attrs=[torch.tensor(a)[None] for a in attrs]
        imgs,attrs = [torch.cat(i) for i in [imgs,attrs]]
        return imgs,attrs

