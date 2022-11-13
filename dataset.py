import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from torchvision.io import read_image
from glob import glob as Glob
import numpy as np


# Add your custom dataset class here

def get_MyDataset_transform(split,patch_size):
    '''
    create the transform to be performed on the dataset samples
    arguments:
        - split : whether 'train' or 'val'
        - patch_size : as dictated by the VAEDataset class
    '''
    if split=='train':
        MyDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    elif (split=='val'):
        MyDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    elif (split=='test'):
        MyDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    else: 
        MyDataset_transform=None 
    return MyDataset_transform

class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


#----------------------------------------------------
def get_DeepFashion_transform(split,patch_size):
    '''
    create the transform to be performed on the dataset samples
    arguments:
        - split : whether 'train' or 'val'
        - patch_size : as dictated by the VAEDataset class
    '''
    if split=='train':
        DeepFashion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
            #                     std=[0.2332, 0.2500, 0.2564]),
                                                    ])
    elif (split=='val'):
        DeepFashion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
            #                     std=[0.2332, 0.2500, 0.2564]),
                                                    ])
    elif (split=='test'):
        DeepFashion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.8323, 0.8108, 0.8040], 
            #                     std=[0.2332, 0.2500, 0.2564]),
                                                    ])
    else: 
        DeepFashion_transform=None 
    return DeepFashion_transform

class DeepFashion(Dataset):
    """DeepFashion dataset."""

    def __init__(self, dataset_path, split='train',transform=None):
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



#----------------------------------------------------
def get_CelebA_transform(split,patch_size):
    '''
    create the transform to be performed on the dataset samples
    arguments:
        - split : whether 'train' or 'val'
        - patch_size : as dictated by the VAEDataset class
    '''
    if split=='train':
        CelebA_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    elif (split=='val'):
        CelebA_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    elif (split=='test'):
        CelebA_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),])
    else: 
        CelebA_transform=None 
    return CelebA_transform


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    
#----------------------------------------------------
def get_OxfordPets_transform(split,patch_size):
    '''
    create the transform to be performed on the dataset samples
    arguments:
        - split : whether 'train' or 'val'
        - patch_size : as dictated by the VAEDataset class
    '''
    if split=='train':
        OxfordPets_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(patch_size),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
    elif (split=='val'):
        OxfordPets_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(patch_size),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    elif (split=='test'):
        OxfordPets_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(patch_size),
                                                transforms.Resize(patch_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else: 
        OxfordPets_transform=None 
    return OxfordPets_transform


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

#===================================================================================
class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
        # train_transforms = get_OxfordPets_transform('train',self.patch_size)
        # val_transforms = get_OxfordPets_transform('val',self.patch_size)

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = get_CelebA_transform('train',self.patch_size)
        val_transforms = get_CelebA_transform('test',self.patch_size)
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       =========================  DeepFashion Dataset  =========================
        # train_transforms = None
        # val_transforms = None
        # self.train_dataset = DeepFashion(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = DeepFashion(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        # )

#       ===============================================================

    def train_dataloader(self) -> DataLoader:
        collate_fn = getattr(self.train_dataset,'collate_fn',None)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = getattr(self.val_dataset,'collate_fn',None)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = getattr(self.val_dataset,'collate_fn',None)
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
     