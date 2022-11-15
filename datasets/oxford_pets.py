from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms




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

    @classmethod
    def get_OxfordPets_transform(cls,split,patch_size):
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
