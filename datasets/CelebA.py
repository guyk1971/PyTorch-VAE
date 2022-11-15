
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision import transforms
from torchvision.datasets import CelebA

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    @classmethod
    def get_transform(cls, split,patch_size):
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

