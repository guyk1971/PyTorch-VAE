import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import *

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
        dataset_name: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
   
        train_transforms = datasets[self.dataset_name].get_transform('train',self.patch_size)
        val_transforms = datasets[self.dataset_name].get_transform('test',self.patch_size)
        self.train_dataset = datasets[self.dataset_name](
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = datasets[self.dataset_name](
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

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
     