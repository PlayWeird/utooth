"""
Extended volume dataloader with K-fold cross validation support
"""

from typing import Optional, List
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import glob
import ct_utils
from volume_dataloader import CTDataSet


class CTScanDataModuleKFold(pl.LightningDataModule):
    """Data module with k-fold cross validation support"""
    
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 5, 
                 num_workers: int = 6, train_indices: List[int] = None, 
                 val_indices: List[int] = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.ct_train = None
        self.ct_val = None
        self.ct_test = None

    def setup(self, stage: Optional[str] = None):
        # Create full dataset
        full_dataset = CTDataSet(self.data_dir)
        
        if self.train_indices is not None and self.val_indices is not None:
            # Use provided fold indices
            self.ct_train = Subset(full_dataset, self.train_indices)
            self.ct_val = Subset(full_dataset, self.val_indices)
            self.ct_test = None  # No test set in k-fold CV
        else:
            # Fall back to original behavior (90/10 split)
            ct_len = len(full_dataset)
            portions = [0.9, 0.1, 0.0]
            sizes = [int(portion * ct_len) for portion in portions]
            if sum(sizes) != ct_len:
                sizes[0] += ct_len - sum(sizes)
            
            from torch.utils.data import random_split
            self.ct_train, self.ct_val, self.ct_test = random_split(full_dataset, sizes)

    def train_dataloader(self):
        return DataLoader(self.ct_train, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ct_val, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if self.ct_test is None:
            return None
        return DataLoader(self.ct_test, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False)