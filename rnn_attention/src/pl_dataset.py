import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from Dataset import SequenceDataset
import numpy as np
import os 
import json
import shutil
import torch 
from torch.utils.data import DataLoader


class SequneceDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 feat_scp_train, 
                 label_train, 
                 feat_scp_dev, 
                 label_dev, 
                 mean_std_file,
                 token_list_path,
                 batch_size=10,
                 num_workers=1):
        
        super(SequneceDataModule, self).__init__()
        
        self.feat_scp_train = feat_scp_train
        self.label_train = label_train
        self.feat_scp_dev = feat_scp_dev
        self.label_dev = label_dev
        
        self.mean_std_file = mean_std_file
        self.token_list_path = token_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feat_mean, self.feat_std = self._load_mean_std(mean_std_file)
        self.token_list, self.num_tokens, self.sos_id, self.eos_id = self._load_token_list(token_list_path)
        
    def _load_mean_std(self, mean_std_file):
        
        """
        特徴量の平均と標準偏差をファイルから読み込む
        """
        
        with open(mean_std_file, 'r') as f:
            lines = f.readlines()
            mean_line = lines[1]
            std_line = lines[3]
            
            feat_mean = np.array(mean_line.split(), dtype=np.float32)
            feat_std = np.array(std_line.split(), dtype=np.float32)
            
        return feat_mean, feat_std
    
    def _load_token_list(self, token_list_path):
        
        token_list = {0 : '<blank'}
        with open(token_list_path, mode='r') as f:
            
            for line in f:
                parts = line.split()
                # トークン名とIDをトークンリストに追加
                token_list[int(parts[1])] = parts[0]
        #<eos>トークンをユニットリストの末部に追加        
        eos_id = len(token_list)
        token_list[eos_id] = '<eos>'
        sos_id = eos_id
        
        num_tokens = len(token_list)
        
        return token_list, num_tokens, sos_id, eos_id
            
    def setup(self, stage=None):
        
        self.train_dataset = SequenceDataset(self.feat_scp_train, 
                                             self.label_train,
                                             self.feat_mean,
                                             self.feat_std)
        
        self.dev_dataset = SequenceDataset(self.feat_scp_dev,
                                           self.label_dev,
                                           self.feat_mean,
                                           self.feat_std)
        
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # CUDAのためのピンメモリを有効にする
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True  # CUDAのためのピンメモリを有効にする
        )
        
