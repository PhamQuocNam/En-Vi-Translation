from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import os
class MyDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor( self.input_ids[idx]), torch.tensor(self.labels[idx])
    
    

def get_dataloader(train_ds, val_ds, test_ds, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_data(file_path):
    train_df = pd.read_csv(os.path.join(file_path,'train.csv'))
    val_df = pd.read_csv(os.path.join(file_path,'valid.csv'))
    test_df = pd.read_csv(os.path.join(file_path,'test.csv'))
    return {
        'train': train_df,
        'valid': val_df,
        'test': test_df
    }