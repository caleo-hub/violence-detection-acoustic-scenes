import numpy as np
import h5py
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_files, exclude_classes=None, transform=None, n_splits=5):
        self.h5_files = h5_files
        self.transform = transform
        self.n_splits = n_splits

        if exclude_classes is None:
            exclude_classes = []

        # Armazena todas as chaves de todos os arquivos
        self.keys = []
        self.labels = []
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                for key in f["features"].keys():
                    label = int(f["annotations"][key][()])
                    if label not in exclude_classes:
                        self.keys.append((h5_file, key))
                        self.labels.append(label)

        self.skf = StratifiedKFold(n_splits=n_splits)

    def __getitem__(self, index):
        h5_file, key = self.keys[index]
        with h5py.File(h5_file, "r") as f:
            sample = f["features"][key][()].astype(np.float32)  # Convert to float
            label = int(f["annotations"][key][()])
            
            
        if self.transform:
            sample = self.transform(sample)

        return sample, label


    def __len__(self):
        return len(self.keys)

    def get_k_fold_data_loaders(self, batch_size=32, num_workers=4):
        data_loaders = []
        for train_index, val_index in self.skf.split(self.keys, self.labels):
            train_dataset = Subset(self, train_index)
            val_dataset = Subset(self, val_index)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            data_loaders.append((train_loader, val_loader))
        return data_loaders
