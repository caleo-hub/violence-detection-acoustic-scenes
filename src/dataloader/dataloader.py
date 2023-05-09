from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import h5py


class H5Dataset(Dataset):
    def __init__(self, h5_files, transform=None):
        self.h5_files = h5_files
        self.transform = transform

        # Armazena todas as chaves de todos os arquivos
        self.keys = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                self.keys.extend([(h5_file, key) for key in f['features'].keys()])

    def __getitem__(self, index):
        h5_file, key = self.keys[index]
        with h5py.File(h5_file, 'r') as f:
            sample = f['features'][key][()].astype(np.float32)  # Convert to float
            label = f['annotations'][key][()].astype(np.long)  # Convert to long
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

    def __len__(self):
        return len(self.keys)
