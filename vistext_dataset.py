from torch.utils.data import Dataset
import numpy as np
import torch
import os

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class CustomDataset(Dataset):
    def __init__(self, pixel_values_path, input_features_path, labels_path):
        self.pixel_values_path = pixel_values_path
        self.pixel_values_paths = [os.path.join(self.pixel_values_path, f) for f in os.listdir(self.pixel_values_path) if f.endswith('.npy')]
        self.pixel_values_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.input_features_path = input_features_path
        self.input_features_paths = [os.path.join(self.input_features_path, f) for f in os.listdir(self.input_features_path) if f.endswith('.pth')]
        self.input_features_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.labels_path = labels_path
        self.labels_paths = [os.path.join(self.labels_path, f) for f in os.listdir(self.labels_path) if f.endswith('.pth')]
        self.labels_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    def __len__(self):
        return len(self.labels_paths)
    
    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.pixel_values_paths[idx])), torch.load(self.input_features_paths[idx]), torch.load(self.labels_paths[idx])