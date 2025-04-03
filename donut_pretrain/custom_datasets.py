from torch.utils.data import Dataset

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class CustomDataset(Dataset):
    def __init__(self, pixel_values, labels, target_sequences):
        self.pixel_values = pixel_values
        self.labels = labels
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.target_sequences)
    
    def __getitem__(self, idx):
        return self.pixel_values[idx], self.labels[idx], self.target_sequences[idx]