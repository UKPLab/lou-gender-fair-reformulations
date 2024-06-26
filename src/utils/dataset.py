import torch
from torch.utils.data import Dataset

class FineTuningDataset(Dataset):
    def __init__(self, samples):
        self.x = list(samples["input_ids"])
        self.y = list(samples["label"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {
            "input_ids": torch.tensor(self.x[index]),
            "label": torch.tensor(self.y[index])
        }
