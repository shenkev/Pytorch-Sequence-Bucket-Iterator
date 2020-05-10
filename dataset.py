from torch.utils.data import Dataset
import numpy as np
import torch

class MyDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        
    self.data = [
        np.array(range(10)),
        np.array(range(5)),
        np.array(range(8)),
        np.array(range(10)),
        np.array(range(6)),
        np.array(range(10)),
        np.array(range(2)),
        np.array(range(10)),
        np.array(range(3)),
        np.array(range(7)),
        np.array(range(10)),
        np.array(range(7)),
        np.array(range(10)),
    ]


  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return torch.tensor(self.data[index])