from numpy_sampler import BySequenceLengthSampler
from torch.utils.data import DataLoader
from dataset import MyDataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

np.random.seed(0)
torch.manual_seed(0)


bucket_boundaries = [1, 4, 7, 10]
batch_sizes=32

my_data = MyDataset()

sampler = BySequenceLengthSampler(my_data, bucket_boundaries, 2)

# for x in sampler:
#     print(x)

def collate(examples):
    # examples = [torch.tensor(e) for e in examples]
    return pad_sequence(examples, batch_first=True, padding_value=-1)

dataloader = DataLoader(my_data, batch_size=1, 
                        batch_sampler=sampler, 
                        num_workers=0, 
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)


for batch in dataloader:
    print(batch)

