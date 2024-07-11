import os
import time
import numpy as np
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    d = {'a': torch.tensor([1., 2.]).to(device), 'b': torch.tensor([3., 4.]).to(device)}
    torch.save(d, 'tensor_dict.pt')
    # torch.load('tensor_dict.pt')
