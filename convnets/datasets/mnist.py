import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch 

class MNIST(Dataset):
    def __init__(self, path, train, trans=None, **kwargs):
        dataset = torchvision.datasets.MNIST(path, train=train, download=True)
        self.X, self.y = [], []
        for x, y in dataset:
            self.X.append(np.asarray(x))
            self.y.append(int(y))
        self.X = np.stack(self.X).astype(np.float32) / 255.
        self.y = np.array(self.y, dtype=int)
        self.trans = trans
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ix):
        x, y = self.X[ix], self.y[ix]
        if self.trans is not None:
            x = self.trans(image=x)['image']
        return torch.from_numpy(x).unsqueeze(0), y