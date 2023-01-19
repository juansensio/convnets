from torch.utils.data import Dataset
from skimage import io 
import numpy as np
from einops import rearrange

class ImageClassificationDataset(Dataset):
    def __init__(self, images, labels=None, trans=None):
        self.images = images 
        self.labels = labels 
        self.trans = trans 

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        im = io.imread(self.images[idx]).astype(np.float32) / 255.
        im = self.apply_transforms(im)
        im = rearrange(im, 'h w c -> c h w')
        if self.labels is not None:
            return im, self.labels[idx]
        return im

    # by default, we assume albumentations
    # to use another implementation, override this method
    def apply_transforms(self, im):
        if self.trans is not None:
            return self.trans(image=im)['image']
        return im