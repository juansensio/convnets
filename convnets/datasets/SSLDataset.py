from torch.utils.data import Dataset
from skimage import io 
import numpy as np
from einops import rearrange

class SSLDataset(Dataset):
    def __init__(self, images, bands, trans=None):
        self.images = images 
        self.trans = trans 
        self.bands = bands

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        bands = []
        for band in self.bands:
            bands.append(io.imread(f'{self.images[idx]}/{band}.tif'))
        im = np.stack(bands, axis=-1).astype(np.float32) / 255.
        im1 = self.apply_transforms(im)
        im1 = rearrange(im1, 'h w c -> c h w')
        im2 = self.apply_transforms(im)
        im2 = rearrange(im2, 'h w c -> c h w')
        return im1, im2

    # by default, we assume albumentations
    # to use another implementation, override this method
    def apply_transforms(self, im):
        if self.trans is not None:
            return self.trans(image=im)['image']
        return im