import collections.abc
import torchvision
import torch
from einops import rearrange

def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

# 5 crop + hflips tta
def tta(self, model, x, size):                                     
    B = x.shape[0]                                                      # B, C, H, W    
    crops = torch.stack(torchvision.transforms.FiveCrop(size)(x))       # 5, B, C, H, W
    flips = torch.stack([torch.flip(crop, (-1,)) for crop in crops])    # 5, B, C, H, W
    x = torch.cat([crops, flips])                                       # 10, B, C, H, W
    x = rearrange(x, 'n b c h w -> (n b) c h w')                        # (10*B), C, H, W
    y = model(x)                                                        # (10*B), N
    y = rearrange(y, '(n b) c -> n b c', b=B)                           # 10, B, N
    return y.mean(dim=0)      