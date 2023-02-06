import torch.nn as nn 
import torch 
import torchvision
from einops import rearrange

# only works for 224x224x3 inputs !

class Alexnet(nn.Module):
    def __init__(self, config=None, **kwargs):
        super(Alexnet, self).__init__()
        use_bn = config['batch_norm'] if config is not None and 'batch_norm' in config else False
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96) if use_bn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384) if use_bn else nn.Identity(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384) if use_bn else nn.Identity(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ) 
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096) if use_bn else nn.Identity(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096) if use_bn else nn.Identity(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
    
    def tta(self, x):
        B = x.shape[0]                                                      # B, C, H, W    
        crops = torch.stack(torchvision.transforms.FiveCrop(224)(x))        # 5, B, C, H, W
        flips = torch.stack([torch.flip(crop, (-1,)) for crop in crops])    # 5, B, C, H, W
        x = torch.cat([crops, flips])                                       # 10, B, C, H, W
        x = rearrange(x, 'n b c h w -> (n b) c h w')                        # (10*B), C, H, W
        y = self.forward(x)                                                 # (10*B), N
        y = rearrange(y, '(n b) c -> n b c', b=B)                           # 10, B, N
        return y.mean(dim=0)                                                # B, N
    # paper repots top-1 and top-5 test error rate of 40.7% and 18.2% en imagenet 2012 validation set