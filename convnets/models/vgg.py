import torch.nn as nn 
from dataclasses import dataclass
from torch.nn import Sequential as S 
from torch.nn import Conv2d as C
from torch.nn import ReLU as R 
from torch.nn import MaxPool2d as M
from torch.nn import BatchNorm2d as BN
from torch.nn import Identity as I

@dataclass
class VGGConfig:
    vgg11 = [
        {'repeat': 1, 'filters': 64},
        {'repeat': 1, 'filters': 128},
        {'repeat': 2, 'filters': 256},
        {'repeat': 2, 'filters': 512},
        {'repeat': 2, 'filters': 512},
    ]
    vgg13 = [
        {'repeat': 2, 'filters': 64},
        {'repeat': 2, 'filters': 128},
        {'repeat': 2, 'filters': 256},
        {'repeat': 2, 'filters': 512},
        {'repeat': 2, 'filters': 512},
    ]
    vgg16 = [
        {'repeat': 2, 'filters': 64},
        {'repeat': 2, 'filters': 128},
        {'repeat': 3, 'filters': 256},
        {'repeat': 3, 'filters': 512},
        {'repeat': 3, 'filters': 512},
    ]
    vgg19 = [
        {'repeat': 2, 'filters': 64},
        {'repeat': 2, 'filters': 128},
        {'repeat': 4, 'filters': 256},
        {'repeat': 4, 'filters': 512},
        {'repeat': 4, 'filters': 512},
    ]

def vgg_block(r, f1, f2, bn=False):
    return S(*[S(C(f1 if i == 0 else f2,f2,3,1,1),R(),BN(f2) if bn else I) for i in range(r)],M(2,2))

class VGG(nn.Module):
    def __init__(self, conf, dropout=0.5, batch_norm=False, features_only=False):
        super(VGG, self).__init__()
        self.backbone = nn.ModuleList([
            vgg_block(c['repeat'], conf[i-1]['filters'] if i > 0 else 3, c['filters'], bn=batch_norm) 
            for i, c in enumerate(conf)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(7, 7)), # makes it work with any input size
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1000),
        )
        self.features_only = features_only

    def forward(self, x):
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        if self.features_only:
            return features
        return self.head(features[-1])