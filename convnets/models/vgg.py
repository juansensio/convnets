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
    vgg11 = {
        'd': 0.5, 
        'bn': False, 
        'l': [
            {'r': 1, 'f': 64},
            {'r': 1, 'f': 128},
            {'r': 2, 'f': 256},
            {'r': 2, 'f': 512},
            {'r': 2, 'f': 512}
        ]   
    }        
    vgg13 = {
        'd': 0.5, 
        'bn': False, 
        'l': [
            {'r': 2, 'f': 64},
            {'r': 2, 'f': 128},
            {'r': 2, 'f': 256},
            {'r': 2, 'f': 512},
            {'r': 2, 'f': 512},
        ]
    }
    vgg16 = {
        'd': 0.5, 
        'bn': False, 
        'l': [
            {'r': 2, 'f': 64},
            {'r': 2, 'f': 128},
            {'r': 3, 'f': 256},
            {'r': 3, 'f': 512},
            {'r': 3, 'f': 512},
        ]
    }
    vgg19 = {
        'd': 0.5, 
        'bn': False, 
        'l': [
            {'r': 2, 'f': 64},
            {'r': 2, 'f': 128},
            {'r': 4, 'f': 256},
            {'r': 4, 'f': 512},
            {'r': 4, 'f': 512},
        ]
    }

class VGG(nn.Module):
    def __init__(self, conf, features_only=False, **kwargs):
        super(VGG, self).__init__()
        block = lambda r,f1,f2,bn=False: S(*[S(C(f1 if i == 0 else f2,f2,3,1,1),R(),BN(f2) if bn else I) for i in range(r)],M(2,2))
        self.backbone = nn.ModuleList([
            block(c['r'], conf['l'][i-1]['f'] if i > 0 else 3, c['f'], bn=conf['bn']) 
            for i, c in enumerate(conf['l'])
        ])
        self.features_only = features_only
        if not features_only:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(7, 7)), # makes it work with any input size
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=conf['d']),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=conf['d']),
                nn.Linear(4096, 1000),
            )

    def forward(self, x):
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        if self.features_only:
            return features
        return self.head(features[-1])