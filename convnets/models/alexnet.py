import torch.nn as nn 
from torch.nn import Sequential as S 
from torch.nn import Conv2d as C
from torch.nn import ReLU as R 
from torch.nn import MaxPool2d as M
from torch.nn import BatchNorm2d as BN
from torch.nn import BatchNorm1d as BN1
from torch.nn import Identity as I
from torch.nn import AdaptiveAvgPool2d as Ap
from torch.nn import Flatten as F
from torch.nn import Linear as L
from torch.nn import ReLU as R 
from torch.nn import Dropout as D

class Alexnet(nn.Module):
    def __init__(self, conf=None):
        super(Alexnet, self).__init__()
        use_bn = conf['bn'] if conf is not None and 'bn' in conf else False
        self.backbone = S(
            C(3, 96, 11, 4, 2),
            R(inplace=True),
            BN(96) if use_bn else I(),
            M(3, stride=2),
            C(96, 256, 5, 1, 2),
            R(inplace=True),
            BN(256) if use_bn else I(),
            M(3, 2),
            C(256, 384, 3, 1, 1),
            R(inplace=True),
            BN(384) if use_bn else I(),
            C(384, 384, 3, 1, 1),
            R(inplace=True),
            BN(384) if use_bn else I(),
            C(384, 256, 3, 1, 1),
            R(inplace=True),
            BN(256) if use_bn else I(),
            M(3, stride=2),
        ) 
        self.head = S(
            Ap((6, 6)), # makes it work with any input size
            F(),
            L(256 * 6 * 6, 4096),
            R(inplace=True),
            BN1(4096) if use_bn else I(),
            D(p=0.5),
            L(4096, 4096),
            R(inplace=True),
            BN1(4096) if use_bn else I(),
            D(p=0.5),
            L(4096, 1000),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
