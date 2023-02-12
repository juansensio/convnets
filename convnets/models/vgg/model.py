import torch.nn as nn 
from torch.nn import Sequential as S 
from torch.nn import Conv2d as C
from torch.nn import ReLU as R 
from torch.nn import MaxPool2d as M
from torch.nn import BatchNorm2d as BN
from torch.nn import Identity as I
from torch.nn import AdaptiveAvgPool2d as Ap
from torch.nn import Flatten as F
from torch.nn import Linear as L
from torch.nn import ReLU as R 
from torch.nn import Dropout as D
from ..msfe import MSFE

class VGG(MSFE):
    def __init__(self, conf, features_only=False):
        super(VGG, self).__init__(features_only)
        block = lambda r,f1,f2,bn=False: S(*[S(C(f1 if i == 0 else f2,f2,3,1,1),R(),BN(f2) if bn else I) for i in range(r)],M(2,2))
        self.backbone = nn.ModuleList([block(c['r'], conf['l'][i-1]['f'] if i > 0 else 3, c['f'], bn=conf['bn']) for i, c in enumerate(conf['l'])])
        self.head = S(Ap((7, 7)),F(),L(512*7*7, 4096),R(),D(p=conf['d']),L(4096, 4096),R(),D(p=conf['d']),L(4096, 1000))