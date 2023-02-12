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
from ..msfe import MSFE

class ResBlock(nn.Module):
    def __init__(self, f1, f2, l0, b0):
        super(ResBlock, self).__init__()
        cb = lambda f1,f2,s=1: S(C(f1,f2,3,s,1),BN(f2),R()) 
        self.b = S(cb(f1,f2,2 if b0 and not l0 else 1),cb(f2,f2)) 	
        self.s = C(f1,f2,1,2,0) if b0 and not l0 else I() 			
    def forward(self, x):
        return self.s(x) + self.b(x)
    
class ResBottlBlock(nn.Module):
    def __init__(self, f1, f2, l0, b0):
        super(ResBottlBlock, self).__init__()
        cb = lambda f1,f2,k=3,s=1: S(C(f1,f2,k,s,1 if k == 3 else 0),BN(f2),R()) 
        self.b = S(cb(f1*4 if not b0 else f1,f2,1,2 if b0 and not l0 else 1),cb(f2,f2),cb(f2,f2*4,1))
        self.s = C(f1,f2*4,1,2 if b0 and not l0 else 1,0) if b0 else I()
    def forward(self, x):
        return self.s(x) + self.b(x)

class ResNet(MSFE):
    def __init__(self, conf, features_only=False):
        super(ResNet, self).__init__(features_only)
        self.backbone = nn.ModuleList(
            [S(C(3,conf['l'][0]['f'],7,2,3),BN(conf['l'][0]['f']),R(),M(3, 2, 1))] +  
            [S(*[
                ResBottlBlock(conf['l'][i-1]['f']*4 if i > 0 and j == 0 else l['f'], l['f'], i == 0, j == 0) if conf['b'] else ResBlock(conf['l'][i-1]['f'] if i > 0 and j == 0 else l['f'], l['f'], i == 0, j == 0)
                for j in range(l['r'])]) 
            for i, l in enumerate(conf['l'])]
        )
        self.head = S(Ap((1, 1)),F(),L(conf['l'][-1]['f']*4 if conf['b'] else conf['l'][-1]['f'], 1000))