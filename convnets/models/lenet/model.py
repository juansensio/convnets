import torch 
from torch.nn import Sequential as S 
from torch.nn import Conv2d as C
from torch.nn import Tanh as T
from torch.nn import AvgPool2d as A
from torch.nn import Linear as L
from torch.nn import Flatten as F

class LeNet(torch.nn.Module):
  def __init__(self, conf=None, in_chans=1, num_classes=10):
    super().__init__()
    block = lambda ci,co,k=5,s=1,p=0: S(C(ci,co,k,s,p),T(),A(2,2))
    self.backbone = S(*[block(in_chans,6),block(6,16),S(C(16,120,5,1,0),T())])	
    self.head = S(*[F(),S(L(120, 84),T()),L(84,num_classes)])
  def forward(self, x):
    return self.head(self.backbone(x))