import torch 
import torch.nn as nn 

class Alexnet(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(x):
        return x