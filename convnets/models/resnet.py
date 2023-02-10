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
from dataclasses import dataclass

@dataclass
class ResNetConfig:
	r18 = {
		'l': [
			{'r': 2, 'f': 64},
			{'r': 2, 'f': 128},
			{'r': 2, 'f': 256},
			{'r': 2, 'f': 512}
		], 
		'b': False
	}
	r34 = {
		'l': [
			{'r': 3, 'f': 64},
			{'r': 4, 'f': 128},
			{'r': 6, 'f': 256},
			{'r': 3, 'f': 512}
		], 
		'b': False
	}
	r50 = {
		'l': [
			{'r': 3, 'f': 64},
			{'r': 4, 'f': 128},
			{'r': 6, 'f': 256},
			{'r': 3, 'f': 512}
		], 
		'b': False
	}
	r101 = {
		'l': [
			{'r': 3, 'f': 64},
			{'r': 4, 'f': 128},
			{'r': 23, 'f': 256},
			{'r': 3, 'f': 512}
		], 
		'b': False
	}
	r152 = {
		'l': [
			{'r': 3, 'f': 64},
			{'r': 8, 'f': 128},
			{'r': 236, 'f': 256},
			{'r': 3, 'f': 512}
		], 
		'b': False
	}

class ResBlock(nn.Module):
	def __init__(self, f1, f2, l0, b0):
		# f1 -> input filters, f2 -> output filters, l0 -> is first layer?, b0 -> is first block in layer?
		super(ResBlock, self).__init__()
		cb = lambda f1,f2,s=1: S(C(f1,f2,3,s,1),R(),BN(f2)) 
		self.b = S(cb(f1,f2,2 if b0 and not l0 else 1),cb(f2,f2)) 	# stride 2 to reduce dims in first block of each layer
		self.s = C(f1,f2,1,2,0) if b0 and not l0 else I() 			# apply 1x1 conv to input if needed to match dimensions in residual connection
	def forward(self, x):
		return self.s(x) + self.b(x)
	
class ResBottlBlock(nn.Module):
	def __init__(self, f1, f2, l0, b0):
		super(ResBottlBlock, self).__init__()
		cb = lambda f1,f2,k=3,s=1: S(C(f1,f2,k,s,1 if k == 3 else 0),R(),BN(f2)) 
		self.b = S(cb(f1*4 if not b0 else f1,f2,1,2 if b0 and not l0 else 1),cb(f2,f2),cb(f2,f2*4,1))
		self.s = C(f1,f2*4,1,2 if b0 and not l0 else 1,0) if b0 else I()
	def forward(self, x):
		return self.s(x) + self.b(x)

class ResNet(nn.Module):
	def __init__(self, conf, features_only=False, **kwargs):
		super(ResNet, self).__init__()
		self.backbone = nn.ModuleList([
			S(C(3,conf['l'][0]['f'],7,2,3),BN(conf['l'][0]['f']),R(),M(3, 2, 1)), # first layer
			[S(*[
				ResBottlBlock(conf['l'][i-1]['f']*4 if i > 0 and j == 0 else l['f'], l['f'], i == 0, j == 0) if conf['b'] else ResBlock(conf['l'][i-1]['f'] if i > 0 and j == 0 else l['f'], l['f'], i == 0, j == 0)
				for j in range(l['r'])]) 
			for i, l in enumerate(conf['l'])]
		])
		self.features_only = features_only
		if not features_only:
			self.head = S(Ap((1, 1)),F(),L(conf['l'][-1]['f']*4 if conf['b'] else conf['l'][-1]['f'], 1000))
	def forward(self, x):
		features = []
		for layer in self.backbone:
			x = layer(x)
			features.append(x)
		if self.features_only:
			return features
		return self.head(features[-1])