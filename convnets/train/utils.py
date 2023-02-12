import random 
import torch 
import numpy as np
import os
import albumentations as A

def setup_trans(trans):
    _trans = []
    for t, t_params in trans.items():
        if t == 'SmallestMaxSize' or t == 'LongestMaxSize':
            if 'type' in t_params and t_params['type'] == 'range':
                t_params['max_size'] = list(range(t_params['max_size'][0], t_params['max_size'][1]+1))
                del t_params['type']
        _trans.append(getattr(A, t)(**t_params))
    return A.Compose(_trans)

def seed_everything(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class save_checkpoint():
	def __init__(self, name, path='./checkpoints', monitor='val_error', mode='min', save_top_k=1, save_last=True, verbose=False):
		self.monitor = monitor
		self.mode = mode
		self.save_top_k = save_top_k
		self.save_last = save_last
		self.verbose = verbose
		self.best_metric = np.inf if self.mode == 'min' else -np.inf
		self.ckpt_path = None
		self.path = path
		self.name = name

	def __call__(self, hist, model, optimizer):
		metric = hist[self.monitor][-1]
		epoch = hist['epoch'][-1]
		if (self.mode == 'min' and metric < self.best_metric) or (self.mode == 'max' and metric > self.best_metric):	
			if self.ckpt_path is not None:
				os.remove(self.ckpt_path)
			self.ckpt_path = f'{self.path}/{self.name}_{self.monitor}={metric:.5f}_epoch={epoch}.pth'
			torch.save(model.state_dict(), self.ckpt_path)
			self.best_metric = metric
			if self.verbose:
				print(f'Saved new best model with {self.monitor}={metric:.5f} at epoch={epoch}')