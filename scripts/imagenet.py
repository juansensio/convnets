import argparse
import convnets.train.imagenet.configs as configs
import convnets.models as models
from torch.utils.data import DataLoader
from convnets.train import fit
import torch

def train(config):
	print(config)
	model = getattr(models, config['model'])(config)
	# dataset = {
	# 	'train': Imagenet(trans=config['trans']),
	# 	'val': Imagenet(trans=config['trans'])
	# }
	# dataloader = {
	# 	'train': DataLoader(
	# 		dataset['train'], 
	# 		batch_size=config['batch_size'], 
	# 		shuffle=True, 
	# 		num_workers=config['num_workers'],
	# 		pin_memory=config['pin_memory']
	# 	),
	# 	'val': DataLoader(
	# 		dataset['val'], 
	# 		batch_size=config['batch_size'], 
	# 		num_workers=config['num_workers'],
	# 		pin_memory=config['pin_memory']
	# 	),
	# }
	# optimizer = getattr(torch.optim, config['optimizer'])(**config['optimizer_params'])
	# criterion = torch.nn.CrossEntropyLoss()
	# hist = fit(
	# 	model, 
	# 	dataloader, 
	# 	optimizer, 
	# 	criterion, 
	# 	device="cpu", 
	# 	epochs=config['epochs'],
	# 	overfit=0,
	# 	log=True,
	# 	compile=False,
	# 	on_epoch_end=None,
	# 	limit_train_batches=0,
	# 	use_amp = True, 
	# )
	# return hist
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process imagenet.')
	parser.add_argument('--base-config', help='Base configuration to be used for training', default="alexnet")
	args = parser.parse_args()
	config = getattr(configs, args.base_config)()
	train(config)