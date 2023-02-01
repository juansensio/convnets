import argparse
import convnets.train.imagenet.configs as configs
import convnets.models as models
from torch.utils.data import DataLoader
from convnets.train import fit
import torch
import os
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from convnets.models import Alexnet

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

def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = Alexnet().to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    output = ddp_model(torch.randn(32, 3, 32, 32), log=True)
    print(output.size())
    dist.destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--base-config', help='Base configuration to be used for training', default="alexnet")
    parser.add_argument('-world-size', help='Number of nodes for distrubuted training', default=1)
    args = parser.parse_args()
    world_size = int(args.world_size)
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # fn = train 
        fn = example
        mp.spawn(fn, nprocs=world_size, args=(world_size,))
    else:
        config = getattr(configs, args.base_config)()
        train(config)