import argparse
import convnets.models as models
from torch.utils.data import DataLoader
from convnets.train import fit
import torch
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from convnets.datasets import ImageNet
import albumentations as A
from convnets.metrics import error, top5_error
from convnets.train import seed_everything
import wandb
from convnets.utils import deep_update
import yaml 

# TODO
#   distributed metrics

default_config = {
    'model': 'Alexnet',
    'optimizer': 'SGD',
    'optimizer_params': {
        'lr': 1e-2,
    },
    'scheduler': None,
    'dataloader': {
        'path': '/fastdata/imagenet256',
        'batch_size': 64,
        'transforms': {
            'CenterCrop': {'height': 224, 'width': 224},
        },
        'num_workers': 0,
    },
    'train': {
        'epochs': 10,
        'device': 'cuda',
        'after_epoch_log': False,
        'overfit': False,
    },
}
   
def train(rank, world_size, config):
    seed_everything()
    if rank == 0:
        print(config)
    model = getattr(models, config['model'])(config)
    trans = A.Compose([
        getattr(A, t)(**t_params) for t, t_params in config['dataloader']['transforms'].items()
    ]) if config['dataloader']['transforms'] is not None else None
    dataset = {
        'train': ImageNet(config['dataloader']['path'], 'train', trans=trans),
        'val': ImageNet(config['dataloader']['path'], 'val', trans=A.CenterCrop(224, 224)) # for now...
    }
    if world_size > 1:
        dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        sampler = {
            'train': torch.utils.data.distributed.DistributedSampler(
                dataset['train'],
                shuffle=True if not config['datloader']['overfit'] else False,
                num_replicas=world_size,
                rank=rank
            ),
            'val': torch.utils.data.distributed.DistributedSampler(
                dataset['val'],
                num_replicas=world_size,
                rank=rank
            )
        }
    dataloader = {
    	'train': DataLoader(
    		dataset['train'], 
    		batch_size=config['dataloader']['batch_size'], 
    		shuffle=True if world_size == 1 and not config['train']['overfit'] else False, 
    		num_workers=config['dataloader']['num_workers'],  
    		pin_memory=True if config['train']['device'] == 'cuda' else False,
            sampler=sampler['train'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
    	),
    	'val': DataLoader(
    		dataset['val'], 
    		batch_size=config['dataloader']['batch_size'], 
    		num_workers=config['dataloader']['num_workers'],
    		pin_memory=True if config['train']['device'] == 'cuda' else False,
            sampler=sampler['val'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
    	),
    }
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optimizer_params'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(optimizer, **config['scheduler_params']) if config['scheduler'] is not None else None
    metrics = {'t1err': error, 't5err': top5_error}
    if rank == 0 and 'log' in config:
        wandb.init(project=config['log']['project'], name=config['log']['name'], config=config)
    fit(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        metrics, 
        rank=rank,
        after_val=lambda val_logs: scheduler.step(val_logs['t1err'][-1]) if scheduler is not None else None,
        on_epoch_end=lambda h,m,o: wandb.log({k: v[-1] for k, v in h.items()}) if rank == 0 and 'log' in config is not None else None,
        **config['train']
    )
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--config', help='yml config to override defaults', default=None)
    parser.add_argument('--experiment', help='yml config to override defaults', default=None)
    parser.add_argument('--gpus', help='gpus', default=1, type=int)
    parser.add_argument('--debug', help='debug', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.config:
        path = os.path.join(os.path.dirname(__file__), f'configs/{args.config}.yaml')
        with open(path, 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        deep_update(default_config, loaded_config)
    if args.experiment:
        path = os.path.join(os.path.dirname(__file__), f'experiments/{args.experiment}.yaml')
        with open(path, 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        deep_update(default_config, loaded_config)
    if args.debug:
        path = os.path.join(os.path.dirname(__file__), f'debug.yaml')
        with open(path, 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        deep_update(default_config, loaded_config)
    world_size = args.gpus
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train, nprocs=world_size, args=(world_size, default_config))
    else:
        train(0, 1, default_config)