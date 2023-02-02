import argparse
import convnets.train.imagenet.configs as configs
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

# TODO
#   wandb lgogging
#   distributed metrics
   
def train(rank, world_size, config):
    seed_everything()
    if rank == 0:
        print(config)
    model = getattr(models, config['model'])(config)
    trans = A.Compose([
        getattr(A, t)(**t_params) for t, t_params in config['transforms'].items()
    ])
    dataset = {
        'train': ImageNet(config['path'], 'train', trans=trans),
        'val': ImageNet(config['path'], 'val', trans=A.CenterCrop(224, 224)) # for now...
    }
    if world_size > 1:
        dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        sampler = {
            'train': torch.utils.data.distributed.DistributedSampler(
                dataset['train'],
                shuffle=True,
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
    		batch_size=config['batch_size'], 
    		shuffle=True if world_size == 1 else False, 
    		num_workers=config['num_workers'],  
    		pin_memory=True if config['device'] == 'cuda' else False,
            sampler=sampler['train'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
    	),
    	'val': DataLoader(
    		dataset['val'], 
    		batch_size=config['batch_size'], 
    		num_workers=config['num_workers'],
    		pin_memory=True if config['device'] == 'cuda' else False,
            sampler=sampler['val'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
    	),
    }
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optimizer_params'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.1, verbose=True, threshold_mode='abs')
    metrics = {'t1err': error, 't5err': top5_error}
    hist = fit(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        metrics, 
        device=config['device'], 
        epochs=10, # original paper says 90 epochs 
        after_val=lambda val_logs: scheduler.step(val_logs['t1err'][-1]), 
        rank=rank,
        limit_train_batches=1000 # comment to train on full dataset
    )
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--base-config', help='base configuration to be used for training', default="alexnet")
    parser.add_argument('--world-size', help='number of nodes for distrubuted training', default=1, type=int)
    parser.add_argument('--num-workers', help='number of cores for dataloading', default=10, type=int)
    parser.add_argument('--device', help='cuda or cpu', default='cuda')
    args = parser.parse_args()
    world_size = args.world_size
    config = getattr(configs, args.base_config)()
    config.update(num_workers=args.num_workers, device=args.device)
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train, nprocs=world_size, args=(world_size, config))
    else:
        train(0, 1, config)