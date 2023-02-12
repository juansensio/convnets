import convnets.models as models
import convnets.datasets as datasets
import convnets.metrics as metrics
from torch.utils.data import DataLoader
from convnets.train import fit
import torch
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from convnets.train import seed_everything
import wandb
import sys
import yaml
from pydantic import BaseModel
from convnets.train.utils import setup_trans

# TODO
#   distributed metrics
#   checkpointing

class TrainConfig(BaseModel):
    max_epochs: int = 10
    overfit_batches: int = 0
    accelerator: str = 'gpu'
    devices: int = 1

class Config(BaseModel):
    model: str
    conf: dict = None
    variant: str = None
    model_params: dict = {}
    optimizer: str = 'Adam'
    optimizer_params: dict = {}
    scheduler: str = None
    scheduler_params: dict = {}
    metrics: dict = {}
    dataloader: dict = {}
    logger: dict = None
    train: TrainConfig

def train(config: Config, rank: int = 0, world_size: int = 1):
    torch.set_float32_matmul_precision('high')
    seed_everything()
    if rank == 0:
        print(config)
    if config.conf is None: 
        assert config.variant is not None, 'should pass variant or conf'
        variants = getattr(models, config.model+'Config')
        config.conf = getattr(variants, config.variant)
    model = getattr(models, config.model)(config.conf, **config.model_params)
    config.dataloader['train']['trans'] = setup_trans(config.dataloader['train']['trans']) if 'trans' in config.dataloader['train'] else None
    config.dataloader['val']['trans'] = setup_trans(config.dataloader['val']['trans']) if 'val' in config.dataloader and 'trans' in config.dataloader['val'] else None
    dataset = {
        'train': getattr(datasets, config.dataloader['dataset'])(**config.dataloader['train']),
        'val': getattr(datasets, config.dataloader['dataset'])(**config.dataloader['val']) if 'val' in config.dataloader else None
    }
    if world_size > 1:
        dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        sampler = {
            'train': torch.utils.data.distributed.DistributedSampler(
                dataset['train'],
                shuffle=True if not config.train.overfit_batches else False,
                num_replicas=world_size,
                rank=rank
            ),
            'val': torch.utils.data.distributed.DistributedSampler(
                dataset['val'],
                num_replicas=world_size,
                rank=rank
            ) if dataset['val'] is not None else None
        }
    dataloader = {
        'train': DataLoader(
            dataset['train'], 
            batch_size=config.dataloader['train']['batch_size'], 
            shuffle=True if world_size == 1 and not config.train.overfit_batches else False, 
            num_workers=config.dataloader['train']['num_workers'] if 'num_workers' in config.dataloader['train'] else 0,  
            pin_memory=True if config.train.accelerator == 'gpu' else False,
            sampler=sampler['train'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
        ),
        'val': DataLoader(
            dataset['val'], 
            batch_size=config.dataloader['val']['batch_size'], 
            num_workers=config.dataloader['val']['num_workers'] if 'num_workers' in config.dataloader['val'] else 0,  
            pin_memory=True if config.train.accelerator == 'gpu' else False,
            sampler=sampler['val'] if world_size > 1 else None,
            persistent_workers=True if world_size > 1 else False
        ) if dataset['val'] is not None else None
    }
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer, **config.scheduler_params) if config.scheduler is not None else None
    if rank == 0 and config.logger is not None:
        wandb.init(project=config.logger['project'], name=config.logger['name'], config=config)
    fit(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        metrics={name: getattr(metrics, metric) for metric, name in config.metrics.items()}, 
        device='cuda' if config.train.accelerator == 'gpu' else 'cpu',
        rank=rank,
        after_val=lambda val_logs: scheduler.step(val_logs['t1err'][-1]) if scheduler is not None else None,
        on_epoch_end=lambda h,m,o: wandb.log({k: v[-1] for k, v in h.items()}) if rank == 0 and 'log' in config is not None else None,
        **config.train.dict()
    )
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    with open(f'{sys.argv[1]}', 'r') as stream:
        config = Config(**yaml.safe_load(stream))
    world_size = config.train.devices
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train, nprocs=world_size, args=(config, world_size))
    else:
        train(config)