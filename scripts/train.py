import convnets.models as models
import convnets.datasets as datasets
import convnets.metrics as metrics
from torch.utils.data import DataLoader
from convnets.train import fit
import torch
from convnets.train import seed_everything
import wandb
import sys
import yaml
from pydantic import BaseModel
from convnets.train.utils import setup_trans
import lightning as L
import os

class TrainConfig(BaseModel):
    max_epochs: int = 10
    overfit_batches: int = 0
    accelerator: str = 'gpu'
    devices: int = 1
    strategy: str = 'auto'
    precision: str = 'bf16'
    limit_train_batches: int = 0
    limit_val_batches: int = 0
    compile: bool = True

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
    ckpt_folder: str = None

def train(config: Config):
    seed_everything()
    if config.conf is None and config.variant is not None: 
        # assert config.variant is not None, 'should pass variant or conf'
        variants = getattr(models, config.model+'Config')
        config.conf = getattr(variants, config.variant)
    model = getattr(models, config.model)(config.conf, **config.model_params)
    config.dataloader['train']['trans'] = setup_trans(config.dataloader['train']['trans']) if 'trans' in config.dataloader['train'] else None
    config.dataloader['val']['trans'] = setup_trans(config.dataloader['val']['trans']) if 'val' in config.dataloader and 'trans' in config.dataloader['val'] else None
    dataset = {
        'train': getattr(datasets, config.dataloader['dataset'])(**config.dataloader['train']),
        'val': getattr(datasets, config.dataloader['dataset'])(**config.dataloader['val']) if 'val' in config.dataloader else None
    }
    dataloader = {
        'train': DataLoader(
            dataset['train'], 
            batch_size=config.dataloader['train']['batch_size'], 
            shuffle=True if not config.train.overfit_batches else False, 
            num_workers=config.dataloader['train']['num_workers'] if 'num_workers' in config.dataloader['train'] else 0,  
            pin_memory=True if config.train.accelerator == 'gpu' else False,
        ),
        'val': DataLoader(
            dataset['val'], 
            batch_size=config.dataloader['val']['batch_size'], 
            num_workers=config.dataloader['val']['num_workers'] if 'num_workers' in config.dataloader['val'] else 0,  
            pin_memory=True if config.train.accelerator == 'gpu' else False,
        ) if dataset['val'] is not None else None
    }
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer, **config.scheduler_params) if config.scheduler is not None else None
    class Callbacks:
        def __init__(self):
            self.best_metric = 1e8
            self.ckpt_folder = config.ckpt_folder
            self.ckpt1_name = 'epoch={epoch}.ckpt'
            self.ckpt2_name = 'epoch={epoch}-val_t1err={val_t1_err}.ckpt'
            self.previos_ckpt = None
        def before_start(self):
            if fabric.global_rank == 0 and self.ckpt_folder is not None:
                os.makedirs(self.ckpt_folder, exist_ok=True)
            # load or resume checkpoint
            # TODO
            # init logger
            if fabric.global_rank == 0 and config.logger is not None:
                wandb.init(project=config.logger['project'], name=config.logger['name'], config=config)
        def after_epoch(self, hist):
            gathered_hist = fabric.all_gather(hist)
            # scheduler
            val_t1err = gathered_hist['val_t1err'][-1].float().mean().item()
            print("ei", val_t1err)
            scheduler.step(val_t1err)
            # checkpoints
            if fabric.global_rank == 0 and self.ckpt_folder is not None:
                ckpt_name = self.ckpt1_name.format(epoch=f"{hist['epoch'][-1]:03d}")
                ckpt_path = os.path.join(self.ckpt_folder, ckpt_name)
                fabric.save(ckpt_path, {
                    'epoch': hist['epoch'][-1],
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                })
                if val_t1err < self.best_metric:
                    self.best_metric = val_t1err    
                    if self.previos_ckpt is not None:
                        os.remove(self.previos_ckpt)
                    ckpt_name = self.ckpt2_name.format(epoch=f"{hist['epoch'][-1]:03d}", val_t1_err=f'{val_t1err:.5f}')
                    ckpt_path = os.path.join(self.ckpt_folder, ckpt_name)
                    fabric.save(ckpt_path, {
                        'epoch': hist['epoch'][-1],
                        'model': model,
                        'optimizer': optimizer,
                        'scheduler': scheduler,
                    })
                    self.previos_ckpt = ckpt_path
            # logger
            if fabric.global_rank == 0 and config.logger is not None:
                wandb.log({k: v[-1].float().mean().item() for k, v in gathered_hist.items()})
    torch.set_float32_matmul_precision('high')
    fabric = L.Fabric(
        accelerator=config.train.accelerator, 
        devices=config.train.devices, 
        strategy=config.train.strategy, 
        precision=config.train.precision,
        callbacks=[Callbacks()]
    )
    fit(
        model, 
        dataloader, 
        optimizer, 
        criterion,
        metrics={name: getattr(metrics, metric) for metric, name in config.metrics.items()}, 
        fabric=fabric,
        **config.train.dict()
    )

if __name__ == '__main__':
    with open(f'{sys.argv[1]}', 'r') as stream:
        config = Config(**yaml.safe_load(stream))
    train(config)