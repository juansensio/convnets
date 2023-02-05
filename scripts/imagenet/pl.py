import pytorch_lightning as pl
import argparse
import os 
import yaml 
from convnets.utils import deep_update
import albumentations as A
from convnets.datasets import ImageNet
from torch.utils.data import DataLoader
import convnets.models as models
import torch
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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
        'pin_memory': False,
        'persistent_workers': False,
    },
    'train': {
        'max_epochs': 10,
        'overfit_batches': False,
        'accelerator': 'gpu',
        'devices': 1,
        'enable_checkpointing': False,
    },
}

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        trans = A.Compose([
            getattr(A, t)(**t_params) for t, t_params in self.config['transforms'].items()
        ]) if self.config['transforms'] is not None else None
        self.dataset = {
            'train': ImageNet(self.config['path'], 'train', trans=trans),
            'val': ImageNet(self.config['path'], 'val', trans=A.CenterCrop(224, 224)) # for now...
        }

    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.config['batch_size'],
            shuffle=shuffle if shuffle is not None else True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=self.config['persistent_workers'],
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.dataset['train'], batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.dataset['val'], batch_size, shuffle)

class Module(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = getattr(models, config['model'])(config) if config is not None else None
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
        self.top5acc = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5)

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        top5acc = self.top5acc(y_hat, y)
        return loss, 1. - acc, 1. - top5acc

    def training_step(self, batch, batch_idx):
        loss, error, top5error = self.shared_step(batch)
        self.log('loss', loss)
        self.log('t1e', error, prog_bar=True)
        self.log('t5e', top5error, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, error, top5error = self.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_t1e', error, prog_bar=True, sync_dist=True)
        self.log('val_t5e', top5error, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams['optimizer_params'])
        if self.hparams['scheduler']:
            return {
                'optimizer': optimizer,
                'lr_scheduler': getattr(torch.optim.lr_scheduler, self.hparams['scheduler'])(optimizer, **self.hparams['scheduler_params']),
                'monitor': 'val_t1e'
            }
        return optimizer

def train(config):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42, workers=True)
    dm = DataModule(config['dataloader'])
    module = Module(config)
    config['train']['callbacks'] = []
    if config['train']['enable_checkpointing']:
        config['train']['callbacks'] += [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{config["logger"]["name"]}-{{val_t1e:.5f}}-{{epoch}}',
                monitor='val_t1e',
                mode='min',
                save_top_k=1
            ),
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{config["logger"]["name"]}-{{epoch}}',
                monitor='epoch',
                mode='max',
                save_top_k=1
            )
        ]
    if 'logger' in config:
        config['logger'] = WandbLogger(
            project=config['logger']['project'],
            name=config['logger']['name'],
            config=config
        )
        if config['scheduler']:
            config['train']['callbacks'] += [LearningRateMonitor(logging_interval='step')]
    trainer = pl.Trainer(**config['train'])
    trainer.fit(module, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--config', help='yml config to override defaults', default=None)
    parser.add_argument('--experiment', help='yml config to override defaults', default=None)
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
    train(default_config)