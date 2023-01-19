import torchvision
from convnets.models import BarlowTwins
import torch
from convnets.train import barlow_fit 
from convnets.datasets import SSLDataset 
import albumentations as A
from convnets.datasets import SeCo 
import pandas as pd

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PCTG_START=0.03

seco = SeCo()

dataset = SSLDataset(seco.data.image.values, trans=A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.3),
    A.ToGray(p=0.3),
    A.GaussianBlur(p=0.3),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)

backbone = torch.nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
model = BarlowTwins(backbone)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=PCTG_START, max_lr=LR, total_steps=EPOCHS, verbose=True)

hist = barlow_fit(model, dataloader, optimizer, scheduler, epochs=EPOCHS)

df = pd.DataFrame(hist)
df.to_csv('barlow.csv', index=False)