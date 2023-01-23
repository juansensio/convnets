import torchvision
from convnets.models import BarlowTwins
import torch
from convnets.train import barlow_fit, fit
import albumentations as A
import pandas as pd
from convnets.datasets import EuroSAT, SSLDataset, SeCo, ImageClassificationDataset
import torch
from torch.utils.data import DataLoader

MLP_DIM = 2048
BATCH_SIZE = 256
EPOCHS = 300
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
model = BarlowTwins(backbone, f2=MLP_DIM)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*EPOCHS), int(0.8*EPOCHS)])


def SSLeval(SSLmodel):
    data = EuroSAT()
    dataset = {
        'train': ImageClassificationDataset(data.train.image.values, data.train.label.values),
        'val': ImageClassificationDataset(data.test.image.values, data.test.label.values),
    }
    dataloaders  =  {
        'train': DataLoader(dataset['train'], batch_size=256, shuffle=True, num_workers=10, pin_memory=True),
        'val': DataLoader(dataset['val'], batch_size=256, shuffle=False, num_workers=10, pin_memory=True),
    }
    model = BarlowTwins(backbone, f2=MLP_DIM)
    model.load_state_dict(SSLmodel.state_dict())
    # we freeze the weights of the pretrained model so they don't get updated
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(512, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    hist = fit(model, dataloaders, optimizer, criterion, 'cuda:1', log=False, epochs=30)
    try:
        df = pd.read_csv('barlow_eurosat.csv')
        df = df.append({'val_error': hist['val_error'][-1]}, ignore_index=True)
        df.to_csv('barlow_eurosat.csv', index=False)
    except:
        df = pd.DataFrame({'val_error': [hist['val_error'][-1]]})
    return 


hist = barlow_fit(model, dataloader, optimizer, scheduler, epochs=EPOCHS, SSLeval)

df = pd.DataFrame(hist)
df.to_csv('barlow.csv', index=False)