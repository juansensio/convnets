import argparse
from convnets.datasets import ImageNet
import convnets.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from convnets.metrics import error, top5_error
import numpy as np
import os 

def evaluate(model, checkpoint):
    dataset = ImageNet('/fastdata/imagenet256', 'val') # full res images, model will do tta!
    dataloader = DataLoader(
        dataset, 
        batch_size=128, 
        num_workers=10,
        pin_memory=True,
    )
    model = getattr(models, model)()
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']
    # In pl models the model is under the "model" property, so we need to remove the prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('model.', '')] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    metrics = {'t1err': [], 't5err': []}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x, y = x.cuda(), y.cuda()
            y_hat = model.tta(x)
            err = error(y_hat, y)
            metrics['t1err'].append(err.item())
            top5err = top5_error(y_hat, y)
            metrics['t5err'].append(top5err.item()) 
    for k, v in metrics.items():
        print(f'{k}: {np.mean(v):.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--model', help='the model', default=None)
    parser.add_argument('--checkpoint', help='the checkpoint to load', default=False)
    args = parser.parse_args()
    evaluate(args.model, args.checkpoint)