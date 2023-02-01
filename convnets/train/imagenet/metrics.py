import torch

# def top-1 error
def top1_error(pred, target):
    return 1 - torch.mean((torch.argmax(pred, dim=1) == target).float())

# top-5 error
def top5_error(pred, target):
    return 1 - torch.mean((torch.argsort(pred, dim=1, descending=True)[:, :5] == target[:, None]).any(dim=1).float())

