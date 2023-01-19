import torch

class BarlowTwins(torch.nn.Module):

    def __init__(self, backbone, f1=512, f2=512):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(f1, f2),
            torch.nn.BatchNorm1d(f2),
            torch.nn.ReLU(),
            torch.nn.Linear(f2, f2),
            torch.nn.BatchNorm1d(f2),
            torch.nn.ReLU(),
            torch.nn.Linear(f2, f2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x