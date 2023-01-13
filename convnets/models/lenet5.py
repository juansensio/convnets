import torch 

def block(ci, co, k=5, s=1, p=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, k, s, p),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(2, stride=2)
    )

class LeNet5(torch.nn.Module):
  def __init__(self, in_channels=1, n_classes=10):
    super().__init__()
    self.conv1 = block(in_channels, 6)
    self.conv2 = block(6, 16)
    self.conv3 = torch.nn.Sequential(
        torch.nn.Conv2d(16, 120, 5, padding=0),
        torch.nn.Tanh()
    )
    self.fc1 = torch.nn.Sequential(
        torch.nn.Linear(120, 84),
        torch.nn.Tanh()
    )
    self.fc2 = torch.nn.Linear(84, n_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x