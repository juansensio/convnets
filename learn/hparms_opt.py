import optuna
import torch
from convnets.train import seed_everything, fit
from convnets import LeNet5
import torchvision
from torchvision import transforms
import sys

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
	transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

device = 'cpu'

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    seed_everything()
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True),
        'test': torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True),
    }
    model = LeNet5(in_channels=3, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    hist = fit(model, dataloaders, optimizer, criterion, device, epochs=20, log=False)
    return hist['val_error'][-1]



if __name__ == '__main__':
	gpu = sys.argv[1]
	device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
	study = optuna.create_study(direction='minimize', storage="sqlite:///db.sqlite3",  study_name="lenet5_distributed", load_if_exists=True)
	study.optimize(objective, n_trials=20)
    