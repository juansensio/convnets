# convnets

🚧 Under construction

Convolutional Neural Networks and utilities for Computer Vision.

- [Learn](learn) about convnets.
- Learn about popular [models](models).

## Models API

`convnets` offers implementations for the following models:

- [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [VGG](https://arxiv.org/abs/1409.1556)
- [ResNet](https://arxiv.org/abs/1512.03385)

To instantiate a model you need to import the corresponding class and pass a valid `configuration` object to the constructor:

```python
from convnets.models import ResNet

r18_config = {
	'l': [
		{'r': 2, 'f': 64},
		{'r': 2, 'f': 128},
		{'r': 2, 'f': 256},
		{'r': 2, 'f': 512}
	], 
	'b': False
}

model = ResNet(r18_config)
```

Or you can use one of the predefined configurations, or variants:

```python
from convnets.models import ResNet, ResNetConfig

model = ResNet(ResNetConfig.r18)
```

You can find the implementation of each model and configuration examples in the [`convnets/models`](convnets/models) directory.

## Training API

If you want to train a model in your notebooks, you can use our [fit](convents/train/fit.py) function:

```python
form convnets.train import fit 

hist = fit(model, dataloader, optimizer, criterion, metrics, max_epochs)
```

You can use any Pytorch model. You will need to define the Pytorch dataloader, optimizer and criterion. For the metrics, the function expects a dict with the name of the metric as key and the metric function as value. The metric function must receive the model output and the target and return a scalar value. You can find some examples in [`convnets/metrics`](convnets/metrics.py). The `max_epochs` parameter is the maximum number of epochs to train the model. The function will return a dict with the training history. 

Additionally, we offer a [training script](train/train.py) that you can execute from the command line.

```bash
python scripts/train.py <path_to_config_file>
```

You will have to pass the path to a yaml file with the configuration for your training, including the model, optimizer, criterion, metrics, dataloader, etc. You can find some examples in the [`configs`](scripts/configs) directory (which are `timm` and `pytorch-lightning` compatible).

We also offer Pytorch Lightning interoperability.