import torch.nn as nn

class MSFE(nn.Module):
    # multiscale feature extraction model
    # override this class and define the backbone and head
    def __init__(self, features_only=False):
        super(MSFE, self).__init__()
        self.features_only = features_only
    
    def forward(self, x):
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        return features if self.features_only else self.head(features[-1])