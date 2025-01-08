from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

class BirdClassifierResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True) -> None:
        super(BirdClassifierResNet, self).__init__()
        
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def save(self, path:str = 'model') -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path:str = 'model') -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()