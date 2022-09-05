import torch.nn as nn
from torchvision import models, transforms


class ResNetFCModel(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        # Initialize ResNet50 model with the best available weights
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.eval()
        
        # Create your own FC layer to train during
        in_ftrs = self.params['in_ftrs']
        num_classes = self.params['num_classes']
        self.fc1 = nn.Sequential(
            nn.Linear(in_ftrs, in_ftrs),
            nn.BatchNorm1d(in_ftrs),
            nn.ReLU(True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_ftrs, in_ftrs // 2),
            nn.BatchNorm1d(in_ftrs // 2),
            nn.ReLU(True),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_ftrs // 2, in_ftrs // 4),
            nn.BatchNorm1d(in_ftrs // 4),
            nn.ReLU(True),
        )
        
        self.fc4 = nn.Linear(in_ftrs // 4, num_classes)
        
        
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x