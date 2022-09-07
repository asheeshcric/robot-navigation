import torch
import torch.nn as nn
from torchvision import models, transforms


class EncoderCNN(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        # Initialize ResNet50 model with the best available weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Delete the last FC layer and add your own
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, self.params['num_classes'])
        self.bn = nn.BatchNorm1d(self.params['num_classes'], momentum=0.01)        
        
    def forward(self, images):
        # First, extract features from the pretrained encoder model
        with torch.no_grad():
            features = self.resnet(images)
            
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features