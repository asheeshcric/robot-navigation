import torch
import torch.nn as nn
from torchvision import models, transforms


class EncoderCNN(nn.Module):
    
    def __init__(self, base_model, params):
        super().__init__()
        
        self.params = params
        
        # Delete the last FC layer and add your own
        modules = list(base_model.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        
        self.linear = nn.Linear(base_model.fc.in_features, self.params.num_classes)
        # If you want to use the features as embeddings instead of classification, use a BN layer at the end
        self.bn = nn.BatchNorm1d(self.params.num_classes, momentum=0.01)

        
    def forward(self, images):
        # First, extract features from the pretrained encoder model
        with torch.no_grad():
            features = self.base_model(images)
            
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features