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
        
        self.last_layer = self.get_last_layer(base_model)
        
        # If you want to use the features as embeddings instead of classification, use a BN layer at the end
        # self.bn = nn.BatchNorm1d(self.params.num_classes, momentum=0.01)

        
    def forward(self, images):
        # First, extract features from the pretrained encoder model
        with torch.no_grad():
            features = self.base_model(images)
            
        features = features.reshape(features.size(0), -1)
        features = self.last_layer(features)
        return features
    
    def get_last_layer(self, base_model):
        model_name = self.params.base_model
        num_classes = self.params.num_classes
        if 'resnet' in model_name:
            return nn.Linear(base_model.fc.in_features, num_classes)
        
        elif 'efficientnet' in self.params.base_model:
            return nn.Linear(base_model.classifier[1].in_features, num_classes)
        
        elif 'squeezenet1_0' == self.params.base_model:
            return nn.Sequential(
                nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
            )
            
            
#              Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#     (2): ReLU(inplace=True)
#     (3): AdaptiveAvgPool2d(output_size=(1, 1))
#   )
# )
        
        elif 'vit_b_16' == self.params.base_model:
            return nn.Sequential(
                nn.Linear(in_features=768, out_features=2, bias=True)
            )
        
        else:
            return base_model.fc.in_features