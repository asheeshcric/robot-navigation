"""
What we have?
data_path = '/data/zak/rosbag/labeled/'
1. Three datasets: 'heracleia', 'mocap', 'uc'
2. Inside each dataset dir, there's a labels.csv file that contains labels for each image in the dataset
    - Labels: 'obstacle', 'no_obstacle', 'unknown'
    
Task:
1. Train a model to classify between obstacle and no_obstacle images


"""

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

from dataset import RobotDataset
from model import EncoderCNN


def train(model, train_loader, val_loader, params):
    loss_function = nn.CrossEntropyLoss(weight=params['class_weights'])
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Star the training
    print(f'Training...')
    for epoch in range(params['num_epochs']):
        for batch in train_loader:
            inputs, labels = batch[0].to(params['device']), batch[1].to(params['device'])
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            # Clip the gradients because I'm using LSTM layer in the model
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
        if epoch % 2 != 0:
            # Check train and val accuracy after every two epochs
            print('Validating...')
            _, _, train_acc = test(model, train_loader)
            _, _, val_acc = test(model, val_loader)
            print(f'Epoch: {epoch+1} | Loss: {loss} | Train Acc: {train_acc} | Validation Acc: {val_acc}')
        else:
            print('Training epoch...')
            print(f'Epoch: {epoch+1} | Loss: {loss}')
            
        # Save checkpoint after every 10 epochs
        if (epoch+1) % 10 == 0:
            current_time = datetime.now().strftime('%m_%d_%Y_%H_%M')
            torch.save(model.state_dict(), f'{params["file_name"]}-{current_time}-lr-{params["learning_rate"]}-epochs-{epoch+1}-acc-{val_acc:.2f}.pth')
    
    print('Training complete')
    return model


def test(model, data_loader):
    correct, total = 0, 0
    preds, actual = [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if not batch:
                continue
            inputs, labels =  batch[0].to(params.device), batch[1].to(params.device)
            outputs = model(inputs)
            _, class_pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (class_pred == labels).sum().item()
            preds.extend(list(class_pred.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))
            
    acc = 100*(correct/total)
    model.train()
    return preds, actual, acc


def get_confusion_matrix(params, preds, actual):
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]
    
    cf = confusion_matrix(actual, preds, labels=list(range(params.num_classes)))
    return cf


params = {
    'root_path': '/data/zak/rosbag/labeled/',
    'train_sets': ['heracleia', 'mocap'],
    'test_sets': ['uc'],
    'num_classes': 2,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 128,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'file_name': 'resnet50_encoder_cnn'
    
}

transform = ResNet50_Weights.DEFAULT.transforms()

train_set = RobotDataset(params=params, train=True, transform=transform)
test_set = RobotDataset(params=params, train=False, transform=transform)

# Set class_weights based on the train_set
params['class_weights'] = torch.FloatTensor(
        [train_set.class_weights[label] for label in ['obstacle', 'no_obstacle']]
    ).to(params['device'])

encoder_model = EncoderCNN(params=params).to(params['device'])


# DataParallel settings
params['num_gpus'] = torch.cuda.device_count()
print(f'Number of GPUs available: {params["num_gpus"]}')
if params['device'].type == 'cuda' and params['num_gpus'] > 1:
    encoder_model = nn.DataParallel(encoder_model, list(range(params['num_gpus'])))


# Load train and test datasets
train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=True)

# Train
encoder_model = train(encoder_model, train_loader, test_loader, params)

# Validate the model
preds, actual, acc = test(encoder_model, test_loader)
print(f'Validation Accuracy: {acc}')
print(get_confusion_matrix(params, preds, actual))