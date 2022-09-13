"""
What we have?
data_path = '/data/zak/rosbag/labeled/'
1. Three datasets: 'heracleia', 'mocap', 'uc'
2. Inside each dataset dir, there's a labels.csv file that contains labels for each image in the dataset
    - Labels: 'obstacle', 'no_obstacle', 'unknown'
    
Task:
1. Train a model to classify between obstacle and no_obstacle images


"""
import os
import argparse
from datetime import datetime
import statistics

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader

from dataset import RobotDataset
from model import EncoderCNN


def save_results_to_file(model, params, saved_model_file_name, test_loader):
    # Validate the model
    preds, actual, acc = test(model, test_loader)
    print(f'Average Test Accuracy: {statistics.mean(params.test_accs)}')
    confusion_matrix = get_confusion_matrix(params, preds, actual)
    print(confusion_matrix)
    
    
    # Write the results to a file
    with open('results.txt', 'a') as results_file:
        results_txt = f"""
        ------------------------------------------------------------------------------------
        ------------------------------------------------------------------------------------
            Base Model: {params.base_model}
            Labels File: {params.labels_file}
            Train Sets: {params.train_sets}
            Test Sets: {params.test_sets}
            Test Accs: {params.test_accs}
            Avg. Test Acc: {round(statistics.mean(params.test_accs), 2)}
            Confusion Matrix: {confusion_matrix}
            Loss List: {params.losses}
            Num Epochs: {params.num_epochs}
            Batch Size: {params.batch_size}
            Learning Rate: {params.learning_rate}
            Saved Model Checkpoint: {saved_model_file_name}
        """
        results_file.write(results_txt)


def train(model, train_loader, test_loader, params):
    loss_function = nn.CrossEntropyLoss(weight=params.class_weights)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # Star the training
    print(f'Training...')
    for epoch in range(params.num_epochs):
        for batch in train_loader:
            inputs, labels = batch[0].to(params.device), batch[1].to(params.device)
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
            _, _, val_acc = test(model, test_loader)
            print(f'Epoch: {epoch+1} | Loss: {loss} | Train Acc: {train_acc} | Test Acc: {val_acc}')
            params.test_accs.append(round(val_acc, 2))
        else:
            print('Training epoch...')
            print(f'Epoch: {epoch+1} | Loss: {loss}')
            
        params.losses.append(round(loss.item(), 2))
            
        # Save checkpoint after every 10 epochs
        if (epoch+1) % 2 == 0:
#         if epoch % 2 != 0:
            current_time = datetime.now().strftime('%m_%d_%Y_%H_%M')
            saved_model_file_name = f'saved_checkpoints/{params.model_file_name}-{current_time}-lr-{params.learning_rate}-epochs-{epoch+1}-acc-{val_acc:.2f}.pth'
            if not os.path.exists('saved_checkpoints'):
                os.makedirs('saved_checkpoints')
            torch.save(model.state_dict(), saved_model_file_name)
            save_results_to_file(model, params, saved_model_file_name, test_loader)
    
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


parser = argparse.ArgumentParser(description='Params for training on navigation images')
parser.add_argument('--base_model', type=str, default='resnet50', help='Base Encoder that you want on top of your training layer')
parser.add_argument('--labels_file', type=str, default='labels.csv', help='CSV file name for the labels extracted using the script')
parser.add_argument('--train_sets', type=str, default='heracleia,mocap', help='Datasets to be used for training. Example: "heracleia,mocap,uc". Choose one or more: comma separated')
parser.add_argument('--test_sets', type=str, default='uc', help='Datasets you want the model to be tested on. Format same as train_sets')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of Epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size while training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate for the model')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes you want the model to classify: Obstacle, No_Obstacle, Unknown')
params = parser.parse_args()

# Convert dataset names from string to list
params.train_sets = params.train_sets.split(',')
params.test_sets = params.test_sets.split(',')

params.test_accs = []
params.losses = []

additional_params = {
    'root_path': '/data/zak/rosbag/labeled/',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'model_file_name': f'{params.base_model}_train_{"_".join(params.train_sets)}_test_{"_".join(params.test_sets)}'
}
# Add additional parameters to the argument parser object
for param in additional_params:
    setattr(params, param, additional_params[param])
    

    
transforms = {
    'resnet50': models.ResNet50_Weights.DEFAULT.transforms(),
    'resnet18': models.ResNet18_Weights.DEFAULT.transforms(),
    'resnet34': models.ResNet34_Weights.DEFAULT.transforms(),
    'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT.transforms(),
    'squeezenet1_0': models.SqueezeNet1_0_Weights.DEFAULT.transforms(),
#     'vit_b_16': models.ViT_B_16_Weights.DEFAULT.transforms(),
}

base_models = {
    'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'resnet34': models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
    'efficientnet_b0': models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    'squeezenet1_0': models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT),
#     'vit_b_16': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
}

train_set = RobotDataset(params=params, train=True, transform=transforms[params.base_model])
test_set = RobotDataset(params=params, train=False, transform=transforms[params.base_model])

# Set class_weights based on the train_set
params.class_weights = torch.FloatTensor(
        [train_set.class_weights[label] for label in ['obstacle', 'no_obstacle']]
    ).to(params.device)

encoder_model = EncoderCNN(base_models[params.base_model], params=params).to(params.device)


# DataParallel settings
params.num_gpus = torch.cuda.device_count()
print(f'Number of GPUs available: {params.num_gpus}')
if params.device.type == 'cuda' and params.num_gpus > 1:
    encoder_model = nn.DataParallel(encoder_model, list(range(params.num_gpus)))


# Load train and test datasets
train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=True)

# Train
encoder_model = train(encoder_model, train_loader, test_loader, params)




