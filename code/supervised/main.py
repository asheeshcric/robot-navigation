"""
What we have?
data_path = '/data/zak/rosbag/labeled/'
1. Three datasets: 'heracleia', 'mocap', 'uc'
2. Inside each dataset dir, there's a labels.csv file that contains labels for each image in the dataset
    - Labels: 'obstacle', 'no_obstacle', 'unknown'
    
Task:
1. Train a model to classify between obstacle and no_obstacle images


"""
import matplotlib.pyplot as plt

from dataset import RobotDataset

params = {
    'root_path': '/data/zak/rosbag/labeled/'
    'train_sets': ['heracleia', 'mocap'],
    'test_sets': ['uc'],
}

transform = None

train_set = RobotDataset(params=params, train=True transform=transform)
test_set = RobotDataset(params=params, train=False, transform=transform)

    