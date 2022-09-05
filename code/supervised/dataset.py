import os
import pandas as pd

from skimage import io
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    
    
    def __init__(self, params, transform=None):
        self.params = params
        self.transform = transform
        
        # Add all samples from the train_sets
        self.samples, self.label_counts = self._get_samples()
        self.class_weights = self._get_class_weights()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = self.samples[idx]
        img_path, label = self.samples[idx]
        img = io.imread(img_path)
        label = self._label_class(label)
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def _get_samples(self):
        dfs = []
        for dataset in self.params['train_sets']:
            csv_path = os.path.join(self.params['root_path'], dataset, 'labels.csv')
            img_labels = pd.read_csv(csv_path, delimiter=',')
            dfs.append(img_labels)
            
        df = pd.concat(dfs)
        label_counts = df['label'].value_counts().to_dict()
        return df.values.tolist(), label_counts
    
    def _get_class_weights(self):
        max_num = max(self.label_counts.values())
        weights = dict()
        for label, count in self.label_counts.items():
            weights[label] = count/max_num
            
        return weights
    
    def _label_class(self, label):
        labels = {'obstacle': 0, 'no_obstacle': 1, 'unknown': 2}
        return labels[label]
        
                