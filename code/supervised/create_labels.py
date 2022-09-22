import csv
import os
import pickle

import shutil
import tqdm

import pandas as pd


data_path = '/data/zak/robot/labels'


# def get_labels(joint_filenames):
#     labels = []
#     for joint_path in joint_filenames:
#         with open(joint_path, 'rb') as joint_file:
#             vel = pickle.load(joint_file)['velocity']
#             front_avg_vel = (vel[2] + vel[3]) / 2
#             if min(vel) > 1:
#                 label = 'no_obstacle'
#             elif -0.1 < front_avg_vel < 0.1:
#                 label = 'obstacle'
#             else:
#                 label = 'unknown'
#         labels.append(label)
#     return labels



# def create_csv(img_filenames, joint_filenames, dataset_path):
#     labels = get_labels(joint_filenames)
#     path_labels = list(zip(img_filenames, labels))
#     print(f'Writing labels.csv to {dataset_path}')
#     with open(os.path.join(dataset_path, 'labels.csv'), 'w') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(['img_path', 'label'])
#         writer.writerows(path_labels)
        

def get_direction(joint_path):
    with open(joint_path, 'rb') as f:
        values = pickle.load(f)
    velocities = values['velocity']
    BL, BR, FL, FR = velocities
    # Check stop condition
    if all([-0.5 < v < 0.5 for v in velocities]):
        return 'STOP'
    elif all([v > 2 for v in velocities]):
        return 'FORWARD'
    elif all([FL > FR, BL > BR]):
        return 'RIGHT'
    elif all([FL < FR, BL < BR]):
        return 'LEFT'
    else:
        return 'UNKNOWN'
        
for dataset in os.listdir(data_path):
    if '.ipynb_checkpoints' == dataset:
        continue
    info_csv_path = os.path.join(data_path, dataset, 'info.csv')    

    # Read info.csv file that contains img_paths, joints_paths, for all jointLabeled == True
    info_df = pd.read_csv(info_csv_path)
    # Get all labeled_data
    labeled_data = info_df.loc[info_df['jointLabeled'] == True]
    # Drop 'Unnamed: 0' Column and 'label' column
    labeled_data = labeled_data.loc[:, ~labeled_data.columns.str.contains('^Unnamed')]
    
    # Get directions for labeling the data
    labeled_data['label'] = labeled_data.joints.apply(lambda joint_path: get_direction(joint_path))
    
    print('---------------------------------')
    print(f'For {dataset} Dataset:')
    print(labeled_data.label.value_counts())
    
    labeled_data.to_csv(os.path.join(data_path, dataset, 'labels_directions_ashish.csv'), index=False)
    
    
    
    
    
