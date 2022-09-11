import csv
import os
import pickle

import shutil
import tqdm


data_path = '/data/zak/rosbag/labeled'


def get_labels(joint_filenames):
    labels = []
    for joint_path in joint_filenames:
        with open(joint_path, 'rb') as joint_file:
            vel = pickle.load(joint_file)['velocity']
            front_avg_vel = (vel[2] + vel[3]) / 2
            if min(vel) > 1:
                label = 'no_obstacle'
            elif -0.1 < front_avg_vel < 0.1:
                label = 'obstacle'
            else:
                label = 'unknown'
        labels.append(label)
    return labels



def create_csv(img_filenames, joint_filenames, dataset_path):
    labels = get_labels(joint_filenames)
    path_labels = list(zip(img_filenames, labels))
    print(f'Writing labels.csv to {dataset_path}')
    with open(os.path.join(dataset_path, 'labels.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['img_path', 'label'])
        writer.writerows(path_labels)
        
        
        
for dataset in os.listdir(data_path):
    dataset_path = os.path.join(data_path, dataset)    

    # Sort all img and joints contents in ascending order and put them on a csv file with the obtained label
    img_filenames = sorted(os.listdir(os.path.join(dataset_path, 'img')))
    img_filenames = [os.path.join(dataset_path, 'img', img) for img in img_filenames]
    joint_filenames = sorted(os.listdir(os.path.join(dataset_path, 'joints')))
    joint_filenames = [os.path.join(dataset_path, 'joints', joint) for joint in joint_filenames]
    create_csv(img_filenames, joint_filenames, dataset_path)
    
    
    
