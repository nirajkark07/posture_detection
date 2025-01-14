import os
import numpy as np

import os
import shutil
import random

# Defintion
dataset_dir = r'E:\NIRAJ\Datasets\2024-09-16 GearBox\Niraj_2'  # Change this to the directory where your dataset is stored
output_dir = r'E:\NIRAJ\GIT\yolov10\dataset'  # Change this to where you want the train/valid/test folders to be created
train_split = 0.7
valid_split = 0.2
test_split = 0.1

# Ensure the split ratios sum to 1
assert train_split + valid_split + test_split == 1.0, "Split ratios should sum to 1."

# Create output directories if they don't exist
for folder in ['train', 'valid', 'test']:
    for subfolder in ['images', 'labels']:
        os.makedirs(os.path.join(output_dir, folder, subfolder), exist_ok=True)

# Get a list of all .png files (images) and corresponding .json files (labels)
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
json_files = [f.replace('.png', '.json') for f in image_files]

# Ensure there's a corresponding json file for each image
assert all(os.path.exists(os.path.join(dataset_dir, json_file)) for json_file in json_files), "Some JSON files are missing."

# Shuffle the dataset to ensure randomness in the split
combined_files = list(zip(image_files, json_files))
random.shuffle(combined_files)

# Calculate the number of samples for each split
total_images = len(combined_files)
train_count = int(train_split * total_images)
valid_count = int(valid_split * total_images)
test_count = total_images - train_count - valid_count

# Split the data
train_files = combined_files[:train_count]
valid_files = combined_files[train_count:train_count + valid_count]
test_files = combined_files[train_count + valid_count:]

# Helper function to copy files to destination folder
def copy_files(file_pairs, destination_folder):
    for image_file, json_file in file_pairs:
        # Copy image and corresponding json file
        shutil.copy(os.path.join(dataset_dir, image_file), os.path.join(destination_folder, 'images', image_file))
        shutil.copy(os.path.join(dataset_dir, json_file), os.path.join(destination_folder, 'images', json_file))

# Copy files to respective folders
copy_files(train_files, os.path.join(output_dir, 'train'))
copy_files(valid_files, os.path.join(output_dir, 'valid'))
copy_files(test_files, os.path.join(output_dir, 'test'))

print("Dataset organized successfully into train, valid, and test folders.")