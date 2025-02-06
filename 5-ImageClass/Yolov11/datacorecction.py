#divides to data and validation 


import os
import shutil
import random

# Paths
source_dir = "data"
train_dir = "train"
valid_dir = "validation"
split_ratio = 0.8  # 80% train, 20% valid

# Create train and valid directories
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    
    if os.path.isdir(category_path):  # Ensure it's a directory
        train_category_path = os.path.join(train_dir, category)
        valid_category_path = os.path.join(valid_dir, category)

        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(valid_category_path, exist_ok=True)

        # Get image files
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)  # Shuffle for randomness

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        valid_images = images[split_idx:]

        # Move files
        for img in train_images:
            shutil.move(os.path.join(category_path, img), os.path.join(train_category_path, img))

        for img in valid_images:
            shutil.move(os.path.join(category_path, img), os.path.join(valid_category_path, img))

print("Dataset split completed!")
