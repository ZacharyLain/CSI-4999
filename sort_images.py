# imports
import os
import shutil
import random

def split_images(real_src_dir, fake_src_dir, base_output_dir, training_ratio):
    # Define output directories for real and fake images
    training_real_dir = os.path.join(base_output_dir, 'training/real')
    training_fake_dir = os.path.join(base_output_dir, 'training/fake')
    validation_real_dir = os.path.join(base_output_dir, 'validation/real')
    validation_fake_dir = os.path.join(base_output_dir, 'validation/fake')
    
    # Create output directories
    os.makedirs(training_real_dir, exist_ok=True)
    os.makedirs(training_fake_dir, exist_ok=True)
    os.makedirs(validation_real_dir, exist_ok=True)
    os.makedirs(validation_fake_dir, exist_ok=True)
    
    print(f'Copying images from {real_src_dir} and {fake_src_dir} to {base_output_dir}')
    
    # Get list of img file paths
    all_real_img = [ ]
    all_fake_img = [ ]
    
    for i in os.listdir(real_src_dir)
        if os.path.isfile(os.path.join(real_src_dir, i)) and i.endswith('.jpg'):
            all_real_img.append(i)
    
    for i in os.listdir(fake_src_dir):
            if os.path.isfile(os.path.join(fake_src_dir, i)) and i.endswith('.jpg'):
                all_fake_img.append(i)
    
    # Split images into training and validation sets
    real_split = int(len(all_real_img) * training_ratio)
    fake_split = int(len(all_fake_img) * training_ratio)
    
    real_train_files = all_real_img[:real_split]
    real_val_files = all_real_img[real_split:]
    
    fake_train_files = all_fake_img[:fake_split]
    fake_val_files = all_fake_img[fake_split:]
    
    # Combine real and fake files
    train_files = real_train_files + fake_train_files
    val_files = real_val_files + fake_val_files

    # Shuffle files
    random.shuffle(train_files)
    random.shuffle(val_files)

    # Copy images into output directories
    for img in train_files:
        shutil.copy(os.path.join(real_src_dir, img), training_real_dir)
        shutil.copy(os.path.join(fake_src_dir, img), training_fake_dir)
    
    for img in val_files:
        shutil.copy(os.path.join(real_src_dir, img), validation_real_dir)
        shutil.copy(os.path.join(fake_src_dir, img), validation_fake_dir)
        
    print(f'Copied {len(train_files)} training images and {len(val_files)} validation images')