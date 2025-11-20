"""
Select images from a CSV file for training, choosing a specified number per class.

Usage:
    python choose_images_flim_train.py <num_classes> <split> <num_images_per_class> <seed>

Arguments:
    dataset: Dataset name (e.g., 'cistos', 'eggs', 'larvae').
    num_classes: Number of classes in the dataset.
    split: Split identifier (integer).
    num_images_per_class: Number of images to select per class.
    seed: Random seed for reproducibility.

This script:
    - Reads image names from a CSV file.
    - Selects a specified number of images per class.
    - Copies selected images to a target directory.
    - Logs the selection and seed to a CSV file.
"""

import os
import random
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Choose images for GFLIM training")
    parser.add_argument("dataset", type=str, help="Dataset path")
    parser.add_argument("num_classes", type=int, help="Number of classes in the dataset")
    parser.add_argument("split", type=int, help="Split identifier (integer)")
    parser.add_argument("num_images_per_class", type=int, help="Number of images to select per class")
    parser.add_argument("random_state", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args.dataset, args.num_classes, args.split, args.num_images_per_class, args.random_state

def read_image_names(csv_path):
    # Read image names from the CSV file and sort them
    print(f"Reading image names from {csv_path}")
    with open(csv_path, "r") as f:
        image_names = [line.strip() for line in f]
    image_names.sort()
    return image_names

def group_images_by_label(image_names):
    # Group images by their label (assumed to be the first part of the filename)
    labels_to_images = {}
    for file_path in image_names:
        name = file_path.split("/")[-1]          # Get the filename only
        label = int(name.split("/")[-1].split("_")[0])     # Extract label from filename
        labels_to_images.setdefault(label, []).append(file_path)
    return labels_to_images

def select_images(labels_to_images, num_classes, num_images_per_class, random_state):
    # Select a specified number of images per class using the given random state
    selected_images = []
    for class_idx in range(1, num_classes + 1):
        if class_idx in labels_to_images:
            random.seed(random_state)
            selected = random.sample(labels_to_images[class_idx], num_images_per_class)
            selected_images.extend(selected)
        else:
            print(f"Class {class_idx} not found in the dataset.")
            sys.exit(1)
    return selected_images

def copy_csv_files(split, dataset, rem_files : list = []):
    # Copy train and test CSV files for the given split
    os.system(f"cp {dataset}/build/splits/train-val{split}.csv {dataset}/build/train-val{split}.csv")
    os.system(f"cp {dataset}/build/splits/test{split}.csv {dataset}/build/test{split}.csv")

    with open(f"{dataset}/build/splits/train-val{split}.csv", "r") as f:
        lines = f.readlines()
    with open(f"{dataset}/build/splits/train-val{split}.csv", "w") as f:
        for line in lines:
            if line.strip() not in rem_files:
                f.write(line)

def remove_existing_train_dir(split, dataset):
    # Remove existing train directory for the split, if it exists
    train_dir = f"{dataset}/build/train{split}.csv"
    if os.path.exists(train_dir):
        os.system(f"rm -f {train_dir}")

def copy_selected_images(selected_images, split, dataset):
    # Copy selected images to the output directory for the split
    #output_dir = f"{dataset}/build/train{split}"
    #os.makedirs(output_dir, exist_ok=True)
    for image in selected_images:
        #src = f"{dataset}/images/{image}"
        #dst = os.path.join(output_dir, image)
        #os.system(f"cp {src} {dst}")

        # write image paths in a csv file
        with open(f"{dataset}/build/train{split}.csv", "w") as f:
            for image in selected_images:
                f.write(f"{image}\n")

def main():
    # Main workflow: parse args, select images, copy files, and log selection
    dataset, num_classes, split, num_images_per_class, random_state = parse_args()
    
    csv_path = f"{dataset}/build/splits/train-val{split}.csv"    
    image_names = read_image_names(csv_path)
    labels_to_images = group_images_by_label(image_names)
    selected_images = select_images(labels_to_images, num_classes, num_images_per_class, random_state)
    copy_csv_files(split, dataset)
    remove_existing_train_dir(split, dataset)
    copy_selected_images(selected_images, split, dataset)
    print(f"Selected {len(selected_images)} images for training.")

if __name__ == "__main__":
    main()
