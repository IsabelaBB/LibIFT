import os
import argparse
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Create Seeds From All Superpixels")
    parser.add_argument("train_file", type=str, help="Path to the training file.")
    parser.add_argument("mask_dir", type=str, help="Path to the mask directory.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    return parser.parse_args()

def get_train_filenames(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        image_filenames = [line.strip() for line in lines]
    elif os.path.isdir(path):
        image_filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(('.png', '.jpg', '.jpeg', '.tif', 'pgm'))]
    else:
        raise ValueError(f"The path '{path}' is neither a file nor a directory.")

    if not image_filenames:
        raise ValueError(f"No image filenames found in the training file '{path}'.")
    return image_filenames

def write_seeds_file(seeds, output_path, basename, img_shape, label):
    seeds_path = os.path.join(output_path, f"{basename}-seeds.txt")
    with open(seeds_path, 'w') as f:
        f.write("{} {} {}\n".format(len(seeds), img_shape[1], img_shape[0]))
        for x, y in seeds:
            f.write(f"{y} {x} -1 {label} 0\n")

def run(image_file, output_path, label, mask_dir):

    if label < 0:
        raise ValueError("Label must be non-negative.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    basename = os.path.basename(image_file).split('.')[0]
    img = np.array(Image.open(image_file))

    if len(img.shape) != 2:
        img = img[:, :, 0]
    
    if mask_dir is not None:
        mask_filename = os.path.join(mask_dir, f"{basename}.png")
        mask = np.array(Image.open(mask_filename))
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        seeds = [(x,y) for x in range(img.shape[0]) for y in range(img.shape[1]) if mask[x, y] > 0]
    else:
        seeds = [(x,y) for x in range(img.shape[0]) for y in range(img.shape[1])]

    write_seeds_file(seeds, output_path, basename, img.shape, label)

if __name__ == "__main__":
    args = parse_args()

    image_filenames = get_train_filenames(args.train_file)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    for img_name in image_filenames:
        label = int(os.path.basename(img_name).split('_')[0])
        run(img_name, args.output_path, label, args.mask_dir)