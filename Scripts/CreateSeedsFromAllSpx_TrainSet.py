import os
import argparse
from CreateSeedsFromAllSuperpixels import run

def parse_args():
    parser = argparse.ArgumentParser("Create Seeds From All Superpixels")
    parser.add_argument("train_file", type=str, help="Path to the training file.")
    parser.add_argument("spx_images_dir", type=str, help="Path to the superpixel images directory.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    return parser.parse_args()

def get_train_filenames(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    image_filenames = [os.path.basename(line.strip()).split('.')[0] for line in lines]
    if not image_filenames:
        raise ValueError(f"No image filenames found in the training file '{path}'.")
    return image_filenames

if __name__ == "__main__":
    args = parse_args()

    image_filenames = get_train_filenames(args.train_file)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    for img_name in image_filenames:
        spx_img_path = os.path.join(args.spx_images_dir, img_name + '.pgm')
        if os.path.isfile(spx_img_path):
            label = int(img_name.split('_')[0])
            run(spx_img_path, args.output_path, label=label)