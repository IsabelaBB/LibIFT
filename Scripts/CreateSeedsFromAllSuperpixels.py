import os
import argparse
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Create Seeds From All Superpixels")
    parser.add_argument("spx_image", type=str, help="Path to the superpixel image.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    parser.add_argument("label", type=int, help="Label for the seeds.")
    return parser.parse_args()

def read_img(path):
    img = np.array(Image.open(path))
    if img is None:
        raise ValueError(f"Failed to read image '{path}'.")
    return img

def get_seeds_from_superpixels(img):
    visited = set()
    seeds = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            spx_id = img[x,y]
            if spx_id not in visited and spx_id != 0:
                visited.add(spx_id)
                seeds.append((x,y))
    return seeds

def write_seeds_file(seeds, output_path, basename, img_shape, label):
    seeds_path = os.path.join(output_path, f"{basename}-seeds.txt")
    with open(seeds_path, 'w') as f:
        f.write("{} {} {}\n".format(len(seeds), img_shape[1], img_shape[0]))
        for x, y in seeds:
            f.write(f"{y} {x} -1 {label} 0\n")

def run(spx_image_file, output_path, label):

    if label < 0:
        raise ValueError("Label must be non-negative.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    basename = os.path.basename(spx_image_file).split('.')[0]
    img = np.array(Image.open(spx_image_file))
    if len(img.shape) != 2:
        img = img[:, :, 0]
    visited = set()
    seeds = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            spx_id = img[x,y]
            if spx_id not in visited and spx_id != 0:
                visited.add(spx_id)
                seeds.append((x,y))

    write_seeds_file(seeds, output_path, basename, img.shape, label)

if __name__ == "__main__":
    args = parse_args()
    run(args.spx_image, args.output_path, args.label)