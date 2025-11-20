"""
Add images to the training set based on misclassification certainty.
"""

import os
import sys

def main():
    # Check command line arguments
    if len(sys.argv) != 5:
        print("Usage: python add_images_train_gflim.py <dataset_path> <num_classes> <split> <input_csv>")
        print("dataset path: path to dataset")
        print("split: split identifier (integer)")
        print("misclassified images csv file")
        print("num_classes: number of classes")
        sys.exit(1)

    dataset_path = sys.argv[1]
    split = int(sys.argv[2])
    input_csv = sys.argv[3]
    num_classes = int(sys.argv[4])

    # Read misclassified.csv and create list of image entries
    with open(input_csv, "r") as f:
        image_lines = [line.strip() for line in f]

    # Remove header if present
    if image_lines and image_lines[0].startswith("Misclassified Image; Certainty;"):
        image_lines.pop(0)

    print(f"Total images in misclassified.csv: {len(image_lines)}")

    # Build dictionary mapping image names to certainty values
    image_certainty = {}
    for line in image_lines:
        # Extract image name and certainty value
        name = line.split("/")[-1].split(";")[0]
        certainty = float(line.split(";")[1])
        image_certainty[name] = certainty

    # Sort images by certainty in descending order
    sorted_images = dict(sorted(image_certainty.items(), key=lambda item: item[1], reverse=True))

    # Create a list to hold the training file paths
    train_files = []
    with open(os.path.join(dataset_path, "build", f"train{split}.csv"), "r") as f:
        train_files = [line.strip() for line in f.readlines()]

    # Select and copy images for each class
    for class_id in range(1, num_classes + 1):
        # Filter images belonging to the current class
        class_images = [name for name in sorted_images.keys() if int(name.split("_")[0]) == class_id]
        class_images.sort(key=lambda x: sorted_images[x], reverse=True)

        # Try to write the image with highest certainty in the train csv file, skip if already exists
        for img in class_images:
            img_name = img.split("/")[-1]
            img_path = os.path.join(dataset_path,"images", img)
            if img_path not in train_files:
                train_files.append(img_path)
                print(f"Selected image for class {class_id}: {img} {sorted_images[img]}")
                break
            else:
                print(f"Image {img_name} already exists, trying next for class {class_id}.")

    # Write the selected training files to the train CSV
    with open(os.path.join(dataset_path, "build", f"train{split}.csv"), "w") as f:
        for train_file in train_files:
            f.write(f"{train_file}\n")

    # remove train files from train-val csv
    val_files = []
    with open(os.path.join(dataset_path, "build", f"train-val{split}.csv"), "r") as f:
        val_files = [line.strip() for line in f.readlines()]

    # Remove train files from val files
    val_files = [f for f in val_files if f not in train_files]

    with open(os.path.join(dataset_path, "build", f"train-val{split}.csv"), "w") as f:
        for val_file in val_files:
            f.write(f"{val_file}\n")

if __name__ == "__main__":
    main()
