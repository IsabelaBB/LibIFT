"""
Train a GFLIM-BoFP model incrementally using different percentages of the training set of a data split.
The number of images per class used for GFLIM training can be specified, but its fixed for all percentages.
For each percentage, the model is trained and SVMs are created for each layer.
The model is then used to extract features from the validation set and classify it using the trained SVMs.
Results are saved in the model directory.
"""

import json
import argparse
import os
from tqdm import tqdm
import random
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

tmp_file = "/tmp/run_incremental_splits_log.txt"
IFT_dir="/home/isabela/Codes/LibIFT"
pre_descr = ""

def run_script(cmd):
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if tmp_file is not None:
        with open(tmp_file, "a+") as f:
            f.write(process.stdout)
    if process.returncode != 0:
        print(f"Error running command: {cmd}")
        print(process.stderr)
        sys.exit(process.returncode)
    return process.stdout

def create_graphs_and_labels(train_csv_GFLIM, spx, initial_seeds, model_dir, 
                             graph_feat, arch_file, fixed_spx, 
                             kernel_neighbors, pool_neighbors, stride_neighbors):
    # create graphs and spx labels
    if pool_neighbors == "color" or stride_neighbors == 'color' or kernel_neighbors == 'color':
        run_script(f"{IFT_dir}/bin/iftGFLIMCreateNeighbMatricesAndGraphs \
            {train_csv_GFLIM} \
            {spx} {initial_seeds} \
            {model_dir} {graph_feat} \
            {arch_file} 1 {fixed_spx} \
            {kernel_neighbors} {pool_neighbors} {stride_neighbors}")
    else:
        run_script(f"{IFT_dir}/bin/iftCreateGFLIMGraph \
            {train_csv_GFLIM} \
            {spx} \
            {model_dir} {graph_feat} DISF {initial_seeds} 1")

def create_seeds(model_dir, dataset_path, train_csv, 
                 kernel_neighbors, n_kernels, kernel_size, 
                 aggregation_opt, attention):
    
    # create image seeds with all superpixels
    markers_folder = f"{model_dir}/img_seeds"
    os.makedirs(markers_folder, exist_ok=True)
    run_script(f"python3 {IFT_dir}/Scripts/CreateSeedsFromAllPix_TrainSet.py \
        {train_csv} \
        {dataset_path}/masks/ \
        {markers_folder}")

    # create initial graph seeds
    if kernel_neighbors == "color": 
        run_script(f"{IFT_dir}/bin/iftGFLIMPrecConvertSeeds \
            {model_dir} \
            {markers_folder} \
            {n_kernels} {kernel_size} {aggregation_opt}")
    else:
        run_script(f"{IFT_dir}/bin/iftGFLIMConvertSeeds \
            {model_dir} \
            {markers_folder} \
            {n_kernels} {kernel_size} \
            {attention} {aggregation_opt}")

def train_gflim(model_dir, arch_file, dataset_path, 
                kernel_neighbors, pool_neighbors, stride_neighbors, 
                graph_feat, aggregation_opt, attention, nlayers):
    
    # create layer models and extract activations
    for layer in range(1, 4):

        if kernel_neighbors == "color" or stride_neighbors == "color":
            run_script(f"{IFT_dir}/bin/iftGFLIMPrecCreateLayerModel \
                {model_dir} \
                {arch_file} {layer} \
                {kernel_neighbors} {stride_neighbors} \
                {graph_feat} {aggregation_opt} {dataset_path}")
        else:
            run_script(f"{IFT_dir}/bin/iftGFLIMCreateLayerModel \
                {model_dir} \
                {arch_file} {layer} \
                graph-based {attention} {aggregation_opt}")

        run_script(f"{IFT_dir}/bin/iftGFLIMMergeLayerModels \
            {model_dir} {layer}")

        if kernel_neighbors == "color" or pool_neighbors == "color" or stride_neighbors == "color": 
            run_script(f"{IFT_dir}/bin/iftGFLIMEncodeMergedLayerBatchV3 \
                {model_dir} {arch_file} {layer} \
                {batch_size} {attention} {kernel_neighbors} \
                {pool_neighbors} {stride_neighbors} \
                {graph_feat} {aggregation_opt} {dataset_path}")
        else:
            run_script(f"{IFT_dir}/bin/iftGFLIMEncodeMergedLayerBatch \
                {model_dir} {arch_file} {layer} {batch_size} graph-based \
                {attention} {aggregation_opt}")

        run_script(f"{IFT_dir}/bin/iftGFLIMWriteActivations \
            {model_dir} {layer} {batch_size}")

def read_image_names(csv_path):
    # Read image names from the CSV file and sort them
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

def copy_selected_images(selected_images, train_csv_GFLIM):
    # Copy selected images to the output directory for the split
    for image in selected_images:
        # write image paths in a csv file
        with open(train_csv_GFLIM, "w") as f:
            for image in selected_images:
                f.write(f"{image}\n")

def parse_args():
    parser = argparse.ArgumentParser("Train and extract GFLIM-BoFP model.")

    # madatory arguments
    parser.add_argument("inc_split_file", help="Path to the incremental split file.", type=str)
    parser.add_argument("dataset_path", help="Path to the dataset.", type=str)
    parser.add_argument("num_classes", help="Number of classes.", type=int)
    parser.add_argument("split_id", help="Split id.", type=str) 
    parser.add_argument("spx", help="Final number of superpixels.", type=int) 
    parser.add_argument("init_seeds", help="Initial number of seeds.", type=int) 
    parser.add_argument("kernel_size", help="Kernel size.", type=int) 
    parser.add_argument("kernel_neigh", help="Kernel neighbors.", type=str) 
    parser.add_argument("pool_size", help="Pool size.", type=int)
    parser.add_argument("pool_neigh", help="Pool neighbors.", type=str) 
    parser.add_argument("pool_name", help="Pool name.", type=str)
    parser.add_argument("nkernels", help="Number of kernels.", type=int)
    parser.add_argument("stride_size", help="Stride size.", type=int)
    parser.add_argument("stride_neighbors", help="Stride neighbors.", type=str) 
    parser.add_argument("graph_feat", help="Graph feature.", type=str) 
    parser.add_argument("aggregation_opt", help="Aggregation option.", type=str)
    parser.add_argument("attention", help="Attention.", type=str)
    # optional arguments
    parser.add_argument("--fixed_spx", help="Use fixed superpixels.", type=int, default=0)
    parser.add_argument("--random_state", help="Random state for reproducibility.", type=int, default=42)
    parser.add_argument("--num_images_per_class", help="Number of images per class for GFLIM training.", type=int, default=1)
    parser.add_argument("--nlayers", help="Number of layers in the GFLIM model.", type=int, default=3)
    parser.add_argument("--batch_size", help="Batch size for training.", type=int, default=64)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

    arch_name=f"{args.kernel_size}-1-1-{args.nkernels}-{args.pool_name}-{args.pool_size}-{args.stride_size}_3_noSkip"

    # check if arch file exists
    arch_file=f"{IFT_dir}/Datasets/archs/{arch_name}.json"
    if os.path.isfile(arch_file) == False:
        print(f"Architecture file {arch_file} does not exist. Exiting.")
        exit(-1)

    if os.path.isdir(args.inc_split_file) == False:
        print(f"Incremental split file {args.inc_split_file} does not exist. Exiting.")
        exit(-1)

    split = args.split_id
    dataset_path = args.dataset_path
    num_classes = args.num_classes
    spx = args.spx
    initial_seeds = args.init_seeds
    kernel_size = args.kernel_size
    kernel_neighbors = args.kernel_neigh
    pool_size = args.pool_size
    pool_neighbors = args.pool_neigh
    stride_size = args.stride_size
    stride_neighbors = args.stride_neighbors
    n_kernels = args.nkernels
    graph_feat = args.graph_feat
    aggregation_opt = args.aggregation_opt
    attention = args.attention
    fixed_spx = args.fixed_spx
    random_state = args.random_state
    num_images_per_class = args.num_images_per_class
    nlayers = args.nlayers
    batch_size = args.batch_size
    
    split_files = [
        f for f in os.listdir(args.inc_split_file) if f.endswith(".json")
    ]

    for data_descriptor_file in split_files:

        perc = data_descriptor_file.split('_')[-1].replace('perc','').split('.')[0]

        pre_descr = f"Split {split} with {perc}% data"

        with tqdm(range(0,14), desc=f"{pre_descr} - Preparing data", unit="step", leave=False) as pbar:
        
            data_descriptor_path = os.path.join(args.inc_split_file, data_descriptor_file)
            with open(data_descriptor_path, 'r') as f:
                data_descriptor = json.load(f)

            # set model directory
            model_dir=f"{IFT_dir}/Results/helminth-eggs/gflimBoFP_incremental_splits/split{split}/perc{perc}/{arch_name}/{initial_seeds}-{spx}spx_{kernel_neighbors}-kernel_{pool_neighbors}-pool_{stride_neighbors}-stride"
            if fixed_spx == 1: 
                model_dir=f"{model_dir}/fixedSPX_{graph_feat}_{aggregation_opt}_{attention}"
            else:
                model_dir=f"{model_dir}/notFixedSPX_{graph_feat}_{aggregation_opt}_{attention}"

            # create model directory if it does not exist
            os.makedirs(model_dir, exist_ok=True)

            # create csv files for train, validation and test
            train_csv = f"{model_dir}/train{split}.csv"
            val_csv = f"{model_dir}/val{split}.csv"
            test_csv = f"{model_dir}/test{split}.csv"
            with open(train_csv, 'w') as f_train:
                for item in data_descriptor['training']:
                    f_train.write(f"{dataset_path}/images/{item}\n")
            with open(val_csv, 'w') as f_val:
                for item in data_descriptor['validation']:
                    f_val.write(f"{dataset_path}/images/{item}\n")
            with open(test_csv, 'w') as f_test:
                for item in data_descriptor['test']:
                    f_test.write(f"{dataset_path}/images/{item}\n")

            # create GFLIM train csv file
            train_csv_GFLIM = f"{model_dir}/train{split}-GFLIM.csv"
            image_names = read_image_names(train_csv)
            labels_to_images = group_images_by_label(image_names)
            selected_images = select_images(labels_to_images, num_classes, num_images_per_class, random_state)
            copy_selected_images(selected_images, train_csv_GFLIM)

            pbar.set_description_str(f"{pre_descr} - Create graphs and spx labels")
            pbar.update(1)
            create_graphs_and_labels(train_csv_GFLIM, spx, initial_seeds, model_dir, 
                                    graph_feat, arch_file, fixed_spx, 
                                    kernel_neighbors, pool_neighbors, stride_neighbors)

            pbar.set_description_str(f"{pre_descr} - Create seeds")
            pbar.update(1)
            create_seeds(model_dir, dataset_path, train_csv_GFLIM, 
                        kernel_neighbors, n_kernels, kernel_size, 
                        aggregation_opt, attention)

            pbar.set_description_str(f"{pre_descr} - Train the model")
            pbar.update(1)
            train_gflim(model_dir, arch_file, train_csv_GFLIM, 
                        kernel_neighbors, pool_neighbors, stride_neighbors, 
                        graph_feat, aggregation_opt, attention, nlayers)

            pbar.set_description_str(f"{pre_descr} - Extract features from SVM training set")
            pbar.update(1)
            # extracts features from the train-val set
            run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
                {train_csv} \
                {model_dir} 1 {spx} {initial_seeds} {fixed_spx} {graph_feat} \
                {arch_file} {nlayers} {batch_size} {attention} {aggregation_opt} \
                1 -1 -1 graph-adaptive \
                {kernel_neighbors} {pool_neighbors} {stride_neighbors}")

            # train SVMs for each layer
            for layer in range(1,nlayers+1):
                pbar.set_description_str(f"{pre_descr} - Train SVM for layer {layer}")
                pbar.update(1)

                activations_dir=f"{model_dir}/layer{layer}_trainSVM"
                svm_file=f"{model_dir}/svm_{layer}.zip"
                
                run_script(f"cp -r {model_dir}/layer{layer} {activations_dir}")

                run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
                    {activations_dir} \
                    {train_csv} 1 1")

                run_script(f"{IFT_dir}/bin/iftSupTrainBySVM \
                    {activations_dir}.zip 0 0 1e2 0 {svm_file}")

            pbar.set_description_str(f"{pre_descr} - Cleaning data after SVM training")
            pbar.update(1)
            # clean up activations and layer models to save space
            for layer in range(1,nlayers+1):
                activations_dir=f"{model_dir}/layer{layer}_trainSVM"
                shutil.rmtree(f"{activations_dir}", ignore_errors=True)
                shutil.rmtree(f"{activations_dir}.zip", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/labels{layer}", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/layer{layer}", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/layer0", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/labels0", ignore_errors=True)

            # extract test set features and classify
            pbar.set_description_str(f"{pre_descr} - Extract features from validation set")
            pbar.update(1)
            run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
                {val_csv} \
                {model_dir} 1 {spx} {initial_seeds} {fixed_spx} {graph_feat} \
                {arch_file} {nlayers} {batch_size} {attention} {aggregation_opt} \
                1 -1 -1 graph-adaptive \
                {kernel_neighbors} {pool_neighbors} {stride_neighbors}")

            # classify test set with trained SVMs
            for layer in range(1,nlayers+1):
                pbar.set_description_str(f"{pre_descr} - Classify validation set for layer {layer}")
                pbar.update(1)

                activations_dir=f"{model_dir}/layer{layer}_val"
                svm_file=f"{model_dir}/svm_{layer}.zip"
                run_script(f"cp -r {model_dir}/layer{layer} {activations_dir}")

                run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
                    {activations_dir} \
                    {val_csv} 1 1")

                classification_output = run_script(f"{IFT_dir}/bin/iftClassifyBySVM \
                    {activations_dir}.zip \
                    {svm_file} \
                    {model_dir}/results_layer{layer}_val")
            
                with open(f"{model_dir}/classification_layer{layer}_val.txt", "w") as f:
                    f.write(classification_output)

            pbar.set_description_str(f"{pre_descr} - Cleaning data after validation classification")
            pbar.update(1)
            # clean up activations and layer models to save space
            for layer in range(1,nlayers+1):
                activations_dir=f"{model_dir}/layer{layer}_val"
                shutil.rmtree(f"{activations_dir}", ignore_errors=True)
                shutil.rmtree(f"{activations_dir}.zip", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/layer{layer}", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/labels{layer}", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/layer0", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/labels0", ignore_errors=True)

