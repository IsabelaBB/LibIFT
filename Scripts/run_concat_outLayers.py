"""
given a trained GFLIM model, concatenate the output layers to train an SVM classifier and evaluate its performance
"""

import os
import sys
import argparse
import subprocess
import json
from PIL import Image
import numpy as np
import shutil

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

tmp_file = "/tmp/run_concat_outLayers_log.txt"
IFT_dir="/home/isabela/Codes/LibIFT"

def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate output layers from GFLIM model and train SVM")
    #parser.add_argument("imgs_train_path", help="Path to the dataset.", type=str)
    #parser.add_argument("imgs_eval_path", help="Path to the dataset.", type=str)
    parser.add_argument("data_descriptor_perc_path", help="Path to the dataset.", type=str)
    parser.add_argument("dataset_path", help="Path to the dataset.", type=str)
    parser.add_argument("model_path", help="Path to the trained GFLIM model.", type=str)
    parser.add_argument("arch_file", help="Path to the architecture file.", type=str)
    parser.add_argument("num_classes", help="Number of classes.", type=int)
    parser.add_argument("spx", help="Final number of superpixels.", type=int) 
    parser.add_argument("init_seeds", help="Initial number of seeds.", type=int) 
    parser.add_argument("kernel_neigh", help="Kernel neighbors.", type=str) 
    parser.add_argument("pool_neigh", help="Pool neighbors.", type=str) 
    parser.add_argument("stride_neighbors", help="Stride neighbors.", type=str) 
    parser.add_argument("graph_feat", help="Graph feature.", type=str) 
    parser.add_argument("aggregation_opt", help="Aggregation option.", type=str)
    parser.add_argument("attention", help="Attention.", type=str)
    parser.add_argument("--fixed_spx", help="Use fixed superpixels.", type=int, default=0)
    parser.add_argument("--random_state", help="Random state for reproducibility.", type=int, default=42)
    parser.add_argument("--num_images_per_class", help="Number of images per class for GFLIM training.", type=int, default=1)
    parser.add_argument("--batch_size", help="Batch size for processing.", type=int, default=64)
    parser.add_argument("--nlayers", help="Number of layers.", type=int, default=3)
    return parser.parse_args()

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

def concatenate_graph_features(model_path, nlayers, concatenated_output_dir):
    first_layer_output = f"{model_path}/layer1"
    filenames = [f for f in os.listdir(first_layer_output) if f.endswith(".json")]
    
    # Create a partial function with fixed parameters
    process_fn = partial(_process_single_file, model_path, nlayers, concatenated_output_dir, first_layer_output)
    
    # Process files in parallel
    with Pool() as pool:
        pool.map(process_fn, filenames)

def _process_single_file(model_path, nlayers, concatenated_output_dir, first_layer_output, filename):
    base_name = os.path.basename(filename)
    
    with open(os.path.join(first_layer_output, filename), 'r') as file:
        graph_1 = json.load(file)
    
    nfeats = graph_1['num_feats']

    spx_img_1 = Image.open(os.path.join(model_path, "labels1", base_name.replace(".json", ".pgm")))
    spx_img_1 = np.array(spx_img_1.convert("I;16"))

    for layer in range(2, nlayers + 1):
        try:
            spx_img_i = Image.open(os.path.join(model_path, "labels" + str(layer), base_name.replace(".json", ".pgm")))
            spx_img_i = np.array(spx_img_i.convert("I;16"))
        except FileNotFoundError:
            print(f"Error: Superpixel image for layer {layer} not found.")
            continue

        node_mapping = {}
        for i in range(spx_img_1.shape[0]):
            for j in range(spx_img_1.shape[1]):
                spx_1 = spx_img_1[i, j] - 1
                spx_i = spx_img_i[i, j] - 1
                if spx_1 < 0 or spx_i < 0:
                    continue
                if spx_1 not in node_mapping:
                    node_mapping[spx_1] = set()
                node_mapping[spx_1].add(spx_i)

        try:
            with open(os.path.join(f"{model_path}/layer{layer}", filename), 'r') as file:
                graph_i = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file '{filename}'.")
            continue
        
        for node_1, mapped_nodes in node_mapping.items():
            concatenated_features = []
            if 'node'+str(node_1) in graph_1:
                concatenated_features.extend(graph_1['node'+str(node_1)]['feats'])
            
            mapped_feats = [graph_i['node'+str(node_i)]['feats'] 
                            for node_i in mapped_nodes 
                            if 'node'+str(node_i) in graph_i]
            if mapped_feats:
                mapped_feats = np.array(mapped_feats)
                avg_feats = np.mean(mapped_feats, axis=0).tolist()
                concatenated_features.extend(avg_feats)
            
            if 'node'+str(node_1) in graph_1:
                graph_1['node'+str(node_1)]['feats'] = concatenated_features
        
        nfeats += graph_i['num_feats']

    graph_1['num_feats'] = nfeats
    with open(os.path.join(concatenated_output_dir, filename), 'w') as out_file:
        json.dump(graph_1, out_file)

if __name__ == "__main__":
    args = parse_args()

    #imgs_train_path = args.imgs_train_path
    #imgs_eval_path = args.imgs_eval_path
    data_descriptor_perc_path = args.data_descriptor_perc_path
    dataset_path = args.dataset_path
    model_path = args.model_path
    arch_file = args.arch_file
    num_classes = args.num_classes
    spx = args.spx
    initial_seeds = args.init_seeds
    kernel_neighbors = args.kernel_neigh
    pool_neighbors = args.pool_neigh
    stride_neighbors = args.stride_neighbors
    graph_feat = args.graph_feat
    aggregation_opt = args.aggregation_opt
    attention = args.attention
    fixed_spx = args.fixed_spx
    random_state = args.random_state
    num_images_per_class = args.num_images_per_class
    batch_size = args.batch_size
    nlayers = args.nlayers

    """
    if os.path.exists(imgs_train_path) is False or os.path.isdir(imgs_train_path) is False:
        print(f"Training images directory {imgs_train_path} does not exist.")
        sys.exit(1)
    if os.path.exists(imgs_eval_path) is False or os.path.isdir(imgs_eval_path) is False:
        print(f"Evaluation images directory {imgs_eval_path} does not exist.")
        sys.exit(1)
    """

    if os.path.exists(data_descriptor_perc_path) is False:
        print(f"Data descriptor percentage file {data_descriptor_perc_path} does not exist.")
        sys.exit(1)
    if os.path.isdir(model_path) is False:
        print(f"Model path {model_path} does not exist.")
        sys.exit(1)
    if os.path.exists(arch_file) is False:
        print(f"Architecture file {arch_file} does not exist.")
        sys.exit(1)

    with open(data_descriptor_perc_path, 'r') as f:
        data_descriptor = json.load(f)
    
    train_csv = f"{model_path}/concatenate_{nlayers}layers_trainSVM.csv"
    val_csv = f"{model_path}/concatenate_{nlayers}layers_val.csv"
    test_csv = f"{model_path}/concatenate_{nlayers}layers_test.csv"
    with open(train_csv, 'w') as f_train:
        for item in data_descriptor['training']:
            f_train.write(f"{dataset_path}/images/{item}\n")
    with open(val_csv, 'w') as f_val:
        for item in data_descriptor['validation']:
            f_val.write(f"{dataset_path}/images/{item}\n")
    with open(test_csv, 'w') as f_test:
        for item in data_descriptor['test']:
            f_test.write(f"{dataset_path}/images/{item}\n")

    with tqdm(range(0,15), desc=f"Extracting SVM training data", unit="step", leave=False) as pbar:

        # extracts features from the dataset
        run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
            {train_csv} \
            {model_path} 1 {spx} {initial_seeds} {fixed_spx} {graph_feat} \
            {arch_file} {nlayers} {batch_size} {attention} {aggregation_opt} \
            1 -1 -1 graph-adaptive \
            {kernel_neighbors} {pool_neighbors} {stride_neighbors}")

        pbar.set_description_str(f"Concatenating output layers of SVM training data")
        pbar.update(1)

        first_layer_output = f"{model_path}/layer1"
    
        concatenated_output_dir = f"{model_path}/concatenated_{nlayers}layers_trainSVM"
        if not os.path.exists(concatenated_output_dir):
            os.makedirs(concatenated_output_dir)

        # concatenates the output layers
        concatenate_graph_features(model_path, nlayers, concatenated_output_dir)

        pbar.set_description_str(f"Converting graphs files in a dataset file")
        pbar.update(1)
        
        # trains a SVM using the concatenated graph
        svm_file=f"{model_path}/svm_concatenated_{nlayers}layers.zip"
            
        run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
               {concatenated_output_dir} \
                {train_csv} 1 1")

        pbar.set_description_str(f"Training SVM using concatenated output layers")
        pbar.update(1)

        run_script(f"{IFT_dir}/bin/iftSupTrainBySVM \
               {concatenated_output_dir}.zip 0 0 1e2 0 {svm_file}")
    
        # clean temporary files
        shutil.rmtree(concatenated_output_dir)
        os.remove(f"{concatenated_output_dir}.zip")
        for layer in range(nlayers+1):
            shutil.rmtree(f"{model_path}/layer{layer}")
            shutil.rmtree(f"{model_path}/labels{layer}")

        for setName, set_csv in zip(['val', 'test'], [val_csv, test_csv]):

            pbar.set_description_str(f"Extracting SVM {setName} data")
            pbar.update(1)
            # extracts features from the evaluation dataset
            run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
                        {set_csv} \
                        {model_path} 1 {spx} {initial_seeds} {fixed_spx} {graph_feat} \
                        {arch_file} {nlayers} {batch_size} {attention} {aggregation_opt} \
                        1 -1 -1 graph-adaptive \
                        {kernel_neighbors} {pool_neighbors} {stride_neighbors}")

            concatenated_output_dir = f"{model_path}/concatenated_{nlayers}layers_{setName}SVM"
            if not os.path.exists(concatenated_output_dir):
                os.makedirs(concatenated_output_dir)

            pbar.set_description_str(f"Concatenating output layers of SVM {setName} data")
            pbar.update(1)
            # conatenates the output layers for the evaluation dataset
            concatenate_graph_features(model_path, nlayers, concatenated_output_dir)

            pbar.set_description_str(f"Converting graphs files in a dataset file for SVM {setName} data")
            pbar.update(1)
            # classifies the evaluation dataset using the trained SVM
            run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
                    {concatenated_output_dir} \
                        {set_csv} 1 1")
            
            pbar.set_description_str(f"Classifying SVM {setName} data using concatenated output layers")
            pbar.update(1)
            classification_output = run_script(f"{IFT_dir}/bin/iftClassifyBySVM \
                                        {concatenated_output_dir}.zip \
                                            {svm_file} \
                                            {model_path}/results_concatenated_{nlayers}layers_{setName}")
            
            pbar.set_description_str(f"Saving classification results for SVM {setName} data")
            pbar.update(1)
            # save classification output
            with open(f"{model_path}/classification_concatenated_{nlayers}layers_{setName}.txt", "w") as f:
                f.write(classification_output)
            
            pbar.set_description_str(f"Cleaning temporary files for SVM {setName} data")
            pbar.update(1)
            # clean temporary files
            shutil.rmtree(concatenated_output_dir)
            os.remove(f"{concatenated_output_dir}.zip")
            for layer in range(nlayers+1):
                shutil.rmtree(f"{model_path}/layer{layer}")
                shutil.rmtree(f"{model_path}/labels{layer}")