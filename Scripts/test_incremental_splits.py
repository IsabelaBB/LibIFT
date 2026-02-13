import json
import argparse
import os
from tqdm import tqdm
import random
import sys
import shutil
import subprocess
import numpy as np

tmp_file = "/tmp/test_incremental_splits_log.txt"
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

def strToList(s):
    return s.split(',')

def vector_from_graph(graph_data):
    num_nodes = int(graph_data['num_nodes'])
    num_feats = int(graph_data['num_feats'])
    feat_matrix = np.zeros((num_nodes, num_feats))
    for i in range(int(graph_data['num_nodes'])):
        node_feats = graph_data['node'+str(i)]['feats']
        feat_matrix[i, :] = np.array(node_feats)
    
    # sort the values of each feature column and append into a single vector
    vector = []
    for j in range(num_feats):
        sorted_column = np.sort(feat_matrix[:, j])
        vector.extend(sorted_column.tolist())
    return vector


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
    parser.add_argument("--c_vals", help="C values for training.", type=strToList, default=["1e2"])
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
    c_vals = args.c_vals
    split_files = [
        f for f in os.listdir(args.inc_split_file) if f.endswith(".json")
    ]

    for data_descriptor_file in split_files:

        perc = data_descriptor_file.split('_')[-1].replace('perc','').split('.')[0]
        pre_descr = f"Split {split} with {perc}% data"

        with tqdm(range(0,4 + len(c_vals)*(1 + nlayers)), desc=f"{pre_descr} - Preparing data", unit="step", leave=False) as pbar:
        
            data_descriptor_path = os.path.join(args.inc_split_file, data_descriptor_file)
            with open(data_descriptor_path, 'r') as f:
                data_descriptor = json.load(f)

            # set model directory
            model_dir=f"{IFT_dir}/Results/protozoan-cysts/gflimBoFP_incremental_splits_1e2/split{split}/perc{perc}/{arch_name}/{initial_seeds}-{spx}spx_{kernel_neighbors}-kernel_{pool_neighbors}-pool_{stride_neighbors}-stride"
            if fixed_spx == 1: 
                model_dir=f"{model_dir}/fixedSPX_{graph_feat}_{aggregation_opt}_{attention}"
            else:
                model_dir=f"{model_dir}/notFixedSPX_{graph_feat}_{aggregation_opt}_{attention}"

            # create model directory if it does not exist
            if os.path.exists(model_dir) == False:
                print(f"Model directory {model_dir} does not exist. Skipping.")
                continue

            # clean previous data in model directory
            for layer in range(0,nlayers+1):
                shutil.rmtree(f"{model_dir}/layer{layer}", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/labels{layer}", ignore_errors=True)

            # create csv files for train, validation and test
            test_csv = f"{model_dir}/test{split}.csv"
            #begin comment
            with open(test_csv, 'w') as f_test:
                for item in data_descriptor['test']:
                    f_test.write(f"{dataset_path}/images/{item}\n")
            #end comment

            # extract test set features and classify
            pbar.set_description_str(f"{pre_descr} - Extract features from test set")
            pbar.update(1)
            run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
                {test_csv} {model_dir} {arch_file} \
                {nlayers} {batch_size} {attention} {aggregation_opt} \
                {graph_feat} DISF {spx} {fixed_spx} \
                1 -1 -1 graph-adaptive \
                {kernel_neighbors} {pool_neighbors} {stride_neighbors} \
                {initial_seeds} 1")

            #begin comment
            pbar.set_description_str(f"{pre_descr} - Converting activation graphs in dataset format")
            pbar.update(1)
            for layer in range(1,nlayers+1):
                activations_dir=f"{model_dir}/layer{layer}_test"
                run_script(f"cp -r {model_dir}/layer{layer} {activations_dir}")
                run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
                        {activations_dir} \
                        {test_csv} 1 1")
            #end comment

            """
            os.makedirs(os.path.join(model_dir,"layer1_test_vectors"), exist_ok=True)
            for graph_file in os.listdir(f"{model_dir}/layer1"):
                with open(os.path.join(f"{model_dir}/layer1", graph_file), "r") as gf:
                    graph_data = json.load(gf)
                vector = vector_from_graph(graph_data)
                vector_file = os.path.join(model_dir,"layer1_test_vectors", graph_file.replace(".json", ".npy"))
                np.save(vector_file, vector)
            """
            
            pbar.set_description_str(f"{pre_descr} - Cleaning data layer and labels folders")
            pbar.update(1)
            for layer in range(1,nlayers+1):
                shutil.rmtree(f"{model_dir}/layer{layer}_test", ignore_errors=True) #uncommented
                shutil.rmtree(f"{model_dir}/labels{layer}", ignore_errors=True)
                shutil.rmtree(f"{model_dir}/layer{layer}", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/layer0", ignore_errors=True)
            shutil.rmtree(f"{model_dir}/labels0", ignore_errors=True)

            #begin comment
            for c_val in c_vals:

                # classify test set with trained SVMs
                for layer in range(1,nlayers+1):
                    pbar.set_description_str(f"{pre_descr} - Classify test set for layer {layer}")
                    pbar.update(1)
                    activations_dir=f"{model_dir}/layer{layer}_test"
                    svm_file=f"{model_dir}/svm_{layer}_{c_val}.zip"
                    run_script(f"cp {activations_dir}.zip {activations_dir}_{c_val}.zip")
                    classification_output = run_script(f"{IFT_dir}/bin/iftClassifyBySVM \
                        {activations_dir}_{c_val}.zip \
                        {svm_file} \
                        {model_dir}/results_layer{layer}_test_{c_val}")
                    with open(f"{model_dir}/classification_layer{layer}_test_{c_val}.txt", "w") as f:
                        f.write(classification_output)

                pbar.set_description_str(f"{pre_descr} - Cleaning data after SVM classification")
                pbar.update(1)
                for layer in range(1,nlayers+1):
                    os.remove(f"{model_dir}/layer{layer}_test_{c_val}.zip")

            pbar.set_description_str(f"{pre_descr} - Cleaning data after test classification")
            pbar.update(1)
            # clean up activations and layer models to save space
            for layer in range(1,nlayers+1):
                activations_dir=f"{model_dir}/layer{layer}_test"
                shutil.rmtree(f"{activations_dir}", ignore_errors=True)
                os.remove(f"{activations_dir}.zip")
            #end comment