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
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from scipy import stats
import joblib
import pandas as pd

tmp_file = "/tmp/finetuneSVM_log.txt"
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

class Model():
    def __init__(self, model_dir, arch_file,
                 dataset_path, num_classes,
                 spx, initial_seeds, kernel_size, kernel_neighbors,
                 pool_size, pool_neighbors, stride_size, stride_neighbors,
                 n_kernels, graph_feat, aggregation_opt, attention,
                 fixed_spx, nlayers, batch_size):
        self.model_dir = model_dir
        self.arch_file = arch_file
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.spx = spx
        self.initial_seeds = initial_seeds
        self.kernel_size = kernel_size
        self.kernel_neighbors = kernel_neighbors
        self.pool_size = pool_size
        self.pool_neighbors = pool_neighbors
        self.stride_size = stride_size
        self.stride_neighbors = stride_neighbors
        self.n_kernels = n_kernels
        self.graph_feat = graph_feat
        self.aggregation_opt = aggregation_opt
        self.attention = attention
        self.fixed_spx = fixed_spx
        self.nlayers = nlayers
        self.batch_size = batch_size
        self.svm_file = None

    def create_graphs_and_labels(self, train_csv_GFLIM):
        # create graphs and spx labels
        if self.pool_neighbors == "color" or self.stride_neighbors == 'color' or self.kernel_neighbors == 'color':
            run_script(f"{IFT_dir}/bin/iftGFLIMCreateNeighbMatricesAndGraphs \
                {train_csv_GFLIM} \
                {self.spx} {self.initial_seeds} \
                {self.model_dir} {self.graph_feat} \
                {self.arch_file} 1 {self.fixed_spx} \
                {self.kernel_neighbors} {self.pool_neighbors} {self.stride_neighbors}")
        else:
            run_script(f"{IFT_dir}/bin/iftCreateGFLIMGraph \
                {train_csv_GFLIM} \
                {self.spx} \
                {self.model_dir} {self.graph_feat} DISF {self.initial_seeds} 1")

    def create_seeds(self, train_path):
        
        # create image seeds with all superpixels
        markers_folder = f"{self.model_dir}/img_seeds"
        os.makedirs(markers_folder, exist_ok=True)
        run_script(f"python3 {IFT_dir}/Scripts/CreateSeedsFromAllPix_TrainSet.py \
            {train_path} \
            {self.dataset_path}/masks/ \
            {markers_folder}")

        # create initial graph seeds
        if self.kernel_neighbors == "color": 
            run_script(f"{IFT_dir}/bin/iftGFLIMPrecConvertSeeds \
                {self.model_dir} \
                {markers_folder} \
                {self.n_kernels} {self.kernel_size} {self.aggregation_opt}")
        else:
            run_script(f"{IFT_dir}/bin/iftGFLIMConvertSeeds \
                {self.model_dir} \
                {markers_folder} \
                {self.n_kernels} {self.kernel_size} \
                {self.attention} {self.aggregation_opt}")

    def train_gflim(self):
        
        # create layer models and extract activations
        for layer in range(1, self.nlayers + 1):

            if self.kernel_neighbors == "color" or self.stride_neighbors == "color":
                run_script(f"{IFT_dir}/bin/iftGFLIMPrecCreateLayerModel \
                    {self.model_dir} \
                    {self.arch_file} {layer} \
                    {self.kernel_neighbors} {self.stride_neighbors} \
                    {self.graph_feat} {self.aggregation_opt} {self.dataset_path}")
            else:
                run_script(f"{IFT_dir}/bin/iftGFLIMCreateLayerModel \
                    {self.model_dir} \
                    {self.arch_file} {layer} \
                    graph-based {self.attention} {self.aggregation_opt}")

            run_script(f"{IFT_dir}/bin/iftGFLIMMergeLayerModels \
                {self.model_dir} {layer}")

            if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color": 
                run_script(f"{IFT_dir}/bin/iftGFLIMEncodeMergedLayerBatchV3 \
                    {self.model_dir} {self.arch_file} {layer} \
                    {self.batch_size} {self.attention} {self.kernel_neighbors} \
                    {self.pool_neighbors} {self.stride_neighbors} \
                    {self.graph_feat} {self.aggregation_opt} {self.dataset_path}")
            else:
                run_script(f"{IFT_dir}/bin/iftGFLIMEncodeMergedLayerBatch \
                    {self.model_dir} {self.arch_file} {layer} {self.batch_size} graph-based \
                    {self.attention} {self.aggregation_opt}")
                
            #run_script(f"{IFT_dir}/bin/iftGFLIMWriteActivations \
            #    {self.model_dir} {layer} {self.batch_size}")

    def extract_features(self, extract_path):
        # extracts features from the train-val set
        run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
            {extract_path} \
            {self.model_dir} 1 {self.spx} {self.initial_seeds} {self.fixed_spx} {self.graph_feat} \
            {self.arch_file} {self.nlayers} {self.batch_size} {self.attention} {self.aggregation_opt} \
            1 -1 -1 graph-adaptive \
            {self.kernel_neighbors} {self.pool_neighbors} {self.stride_neighbors}")

    def clean_extracted_features(self, layer):
        # clean up activations and layer models to save space
        shutil.rmtree(f"{self.model_dir}/layer{layer}", ignore_errors=True)
        shutil.rmtree(f"{self.model_dir}/labels{layer}", ignore_errors=True)

    def graphs_to_vectors(self, graphs_folder):
        graph_files = [
            f for f in os.listdir(graphs_folder) if f.endswith(".json")
        ]
        vectors = []
        for graph_file in graph_files:
            with open(os.path.join(graphs_folder, graph_file), 'r') as f:
                graph_data = json.load(f)
            vector = vector_from_graph(graph_data)
            vectors.append(vector)
        return np.array(vectors)


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

def select_images_it0(labels_to_images, num_classes, num_images_per_class, random_state):
    # Select a specified number of images per class using the given random state
    selected_images = []
    for class_idx in range(1, num_classes + 1):
        if class_idx in labels_to_images:
            random.seed(random_state)
            print(f"Selecting {num_images_per_class} images for class {class_idx} from {len(labels_to_images[class_idx])} available images.")
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
    parser.add_argument("num_images_per_class", help="Number of images per class for GFLIM training.", type=int, default=1)

    # optional arguments
    parser.add_argument("--fixed_spx", help="Use fixed superpixels.", type=int, default=0)
    parser.add_argument("--random_state", help="Random state for reproducibility.", type=int, default=42)
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

    # set model directory
    model_dir=f"{IFT_dir}/Results/helminth-eggs/gflimBoFP_finetuneSVM/split{split}/{arch_name}/{initial_seeds}-{spx}spx_{kernel_neighbors}-kernel_{pool_neighbors}-pool_{stride_neighbors}-stride"
    if fixed_spx == 1: 
        model_dir=f"{model_dir}/fixedSPX_{graph_feat}_{aggregation_opt}_{attention}"
    else:
        model_dir=f"{model_dir}/notFixedSPX_{graph_feat}_{aggregation_opt}_{attention}"

    # -------------------------------------------------------
    model_dir_GFLIM = f"{model_dir}/GFLIM"
    os.makedirs(model_dir_GFLIM, exist_ok=True)

    # read data descriptor file for 5% training data from which the GFLIM training data will be extracted
    data_descriptor_file = f"{args.inc_split_file}/data_descriptor_perc5.json"
    if os.path.isfile(data_descriptor_file) == False:
        raise FileNotFoundError(f"Data descriptor file {data_descriptor_file} does not exist. Exiting.")
    with open(data_descriptor_file, 'r') as f:
        data_descriptor = json.load(f)

    # create csv files for the train set
    train_csv = f"{model_dir_GFLIM}/train{split}.csv"
    with open(train_csv, 'w') as f_train:
        for item in data_descriptor['training']:
            f_train.write(f"{dataset_path}/images/{item}\n")
    image_names = read_image_names(train_csv)
    labels_to_images = group_images_by_label(image_names)
    selected_images = select_images_it0(labels_to_images, num_classes, num_images_per_class, random_state)

    # create GFLIM train csv file
    train_csv_GFLIM = f"{model_dir_GFLIM}/train{split}-GFLIM.csv"
    copy_selected_images(selected_images, train_csv_GFLIM)
    with open(train_csv_GFLIM, 'w') as f_train:
        for item in selected_images:
            f_train.write(f"{item}\n")

    # train a GFLIM model using one image per class
    model = Model(model_dir_GFLIM, arch_file,
                  dataset_path, num_classes,
                  spx, initial_seeds, kernel_size, kernel_neighbors,
                  pool_size, pool_neighbors, stride_size, stride_neighbors,
                  n_kernels, graph_feat, aggregation_opt, attention,
                  fixed_spx, nlayers, batch_size)
    
    # -------------------------------------------------------

    with tqdm(range(21), desc=f"Preparing data", unit="step", leave=False) as pbar:
    
        # -------------------------------------------------------
        pre_descr = f"GFLIM Training"

        # create graphs and labels
        pbar.set_description_str(f"{pre_descr} - Create graphs and spx labels")
        pbar.update(1)
        model.create_graphs_and_labels(train_csv_GFLIM)
        
        # create seeds
        pbar.set_description_str(f"{pre_descr} - Create seeds")
        pbar.update(1)
        model.create_seeds(train_csv_GFLIM)

        # train GFLIM model
        pbar.set_description_str(f"{pre_descr} - Train the model")
        pbar.update(1)
        model.train_gflim()

        model_dir_fineTuningSVM = f"{model_dir}/fineTunningSVM"
        os.makedirs(model_dir_fineTuningSVM, exist_ok=True)
        # -------------------------------------------------------

        # -------------------------------------------------------
        # read data descriptor file for 100% training data from which the SVM will be trained
        data_descriptor_file = f"{args.inc_split_file}/data_descriptor_perc100.json"
        if os.path.isfile(data_descriptor_file) == False:
            raise FileNotFoundError(f"Data descriptor file {data_descriptor_file} does not exist. Exiting.")
        with open(data_descriptor_file, 'r') as f:
            data_descriptor = json.load(f)

        # create csv files for the train set
        y_train = []
        train_csv = f"{model_dir_fineTuningSVM}/train{split}.csv"
        with open(train_csv, 'w') as f_train:
            for item in data_descriptor['training']:
                f_train.write(f"{dataset_path}/images/{item}\n")
                y_train.append(int(item.split('_')[0])-1)
        
        pre_descr = f"SVM Fine-tuning"

        #### Extract features from the training set and save them
        vectors_file=f"{model_dir_fineTuningSVM}/layer{nlayers}_train_vectors.npz"
        if os.path.isfile(vectors_file) == False:
            pbar.set_description_str(f"{pre_descr} - Extract features from the training set")
            pbar.update(1)
            model.extract_features(train_csv)

            pbar.set_description_str(f"{pre_descr} - Prepare activations for SVM training")
            pbar.update(1)
            x_train = model.graphs_to_vectors(f"{model_dir_GFLIM}/layer{nlayers}")
            np.savez_compressed(vectors_file, vectors=x_train)
            model.clean_extracted_features(0)
            for layer in range(1, nlayers + 1):
                model.clean_extracted_features(layer)
        else:
            pbar.update(2)
            x_train = np.load(vectors_file)
            x_train = x_train['vectors']
        # -------------------------------------------------------

        # -------------------------------------------------------
        #### Find best parameters SVM with linear kernel

        param_grid_svm = {
            'linear': {
                'svc__C': [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1],                   
                'svc__kernel': ['linear'],#, 'rbf', 'poly', 'sigmoid'],
                'svc__class_weight': ['balanced'],
                'svc__decision_function_shape': ['ovo']
            },
            'rbf': {
                'svc__C': [1e-1, 5e-1, 1e0, 5e0, 1e1, 10, 50, 100, 500, 1000],                   
                'svc__gamma': [500, 100, 50, 10, 5, 1, 0.5, 0.1, 'auto', 'scale'],                
                'svc__kernel': ['rbf'],#, 'poly', 'sigmoid'],
                'svc__class_weight': ['balanced'],
                'svc__decision_function_shape': ['ovo']
            }
        }

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(random_state=42, probability=True))
        ])

        nsplits = min(min([len(labels_to_images[class_idx]) for class_idx in range(1, num_classes + 1)]), 5)
        skf = StratifiedKFold(n_splits=nsplits)

        best_params = {}
        for kernel in ['linear']:#, 'rbf']:
            best_params_file = f"{model_dir_fineTuningSVM}/svm_{kernel}_layer{nlayers}_best_params.txt"
            print(best_params_file)
            if os.path.isfile(best_params_file) == False:
                pbar.set_description_str(f"{pre_descr} - Fine-tune SVM hyperparameters")
                pbar.update(1)

                """
                param_random_svm = {
                    'svc__C': stats.expon(scale=100),                   
                    'svc__gamma': stats.expon(scale=.1),                
                    'svc__kernel': ['linear'],#, 'rbf', 'poly', 'sigmoid'],
                    'svc__class_weight': ['balanced'],
                    'svc__decision_function_shape': ['ovo']
                }
                """
                
                svmModel_grid = HalvingGridSearchCV(estimator=pipe, param_grid=param_grid_svm[kernel], 
                                                    verbose=1, cv=skf, n_jobs=-1, 
                                                    factor=2, random_state=42)
                svmModel_grid.fit(x_train, y_train)

                # save the results of the HalvingGridSearchCV in a csv file
                df = pd.DataFrame(svmModel_grid.cv_results_)
                df.to_csv(f"{model_dir_fineTuningSVM}/svm_{kernel}_layer{nlayers}_cv_results.csv", index=False)

                pbar.set_description_str(f"{pre_descr} - Saving model and parameters")
                pbar.update(1)
                
                # save the best model in a file
                joblib.dump(svmModel_grid.best_estimator_, f"{model_dir_fineTuningSVM}/svm_{kernel}_layer{nlayers}_best_model.pkl")

                # save best parameters in a text file
                with open(f"{model_dir_fineTuningSVM}/svm_{kernel}_layer{nlayers}_best_params.txt", "w") as f:
                    f.write(str(svmModel_grid.best_params_))
                
                best_params[kernel] = svmModel_grid.best_params_
            
            else:
                with open(best_params_file, "r") as f:
                    best_params[kernel] = eval(f.read())
                print(f"Best parameters for kernel {kernel} loaded from file: {best_params[kernel]}")
                pbar.update(2)
        
        
        best_params[kernel] = {
            'svc__C': 1e2,
            'svc__kernel': 'linear',
            'svc__decision_function_shape': 'ovo',
            'svc__gamma': 0
        }
        # -------------------------------------------------------

        for data_descriptor_file in split_files:

            perc = data_descriptor_file.split('_')[-1].replace('perc','').split('.')[0]
            pre_descr = f"{perc}% data"
            if perc != '5':
                continue

            model_dir_perc = f"{model_dir}/perc{perc}"
            os.makedirs(model_dir_perc, exist_ok=True)

            data_descriptor_path = os.path.join(args.inc_split_file, data_descriptor_file)
            with open(data_descriptor_path, 'r') as f:
                data_descriptor = json.load(f)

            # create csv files for train, validation and test
            y_train, y_val, y_test = [], [], []
            train_csv = f"{model_dir_perc}/train{split}.csv"
            val_csv = f"{model_dir_perc}/val{split}.csv"
            test_csv = f"{model_dir_perc}/test{split}.csv"
            with open(train_csv, 'w') as f_train:
                for item in data_descriptor['training']:
                    f_train.write(f"{dataset_path}/images/{item}\n")
                    y_train.append(int(item.split('_')[0])-1)
            with open(val_csv, 'w') as f_val:
                for item in data_descriptor['validation']:
                    f_val.write(f"{dataset_path}/images/{item}\n")
                    y_val.append(int(item.split('_')[0])-1)
            with open(test_csv, 'w') as f_test:
                for item in data_descriptor['test']:
                    f_test.write(f"{dataset_path}/images/{item}\n")
                    y_test.append(int(item.split('_')[0])-1)
            image_names = read_image_names(train_csv)
            labels_to_images = group_images_by_label(image_names)
            selected_images = select_images_it0(labels_to_images, num_classes, num_images_per_class, random_state)

            train_vectors_file=f"{model_dir_perc}/layer{nlayers}_train_vectors.npz"
            if os.path.isfile(train_vectors_file) == False:
                pbar.set_description_str(f"{pre_descr} - Extract features from the training set")
                pbar.update(1)
                model.extract_features(train_csv)

                pbar.set_description_str(f"{pre_descr} - Prepare activations for SVM training")
                pbar.update(1)
                x_train = model.graphs_to_vectors(f"{model_dir_GFLIM}/layer{nlayers}")
                np.savez_compressed(f"{model_dir_perc}/layer{nlayers}_train_vectors.npz", vectors=x_train)
                model.clean_extracted_features(0)
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
            else:
                pbar.update(2)
                x_train = np.load(train_vectors_file)
                x_train = x_train['vectors']

            val_vectors_file=f"{model_dir_perc}/layer{nlayers}_val_vectors.npz"
            if os.path.isfile(val_vectors_file) == False:
                pbar.set_description_str(f"{pre_descr} - Extract features from the validation set")
                pbar.update(1)
                model.extract_features(val_csv)

                pbar.set_description_str(f"{pre_descr} - Prepare activations for SVM classification")
                pbar.update(1)
                x_val = model.graphs_to_vectors(f"{model_dir_GFLIM}/layer{nlayers}")
                np.savez_compressed(f"{model_dir_perc}/layer{nlayers}_val_vectors.npz", vectors=x_val)
                model.clean_extracted_features(0)
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
            else:
                pbar.update(2)
                x_val = np.load(val_vectors_file)
                x_val = x_val['vectors']

            #-------------------------------------------------------
            for kernel in ['linear']:#, 'rbf']:
                pbar.set_description_str(f"{pre_descr} - Train and eval SVM with best hyperparameters with kernel {kernel}")
                pbar.update(1)

                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svc', SVC(random_state=42, probability=True,
                                C=best_params[kernel]['svc__C'],
                                gamma=best_params[kernel]['svc__gamma'],
                                kernel=best_params[kernel]['svc__kernel'],
                                #class_weight=best_params[kernel]['svc__class_weight'],
                                decision_function_shape=best_params[kernel]['svc__decision_function_shape'],
                                tol=1e-5))
                ])
                print("train set shape: ", x_train.shape)
                pipe.fit(x_train, y_train)

                pbar.set_description_str(f"{pre_descr} - Saving model and parameters")
                pbar.update(1)
                # save the SVM model in a file
                joblib.dump(pipe, f"{model_dir_perc}/svm_{kernel}_layer{nlayers}_model.pkl")
                
                pbar.set_description_str(f"{pre_descr} - Saving predictions on train and val sets")
                pbar.update(1)
                # save prediction and probability on train and val sets
                y_pred_train = pipe.predict(x_train)
                print(f'Classes in y_pred_train: {np.unique(y_pred_train)}')
                y_prob_train = pipe.predict_proba(x_train)
                with open(f"{model_dir_perc}/svm_{kernel}_layer{nlayers}_train_predictions.txt", "w") as f:
                    for item, true_label, pred, prob in zip(data_descriptor['training'], y_train, y_pred_train, y_prob_train):
                        f.write(f"{dataset_path}/images/{item};{true_label};{pred};{(';').join(map(str, prob))}\n")
                y_pred_val = pipe.predict(x_val)
                y_prob_val = pipe.predict_proba(x_val)
                with open(f"{model_dir_perc}/svm_{kernel}_layer{nlayers}_val_predictions.txt", "w") as f:
                    for item, true_label, pred, prob in zip(data_descriptor['validation'], y_val, y_pred_val, y_prob_val):
                        f.write(f"{dataset_path}/images/{item};{true_label};{pred};{(';').join(map(str, prob))}\n")

                pbar.set_description_str(f"{pre_descr} - Saving classification reports")
                pbar.update(1)
                # save classification report and confusion matrix in a file along with kappa            
                report = classification_report(y_val, y_pred_val, zero_division=0)
                cm = confusion_matrix(y_val, y_pred_val)
                with open(f"{model_dir_perc}/svm_{kernel}_layer{nlayers}_val_classification_report.txt", "w") as f:
                    f.write("Classification Report:\n")
                    f.write(str(report))
                    f.write("\nConfusion Matrix:\n")
                    f.write(np.array2string(cm))
                    f.write("\nKappa:\n")
                    f.write(str(cohen_kappa_score(y_val, y_pred_val)))

                # save classification report and confusion matrix in a file along with kappa for train set            
                report = classification_report(y_train, y_pred_train, zero_division=0)
                cm = confusion_matrix(y_train, y_pred_train)
                with open(f"{model_dir_perc}/svm_{kernel}_layer{nlayers}_train_classification_report.txt", "w") as f:
                    f.write("Classification Report:\n")
                    f.write(str(report))
                    f.write("\nConfusion Matrix:\n")
                    f.write(np.array2string(cm))
                    f.write("\nKappa:\n")
                    f.write(str(cohen_kappa_score(y_train, y_pred_train)))
            
            # -------------------------------------------------------