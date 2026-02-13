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
from sklearn.metrics import accuracy_score, recall_score

tmp_file = "/tmp/run_incremental_splits_increasingTrainSet_log.txt"
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
            {markers_folder} \
            --mask_dir {self.dataset_path}/masks")

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

    def train_svm(self, layer, train_path, imgSetName, C):
        activations_dir=f"{self.model_dir}/layer{layer}_{imgSetName}"
        self.svm_file=f"{self.model_dir}/svm_{layer}_{imgSetName}.zip"
        
        if os.path.exists(f"{self.model_dir}/layer{layer}"):
            shutil.rmtree(activations_dir, ignore_errors=True)
            run_script(f"cp -r {self.model_dir}/layer{layer} {activations_dir}")

        if os.path.exists(activations_dir) == False:
            raise ValueError(f"Activations directory {activations_dir} does not exist. Cannot train SVM.")

        run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
            {activations_dir} \
            {train_path} 1 1")

        run_script(f"{IFT_dir}/bin/iftSupTrainBySVM \
            {activations_dir}.zip 0 0 {C} 0 {self.svm_file}")
        
    def extract_features(self, extract_path):
        # extracts features from the train-val set
        run_script(f"{IFT_dir}/bin/iftGFLIMExtractV3 \
            {extract_path} {self.model_dir} {self.arch_file} \
            {self.nlayers} {self.batch_size} \
            {self.attention} {self.aggregation_opt} \
            {self.graph_feat} DISF {self.spx} {self.fixed_spx} \
            1 -1 -1 graph-adaptive \
            {self.kernel_neighbors} {self.pool_neighbors} {self.stride_neighbors} \
            {self.initial_seeds} 1")

    def clean_extracted_features(self, layer):
        # clean up activations and layer models to save space
        shutil.rmtree(f"{self.model_dir}/layer{layer}", ignore_errors=True)
        shutil.rmtree(f"{self.model_dir}/labels{layer}", ignore_errors=True)

    def clean_svm_data(self, layer, imgSetName):
        # clean up activations and layer models to save space
        run_script(f"rm -r {self.model_dir}/layer{layer}_{imgSetName}")
        run_script(f"rm -f {self.model_dir}/layer{layer}_{imgSetName}.zip")

    def classify_SVM(self, layer, images_path, imgSetName):
        if self.svm_file is None:
            raise ValueError("SVM file is not set. Train the SVM before classifying.")

        activations_dir=f"{self.model_dir}/layer{layer}_{imgSetName}"
        if os.path.exists(f"{self.model_dir}/layer{layer}"):
            shutil.rmtree(activations_dir, ignore_errors=True)
        run_script(f"cp -r {self.model_dir}/layer{layer} {activations_dir}")

        if os.path.exists(activations_dir) == False:
            raise ValueError(f"Activations directory {activations_dir} does not exist. Cannot classify with SVM.")

        run_script(f"{IFT_dir}/bin/iftGFLIMActivDataSet \
            {activations_dir} \
            {images_path} 1 1")

        classification_output = run_script(f"{IFT_dir}/bin/iftClassifyBySVM \
            {activations_dir}.zip \
            {self.svm_file} \
            {self.model_dir}/results_layer{layer}_{imgSetName}")
                
        with open(f"{self.model_dir}/classification_layer{layer}_{imgSetName}.txt", "w") as f:
            f.write(classification_output)

    def get_new_train_images(self, layer, imgSetName):
        """
        Select one image per class among the missclassified images in the training set, 
        by choosing the image closest to the centroid of the graph feature vectors of the missclassified images of that class.
        """
        with open(f"{self.model_dir}/results_layer{layer}_{imgSetName}/layer{layer}_{imgSetName}-classified-images.csv", "r") as f:
            df = pd.read_csv(f, header=None, sep=";")
        # assign a header to the dataframe
        df.columns = ["image_path", "true_label", "predicted_label"]
        # maintain only missclassified images
        missclassified_df = df[df["true_label"] != df["predicted_label"]]
        
        new_train_images = []
        for c in range(1, self.num_classes + 1):
            # filter missclassified images that belong to class c
            class_missclassified = missclassified_df[missclassified_df["true_label"] == c]
            if class_missclassified.empty:
                continue

            # from f"{self.model_dir}/layer{layer}_{imgSetName}" read the graphs of the selected images
            graphs_vectors, images_names = [], []
            for img_path in class_missclassified["image_path"].tolist():
                images_names.append(img_path)
                graph_file = f"{self.model_dir}/layer{layer}_{imgSetName}/{os.path.basename(img_path).split('.')[0]}.json"
                with open(graph_file, "r") as gf:
                    graph_data = json.load(gf)
                    graphs_vectors.append(vector_from_graph(graph_data))

            # compute the centroid of the vectors
            centroid = np.mean(np.array(graphs_vectors), axis=0)

            # include in new_train_images the image closest to the centroid
            distances = [np.linalg.norm(np.array(vec) - centroid) for vec in graphs_vectors]
            closest_index = distances.index(min(distances))
            new_train_images.append(images_names[closest_index])

        return new_train_images
 
    def get_new_train_images_class(self, layer, imgSetNameTrain, imgSetNameVal):
        """
        Select one image among the missclassified images in the training set that belong to the class with worse accuracy in the validation set, 
        by choosing the image closest to the centroid of the graph feature vectors of the missclassified images of that class.
        """

        with open(f"{self.model_dir}/results_layer{layer}_{imgSetNameVal}/layer{layer}_{imgSetNameVal}-classified-images.csv", "r") as f:
            df = pd.read_csv(f, header=None, sep=";")
        # assign a header to the dataframe
        df.columns = ["image_path", "true_label", "predicted_label"]

        # select the class_id with worse accuracy
        min_acc = 1.0
        class_id = -1
        for c in range(1, self.num_classes + 1):
            df_c = df[df["true_label"] == c]
            accuracy = recall_score(df_c["true_label"], df_c["predicted_label"])
            if accuracy < min_acc:
                min_acc = accuracy
                class_id = c

        with open(f"{self.model_dir}/results_layer{layer}_{imgSetNameTrain}/layer{layer}_{imgSetNameTrain}-classified-images.csv", "r") as f:
            df = pd.read_csv(f, header=None, sep=";")
        # assign a header to the dataframe
        df.columns = ["image_path", "true_label", "predicted_label"]
    
        # maintain only missclassified images that belong to class_id
        class_missclassified = df[(df["true_label"] != df["predicted_label"]) & (df["true_label"] == class_id)]
        if class_missclassified.empty:
            return []
        
        # from f"{self.model_dir}/layer{layer}_{imgSetName}" read the graphs of the selected images
        graphs_vectors, images_names = [], []
        for img_path in class_missclassified["image_path"].tolist():
            images_names.append(img_path)
            graph_file = f"{self.model_dir}/layer{layer}_{imgSetNameTrain}/{os.path.basename(img_path).split('.')[0]}.json"
            with open(graph_file, "r") as gf:
                graph_data = json.load(gf)
                graphs_vectors.append(vector_from_graph(graph_data))

        # compute the centroid of the vectors
        centroid = np.mean(np.array(graphs_vectors), axis=0)

        # include in new_train_images the image closest to the centroid
        distances = [np.linalg.norm(np.array(vec) - centroid) for vec in graphs_vectors]
        closest_index = distances.index(min(distances))

        return [images_names[closest_index]]
 
    def get_new_train_images_classes(self, layer, imgSetNameTrain, imgSetNameVal):
        """
        Select one image per class among the missclassified images in the training set that belong to the classes with worse accuracy in the validation set, 
        by choosing the image closest to the centroid of the graph feature vectors of the missclassified images of those classes.
        """

        # read classification results from validation and training sets
        with open(f"{self.model_dir}/results_layer{layer}_{imgSetNameVal}/layer{layer}_{imgSetNameVal}-classified-images.csv", "r") as f:
            classification_val = pd.read_csv(f, header=None, sep=";")
        classification_val.columns = ["image_path", "true_label", "predicted_label"]
        
        with open(f"{self.model_dir}/results_layer{layer}_{imgSetNameTrain}/layer{layer}_{imgSetNameTrain}-classified-images.csv", "r") as f:
            classification_train = pd.read_csv(f, header=None, sep=";")
        classification_train.columns = ["image_path", "true_label", "predicted_label"]
        missclassified_train = classification_train[classification_train["true_label"] != classification_train["predicted_label"]]

        # select the class_ids with worse accuracy
        min_acc = 1.0
        class_ids = []
        for c in range(1, self.num_classes + 1):
            # do not consider classes with no missclassified images
            if missclassified_train[missclassified_train["true_label"] == c].empty:
                continue
            df_c = classification_val[classification_val["true_label"] == c]
            if df_c.empty:
                continue
            accuracy = recall_score(df_c["true_label"], df_c["predicted_label"])
            if accuracy < min_acc:
                min_acc = accuracy
                class_ids = [c]
            elif accuracy == min_acc:
                class_ids.append(c)

        print(f"Classes with lowest accuracy ({min_acc}): {class_ids}")
    
        # maintain only missclassified images that belong to class_ids
        missclassified_train = missclassified_train[missclassified_train["true_label"].isin(class_ids)]
        if missclassified_train.empty:
            print("No remaining missclassified images found.")
            return []
        
        new_train_images = []
        for class_id in class_ids:
            class_missclassified = missclassified_train[missclassified_train["true_label"] == class_id]
            if class_missclassified.empty:
                continue
            # from f"{self.model_dir}/layer{layer}_{imgSetName}" read the graphs of the selected images
            graphs_vectors, images_names = [], []
            for img_path in class_missclassified["image_path"].tolist():
                images_names.append(img_path)
                graph_file = f"{self.model_dir}/layer{layer}_{imgSetNameTrain}/{os.path.basename(img_path).split('.')[0]}.json"
                with open(graph_file, "r") as gf:
                    graph_data = json.load(gf)
                    graphs_vectors.append(vector_from_graph(graph_data))

            # compute the centroid of the vectors
            centroid = np.mean(np.array(graphs_vectors), axis=0)

            # include in new_train_images the image closest to the centroid
            distances = [np.linalg.norm(np.array(vec) - centroid) for vec in graphs_vectors]
            closest_index = distances.index(min(distances))
            new_train_images.append(images_names[closest_index])

        return new_train_images
 

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

#def strToList(s):
#    return s.split(',')


def prepare_new_experiment(model_dir, dataset_path, 
                           num_classes, random_state,
                           train_csv, val_csv, test_csv):
    
    os.makedirs(model_dir, exist_ok=True)

    # create csv files for train, validation and test
    with open(train_csv, 'w') as f_train:
        for item in data_descriptor['training']:
            f_train.write(f"{dataset_path}/images/{item}\n")
    with open(val_csv, 'w') as f_val:
        for item in data_descriptor['validation']:
            f_val.write(f"{dataset_path}/images/{item}\n")
    with open(test_csv, 'w') as f_test:
        for item in data_descriptor['test']:
            f_test.write(f"{dataset_path}/images/{item}\n")

    image_names = read_image_names(train_csv)
    labels_to_images = group_images_by_label(image_names)
    return select_images_it0(labels_to_images, num_classes, 1, random_state)

def load_experiment(initial_it, model, split, val_csv):
    if(os.path.exists(model.model_dir) == False):
        raise ValueError(f"Model directory {model.model_dir} does not exist. \
                         Cannot continue from iteration {initial_it}.")

    filename = f"{model.model_dir}/train{split}-GFLIM.csv"
    selected_images = read_image_names(filename)
    
    # Extract features from the validation set
    model.extract_features(val_csv)

    # Select new images to add to the GFLIM training set
    new_train_images = model.get_new_train_images_classes(model.nlayers, f"trainIt", f"val")
    selected_images.extend(new_train_images)

    # clean up activations and layer models to save space
    model.clean_extracted_features(0)
    model.clean_svm_data(model.nlayers, 'val')
    model.clean_svm_data(model.nlayers, "trainIt")
    for layer in range(1, model.nlayers + 1):
        model.clean_extracted_features(layer)

    return selected_images


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
    parser.add_argument("num_it", help="Number of iterations for GFLIM training.", type=int, default=1)

    # optional arguments
    parser.add_argument("--fixed_spx", help="Use fixed superpixels.", type=int, default=0)
    parser.add_argument("--random_state", help="Random state for reproducibility.", type=int, default=42)
    parser.add_argument("--nlayers", help="Number of layers in the GFLIM model.", type=int, default=3)
    parser.add_argument("--batch_size", help="Batch size for training.", type=int, default=64)
    parser.add_argument("--C", help="SVM regularization parameter.", type=str, default="1e2")
    parser.add_argument("--initial_it", help="Initial iteration to start from.", type=int, default=0)

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
    num_it = args.num_it
    nlayers = args.nlayers
    batch_size = args.batch_size
    C_val = args.C
    initial_it = args.initial_it
    
    split_files = [
        f for f in os.listdir(args.inc_split_file) if f.endswith(".json")
    ]

    for data_descriptor_file in split_files:

        perc = data_descriptor_file.split('_')[-1].replace('perc','').split('.')[0]
        if perc != "75":
            continue

        pre_descr = f"Split {split} with {perc}% data"

        with tqdm(range(10*(num_it-initial_it)), desc=f"{pre_descr} - Preparing data", unit="step", leave=False) as pbar:
        
            data_descriptor_path = os.path.join(args.inc_split_file, data_descriptor_file)
            with open(data_descriptor_path, 'r') as f:
                data_descriptor = json.load(f)

            # set model directory
            model_dir=f"{IFT_dir}/Results/helminth-eggs/gflimBoFP_incremental_splits_incTrainSet_{C_val}_rec/split{split}/perc{perc}/{arch_name}/{initial_seeds}-{spx}spx_{kernel_neighbors}-kernel_{pool_neighbors}-pool_{stride_neighbors}-stride"
            if fixed_spx == 1: 
                model_dir=f"{model_dir}/fixedSPX_{graph_feat}_{aggregation_opt}_{attention}"
            else:
                model_dir=f"{model_dir}/notFixedSPX_{graph_feat}_{aggregation_opt}_{attention}"

            model_exists = False
            selected_images = []
            
            train_csv = f"{model_dir}/train{split}.csv"
            val_csv = f"{model_dir}/val{split}.csv"
            test_csv = f"{model_dir}/test{split}.csv"

            if(initial_it > 0):
                model_dir_it = f"{model_dir}/{initial_it * num_classes}_images"
                model = Model(model_dir_it, arch_file,
                              dataset_path, num_classes,
                              spx, initial_seeds, kernel_size, kernel_neighbors,
                              pool_size, pool_neighbors, stride_size, stride_neighbors,
                              n_kernels, graph_feat, aggregation_opt, attention,
                              fixed_spx, nlayers, batch_size)
                selected_images = load_experiment(initial_it, model, split, val_csv)
                del model
            else:
                selected_images = prepare_new_experiment(model_dir, dataset_path, 
                                                         num_classes, random_state,
                                                         train_csv, val_csv, test_csv)

            for it in range(initial_it, num_it+1):

                pre_descr = f"Split {split} with {perc}% data - Iteration {it} - {len(selected_images)} images"

                num_images = len(selected_images)
                model_dir_it = f"{model_dir}/{num_images}_images"
                os.makedirs(model_dir_it, exist_ok=True)
                
                # create GFLIM train csv file
                train_csv_GFLIM = f"{model_dir_it}/train{split}-GFLIM.csv"
                copy_selected_images(selected_images, train_csv_GFLIM)

                # write a csv file with train images that will not be used to train GFLIM 
                train_it_csv = f"{model_dir_it}/train{split}.csv"
                with open(train_it_csv, 'w') as f_train:
                    for item in data_descriptor['training']:
                        if item not in selected_images:
                            f_train.write(f"{dataset_path}/images/{item}\n")

                model = Model(model_dir_it, arch_file,
                              dataset_path, num_classes,
                              spx, initial_seeds, kernel_size, kernel_neighbors,
                              pool_size, pool_neighbors, stride_size, stride_neighbors,
                              n_kernels, graph_feat, aggregation_opt, attention,
                              fixed_spx, nlayers, batch_size)

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
                
                #-------------------------------------------------------
                # train SVMs for each layer using the GFLIM training set
                pbar.set_description_str(f"{pre_descr} - Train SVM with GFLIM training set")
                pbar.update(1)
                model.train_svm(nlayers, train_csv_GFLIM, f"trainGFLIM", C_val)

                # clean up activations and layer models to save space
                model.clean_extracted_features(0)
                model.clean_svm_data(nlayers, "trainGFLIM")
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
                
                # extract features from the train set
                pbar.set_description_str(f"{pre_descr} - Extract features from the train set")
                pbar.update(1)
                model.extract_features(train_it_csv)
                
                # classify train set with trained SVMs
                pbar.set_description_str(f"{pre_descr} - Classify the train set")
                pbar.update(1)
                model.classify_SVM(nlayers, train_it_csv, f"trainIt")

                # clean up activations and layer models to save space
                # do not delete the svm files for "trainIt" so they can be used to choose new images for training in GFLIM
                run_script(f"rm -f svm_{nlayers}_trainGFLIM.zip")
                model.clean_extracted_features(0)
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
                #-------------------------------------------------------

                #-------------------------------------------------------
                # extract features from the train set
                pbar.set_description_str(f"{pre_descr} - Extract features from the complete training set")
                pbar.update(1)
                model.extract_features(train_csv)

                # train SVMs for each layer using the GFLwhole training set
                pbar.set_description_str(f"{pre_descr} - Train SVM with the complete training set")
                pbar.update(1)
                model.train_svm(nlayers, train_csv, f"train", C_val)

                # clean up activations and layer models to save space
                model.clean_extracted_features(0)
                model.clean_svm_data(nlayers, "train")
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
                
                # extract features from the validation set
                pbar.set_description_str(f"{pre_descr} - Extract features from the validation set")
                pbar.update(1)
                model.extract_features(val_csv)

                # classify validation set with trained SVMs
                pbar.set_description_str(f"{pre_descr} - Classify the validation set")
                pbar.update(1)
                model.classify_SVM(nlayers, val_csv, f"val")

                # do not delete the svm files for "val" so they can be used to choose new images for training in GFLIM

                #-------------------------------------------------------

                #-------------------------------------------------------
                # filter the missclassified images from the training set
                pbar.set_description_str(f"{pre_descr} - Select new images to add to the GFLIM training set")
                pbar.update(1)
                new_train_images = model.get_new_train_images_classes(nlayers, f"trainIt", f"val")
                selected_images.extend(new_train_images)

                # clean up activations and layer models to save space
                model.clean_extracted_features(0)
                model.clean_svm_data(nlayers, 'val')
                model.clean_svm_data(nlayers, "trainIt")
                for layer in range(1, nlayers + 1):
                    model.clean_extracted_features(layer)
                
                #-------------------------------------------------------