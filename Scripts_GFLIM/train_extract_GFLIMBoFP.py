"""
This program implements functions to train and extract features using the GFLIM-BoFP and multi-scale GFLIM-BoFP.

"""

import os
import json
import sys
import shutil
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import copy
import argparse

# source ~/.local/share/virtualenvs/SH-NTEVX1e9/bin/activate

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def compute_kernels_maps_bkp(model_dir, rows_to_keep = None):
        """
        Read a file at model_dir/model/kernels_map.txt containing a mapping from the kernels of scale 0 to the kernels at others scales. 

        Args:
            model_dir (str): Directory where the model is stored.
            rows_to_keep (list, optional): List of row indices to keep. If None, all rows are kept. Defaults to None.

        Note: rows_to_keep is used to filter the rows in the kernels map file. So, if 10 scales were computed, but only the scales x, y, and z are to be used, then rows_to_keep should be [x, y, z].
        
        Returns:
            list: A list of kernel maps, where each kernel map is a list of integers representing the mapping of kernels from scale 0 to scale i+1.
        """

        def get_kernel_linei(k, l, kernels_maps):
            # find which kernel is k in line (or scale) l
            i = 0
            while(i < l):
                k = kernels_maps[i][k]
                i+=1
            return k

        # contruct a kernels map from a scale 0 to i+1
        lines = []
        with open(os.path.join(model_dir, 'model', 'kernels_map.txt'), 'r') as f:
            lines = f.readlines()

        # para linha 0
        ki = 0
        kernels_map = []
        for i,k0 in enumerate(lines[0].strip().split(' ')):
            k0 = int(k0) - 1
            if(k0 == i):
                kernels_map.append(ki)
                ki += 1
            else:
                kernels_map.append(kernels_map[k0])

        kernels_maps = [kernels_map]

        # para linha i
        line_prev = lines[0].strip().split(' ')
        kernels_prev = kernels_map.copy()
        for l, line in enumerate(lines[1:]):
            line = line.strip().split(' ')
            kernels_map = []
            for i,k0 in enumerate(line):
                k0 = int(k0) - 1
                k0_prev = int(line_prev[k0]) - 1
                if(k0_prev == i):
                    k = get_kernel_linei(k0, l+1, kernels_maps)
                    if(k != k0_prev):
                        k = kernels_map[k]
                    kernels_map.append(k)
                    
            line_prev = line.copy()
            kernels_prev = kernels_map.copy()
            kernels_maps.append(kernels_map)
        return kernels_maps

def compute_kernels_maps(model_dir, rows_to_keep = None):
    def get_kernel_linei(k, l, kernels_maps):
        # find which kernel is k in line (or scale) l
        i = 0
        while(i < l):
            k = kernels_maps[i][k]
            i+=1
        return k
    
    # contruct a kernels map from a scale 0 to i+1
    lines = []
    with open(os.path.join(model_dir, 'model', 'kernels_map.txt'), 'r') as f:
        lines = f.readlines()

    if rows_to_keep is None:
        rows_to_keep = list(range(len(lines)))

    # process line 0
    ki = 0
    kernels_map = []
    for i,k0 in enumerate(lines[0].strip().split(' ')):
        k0 = int(k0) - 1
        if(k0 == i):
            kernels_map.append(ki) # new kernel
            ki += 1
        else:
            kernels_map.append(kernels_map[k0]) # repeated kernel

    kernels_maps = [kernels_map]

    # process each line i
    line_prev = lines[0].strip().split(' ')
    for l, line in enumerate(lines[1:]):
        line = line.strip().split(' ')
        kernels_map = []
        ki = 0
        for i,k0 in enumerate(line):
            k0 = int(k0) - 1 # i-th kernel in scale i indexed as scale 0
            # (e.g., i=5 k0=3 means kernels 3 and 5 in scale 0 are the same in scale i)

            k0_prev = int(line_prev[i]) - 1 # i-th kernel in scale i-1 indexed as scale 0
            # (e.g., i=5 k0_prev=4 means kernels 4 and 5 in scale 0 are the same in scale i-1. 
            # Considering the previous example, kernels 3, 4, and 5 in scale 0 are the same in scale i)

            if(k0_prev == i):
                if(k0 == k0_prev):
                    kernels_map.append(ki)
                    ki+=1
                else: # repeated (merged) kernel. Map from scale i-1 to scale i
                    k = get_kernel_linei(k0, l+1, kernels_maps) # map k0 from scale 0 to scale i-1
                    k = kernels_map[k]
                    kernels_map.append(k)

        line_prev = line.copy()
        kernels_maps.append(kernels_map)

    """
    # erase scales
    kernels_map_final = [kernels_maps[rows_to_keep[0]].copy()]
    for id,r in enumerate(rows_to_keep[:-1]):
        s = rows_to_keep[id+1]
        r+=1
        mapped_kernels = kernels_maps[r]
        r+=1
        while(r <= s):
            mapped_kernels = [kernels_maps[r][k] for k in mapped_kernels]
            r+=1
        kernels_map_final.append(mapped_kernels)
    """
    return kernels_maps    

def erase_scales_from_kernels_map(kernels_maps, rows_to_keep):
    """
    Erase the scales from the kernels map by keeping only the specified rows.
    """
    kernels_map_final = [kernels_maps[rows_to_keep[0]].copy()]
    for id,r in enumerate(rows_to_keep[:-1]):
        s = rows_to_keep[id+1]
        r+=1
        mapped_kernels = kernels_maps[r]
        r+=1
        while(r <= s):
            mapped_kernels = [kernels_maps[r][k] for k in mapped_kernels]
            r+=1
        kernels_map_final.append(mapped_kernels)

    return kernels_map_final

def get_merged_scales_from_kernels_map(kernels_maps, init, end):
    """
    Merge scales in range(init, end). 
    If init == end, return a range of values.
    If init == end+1, return kernels_maps[init]

    Example: kernels_map contains [(0->1), (1->2), (2->3), (3->4)]
    if init = 1, end = 2, return kernels_map[1]
    if init = 1, end = 3, 
            kernels_map_final = kernels_map[1]
            map from kernels_map_final to kernels_map[2]
            return kernels_map_final

    if  init = 2, end = 5,
        kernels_map_final = kernels_map[1]
        map from kernels_map_final to kernels_map[2]
        map from kernels_map_final to kernels_map[3]
        return kernels_map_final
    """

    if end < init:
        raise ValueError("end must be greater than init")

    if end > len(kernels_maps):
        raise ValueError("end must be less than or equal to the number of scales")

    if init == end:
        if init < len(kernels_maps):
            return [i for i in range(len(kernels_maps[init]))]
        else:
            raise ValueError("end must be less than the number of scales")

    kernels_map_final = kernels_maps[init].copy()
    for i in range(init+1, end):
        kernels_map_final = [kernels_maps[i][k] for k in kernels_map_final]
        
    return kernels_map_final

class MergeLayers():
    """
    This class is responsible for merging the layers of the multi-scale GFLIM-BoFP model.
    """
    def __init__(self, model_dir):
        """
        Initialize the MergeLayers class.

        Args:
            model_dir (str): Directory where the model is stored.
        """
        self.node_maps = []
        self.spx_imgs = []
        self.feature_matrices = []
        self.kernels_maps = []
        self.model_dir = model_dir

    def read_img(self, img_path, is_spx_img=False):
        """
        Read an image from a file.

        Args:
            img_path (str): Path to the image file.
            is_spx_img (bool, optional): If True, the image is treated as a grayscale 16bits superpixel image. Default is False.

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        img = Image.open(img_path)
        if is_spx_img:
            img = img.convert("I;16")
            img = np.array(img)
            replace = np.unique(img)
            for p in range(len(replace)):
                img[img == replace[p]] = p + 1
        else:
            img = np.array(img)
        return img

    def read_graph(self, graph_path):
        """
        Read a graph from a JSON file.
        """
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        return graph

    def normalize_graph(self, graph):
        """
        Normalize the features of the graph with a min-max normalization [0,1] over the features channels individually.
        Args:
            graph (dict): The graph to be normalized.

        Note: This function modifies the graph in place.
        """
        # Compute max value for normalization
        max_vals = graph['node0']['feats'].copy()
        for node in range(graph['num_nodes']):
            feats = graph[f'node{node}']['feats']
            for i,f in enumerate(feats):
                if f > max_vals[i]:
                    max_vals[i] = f
        # Normalize each feature to [0,1] with a max normalization
        for node in range(graph['num_nodes']):
            feats = graph[f'node{node}']['feats']
            for i,f in enumerate(feats):
                if max_vals[i] > 0:
                    feats[i] = f / max_vals[i]
                else:
                    feats[i] = 0.0
            graph[f'node{node}']['feats'] = feats

    def graph_to_featMatrix(self, graph):
        """
        Convert a graph to a feature matrix.
        Args:
            graph (dict): The graph to be converted.
        Returns:
            np.ndarray: The feature matrix of shape (num_nodes, num_feats).
        """
        feat_matrix = np.zeros((graph['num_nodes'], graph['num_feats']))
        for node in range(graph['num_nodes']):
            feat_matrix[node,:] = np.array(graph[f'node{node}']['feats'])
        return feat_matrix

    def add_scale(self, spx_path, graph_path):
        """
        Add a new scale to the model by loading the superpixel image and graph, and computing the node mapping.
        Args:
            spx_path (str): Path to the superpixel image file.
            graph_path (str): Path to the graph file.
        """
        # load superpixel image
        spx_img = self.read_img(spx_path, is_spx_img=True)
        self.spx_imgs.append(spx_img)

        # load feature matrix
        graph = self.read_graph(graph_path)
        self.normalize_graph(graph)
        feat_matrix = self.graph_to_featMatrix(graph)
        self.feature_matrices.append(feat_matrix)

    def upsample_featMatrix(self, feat_matrix_target, feat_matrix_increased, node_map, kernel_map):
        """
        Upsample the feat_matrix_target to the size of feat_matrix_increased using the node_map and kernel_map
        
        Args:
            feat_matrix_target (np.ndarray): The target feature matrix to be upsampled
            feat_matrix_increased (np.ndarray): The feature matrix with increased number of nodes and kernels
            node_map (list): The mapping of nodes from feat_matrix_increased to feat_matrix_target
            kernel_map (list): The mapping of kernels from feat_matrix_increased to feat_matrix_target
        Returns:
            np.ndarray: The upsampled feature matrix.
        """
        feat_matrix = np.zeros(feat_matrix_increased.shape)
        for node in range(feat_matrix_increased.shape[0]):
            feats_target = feat_matrix_target[node_map[node], :]
            feat_matrix[node, :] = np.array([feats_target[kernel_map[k]] for k in range(feat_matrix_increased.shape[1])])
            #for k in range(feat_matrix_increased.shape[1]):
            #    feat_matrix[node, k] = feats_target[kernel_map[k]]
            #feat_matrix[node, :] = np.array(feat_matrix[node, :])
        return feat_matrix

    def compute_otsu_threshold(self, feat_matrix):
        """
        Compute the Otsu threshold for each feature channel in the feature matrix.
        Args:
            feat_matrix (np.ndarray): The feature matrix of shape (num_nodes, num_feats).
        Returns:
            list: A list of Otsu thresholds for each feature channel.
        """
        # transform the cols in feat_matrix into vectors
        feats = [feat_matrix[:, i] for i in range(feat_matrix.shape[1])]
        threshs = [threshold_otsu(f) for f in feats]
        return threshs

    def compute_mean_std(self):
        """
        Compute the mean and standard deviation for each node and channel across multiple feature matrices.
        Returns:
            tuple: A tuple containing the means and standard deviations, each with size (num_nodes, num_feats).
        """
        # compute the mean and std per node and channel across multiple feature matrices
        finer_scale_id = self.get_finer_scale()

        num_nodes0, num_feats0 = self.feature_matrices[finer_scale_id].shape
        means = np.zeros((num_nodes0, num_feats0))
        stds = np.zeros((num_nodes0, num_feats0))
        for node in range(num_nodes0):
            for channel in range(num_feats0):
                values = []
                for scale, fm in enumerate(self.feature_matrices):
                    if scale == finer_scale_id:
                        values.append([self.feature_matrices[scale][node, channel]])
                    else:
                        values.append(fm[self.node_maps[scale][node], self.kernels_maps[scale][channel]])
                means[node, channel] = np.mean(values)
                stds[node, channel] = np.std(values)
        return means, stds

    def sum_value(self, feature_matrix, values : list, func_condition = lambda x, y: x > y):
        """ 
        sum a value to the feature matrix based on a condition
        Args:
            feature_matrix (np.ndarray): The feature matrix to be modified.
            values (list): A list of values to be added to each feature channel.
            func_condition (function, optional): A function that takes two arguments (feature value, threshold value) and returns a boolean. Default is lambda x, y: x > y.
        """
        sum_matrix = np.zeros(feature_matrix.shape)
        for i, val in enumerate(values):
            sum_matrix[:,i] = np.array([val if func_condition(f, val) else 0 for f in feature_matrix[:,i]])
        return feature_matrix+sum_matrix

    def get_finer_scale(self):
        return np.argmax([self.feature_matrices[i].shape[0] for i in range(len(self.feature_matrices))])        

    def merge_scales_dotprod(self, thresh=False, std=False):
        """ merge scales with a dot product

        This solution works well when the object activation is stronger than the background activation.
        However, when there are several objects with different intensities, 
        the dot product with further min-max normalization may favor only objects with higher intensities, 
        while objects with lower intensities may be lost.
        """

        # get the largest number of nodes
        scale_largest_nnodes = self.get_finer_scale()
        merged_features = copy.deepcopy(self.feature_matrices[scale_largest_nnodes]) # start with the graph of the scale with largest number of nodes)
        feat_matrix_increased = self.feature_matrices[scale_largest_nnodes]

        if thresh:
            thresholds = self.compute_otsu_threshold(merged_features)
            merged_features = self.sum_value(merged_features, thresholds)

        for i, fm in enumerate(self.feature_matrices):
            if i == scale_largest_nnodes:
                continue

            fm = self.upsample_featMatrix(fm, feat_matrix_increased, self.node_maps[i], self.kernels_maps[i])
            if thresh:
                thresholds = self.compute_otsu_threshold(fm)
                fm = self.sum_value(fm, thresholds)
            merged_features *= fm
        
        if std:
            __, stds = self.compute_mean_std()
            for i in range(merged_features.shape[0]):
                merged_features[i, :] *= (1 - stds[i, :])

        return merged_features

    def merge_scales_dotprod_sum_scale0(self, thresh=False, std=False):
        """ merge scales with a dot product and sum with the first scale
        """

        # get the largest number of nodes
        scale_largest_nnodes = self.get_finer_scale()
        merged_features = copy.deepcopy(self.feature_matrices[scale_largest_nnodes]) # start with the graph of the scale with largest number of nodes)
        feat_matrix_increased = self.feature_matrices[scale_largest_nnodes]

        if thresh:
            thresholds = self.compute_otsu_threshold(merged_features)
            merged_features = self.sum_value(merged_features, thresholds)

        for i, fm in enumerate(self.feature_matrices):
            if i == scale_largest_nnodes:
                continue

            fm = self.upsample_featMatrix(fm, feat_matrix_increased, self.node_maps[i], self.kernels_maps[i])
            if thresh:
                thresholds = self.compute_otsu_threshold(fm)
                fm = self.sum_value(fm, thresholds)
            merged_features *= fm
        
        if std:
            __, stds = self.compute_mean_std()
            for i in range(merged_features.shape[0]):
                merged_features[i, :] *= (1 - stds[i, :])

        merged_features += feat_matrix_increased

        return merged_features

    def merge_scales_dotprod_sum_scalei(self, thresh=False, std=False):
        """ merge scales with a dot product and sum with the first scale
        """

        # get the largest number of nodes
        scale_largest_nnodes = self.get_finer_scale()
        merged_features = copy.deepcopy(self.feature_matrices[scale_largest_nnodes]) # start with the graph of the scale with largest number of nodes)
        feat_matrix_increased = self.feature_matrices[scale_largest_nnodes]

        if thresh:
            thresholds = self.compute_otsu_threshold(merged_features)
            merged_features = self.sum_value(merged_features, thresholds)

        for i, fm in enumerate(self.feature_matrices):
            if i == scale_largest_nnodes:
                continue

            fm = self.upsample_featMatrix(fm, feat_matrix_increased, self.node_maps[i], self.kernels_maps[i])
            if thresh:
                thresholds = self.compute_otsu_threshold(fm)
                fm = self.sum_value(fm, thresholds)
            merged_features *= fm
        
        if std:
            __, stds = self.compute_mean_std()
            for i in range(merged_features.shape[0]):
                merged_features[i, :] *= (1 - stds[i, :])

        if(scale_largest_nnodes < len(self.feature_matrices)-1):
            fm = self.upsample_featMatrix(self.feature_matrices[-1], feat_matrix_increased, self.node_maps[-1], self.kernels_maps[-1])
        else:
            fm = feat_matrix_increased
        merged_features += fm

        return merged_features


    def merge_scales_sum(self, thresh=False, std=False):
        """ merge scales with a sum

        This solution works well when the object activation is stronger than the background activation.
        However, when there are several objects with different intensities, 
        the sum may favor only objects with higher intensities, 
        while objects with lower intensities may be lost.
        """

        # get the largest number of nodes
        scale_largest_nnodes = self.get_finer_scale()
        merged_features = copy.deepcopy(self.feature_matrices[scale_largest_nnodes]) # start with the graph of the scale with largest number of nodes)
        feat_matrix_increased = self.feature_matrices[scale_largest_nnodes]

        if thresh:
            thresholds = self.compute_otsu_threshold(merged_features)
            merged_features = self.sum_value(merged_features, thresholds)

        for i, fm in enumerate(self.feature_matrices):
            if i == scale_largest_nnodes:
                continue

            fm = self.upsample_featMatrix(fm, feat_matrix_increased, self.node_maps[i], self.kernels_maps[i])
            if thresh:
                thresholds = self.compute_otsu_threshold(fm)
                fm = self.sum_value(fm, thresholds)
            merged_features += fm

        if std:
            __, stds = self.compute_mean_std()
            for i in range(merged_features.shape[0]):
                merged_features[i, :] *= (1 - stds[i, :])

        return merged_features

    def add_node_map(self, spx_img_i : np.ndarray, spx_img_0 : np.ndarray):
        """
        Add a node map from the first scale (scale 0) to the current scale (scale i).
        Args:
            spx_img_i (np.ndarray): The superpixel image of the current scale (scale i).
            spx_img_0 (np.ndarray): The superpixel image of the finer scale (with more nodes).
        """
        # add a node maps from a scale 0 to scale i
        self.spx_imgs.append(spx_img_i)
        spx_img0 = spx_img_0.flatten()
        spx_img_i = spx_img_i.flatten()
        node_map = np.zeros(spx_img0.max(), dtype=int)
        for j,p in enumerate(spx_img0):
            node_map[p - 1] = int(spx_img_i[j] - 1)
        return node_map
        
    def compute_node_maps(self):
        if len(self.spx_imgs) < 2:
            raise ValueError("compute_node_maps: invalid number of superpixel maps :", len(self.spx_imgs))
        
        finer_scale_id = self.get_finer_scale()
        self.node_maps = []
        sp_img0 = self.spx_imgs[finer_scale_id].flatten()
        for i in range(len(self.spx_imgs)):
            sp_img = self.spx_imgs[i]
            if i == finer_scale_id:
                node_map = [j for j in range(sp_img0.max())]
            else:
                sp_img = sp_img.flatten()
                node_map = self.add_node_map(sp_img, sp_img0)
            self.node_maps.append(node_map)

class GFLIMFunctions():
    """
    A class to hold the paths to the GFLIM functions.
    """
    def __init__(self, path):
        self.functions = {
            'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path),
            'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path),
            'ConvertSeeds2': '{}/iftGFLIMConvertSeeds_v2'.format(path),
            'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(path),
            'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(path),
            'WriteActivations': '{}/iftGFLIMWriteActivations'.format(path),
            'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(path),
            'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(path),
            'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path),
            'Extract': '{}/iftGFLIMExtract2'.format(path),
            'WriteKernelsCorrespondence': '{}/iftGFLIMWriteKernelsCorrespondence'.format(path),
            'CreateSpxImageSH': 'python3 /home/isabela/Codes/SH/src/boruvkasupix.py',
            'CreateGraphFromLabels': '{}/iftCreateGFLIMGraphFromLabels'.format(path),
            'SkipConnection': '{}/iftGFLIMSkipConnection'.format(path),
            'MergeScales': '{}/iftGFLIMMergeScales'.format(path),
            'ExtractMultiscale': '{}/iftGFLIMExtractMultiscale'.format(path)
        }

    def __callFunction__(self, function, args, msg=None):
        """
        Call a GFLIM function with the given arguments.
        Args:
            function (str): The function to call.
            args (list): The arguments to pass to the function.
            msg (str, optional): A message to display while calling the function.
        """
        if(msg is not None):
            sys.stdout.write(msg)
            sys.stdout.flush()
        print("Calling function: {} {}".format(function, " ".join([str(i) for i in args])))
        os.system("{} {}".format(function, " ".join([str(i) for i in args])))

    def create_graphs(self, images_path, final_spx : list, graphs_path_src, graph_feats, spx_alg, init_seeds=0):
        """
        Create the graphs for the dataset if they do not exist.
        Args:
            images_path (str): images directory
            final_spx (int): Number of superpixels.
            graphs_path_src (str): Path where the graphs and labels will be stored.
            graph_feats (str): Type of graph features to use.
            spx_alg (str): Superpixel algorithm to use.
            init_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
        """
        if spx_alg in ['DISF', 'ERGC']:
            if len(final_spx) != 1:
                raise ValueError("create_graphs: final_spx must have only one value when spx_alg is DISF or ERGC")
            self.__callFunction__(self.functions['CreateGraph'], [images_path, final_spx[0], graphs_path_src, graph_feats, spx_alg, init_seeds], 'Creating graphs')
        else:
            self.__callFunction__(self.functions['CreateSpxImageSH'], [f'--img {images_path}', f'--save {graphs_path_src}', f'--node_feat {graph_feats}', f'--k {",".join([str(spx) for spx in final_spx])}'], 'Creating superpixel images')
            for spx in final_spx:
                self.__callFunction__(self.functions['CreateGraphFromLabels'], [f'{images_path}', f'{graphs_path_src}/{spx}/labels0', graph_feats, f'{graphs_path_src}/{spx}/layer0'], 'Creating graphs from superpixel images')

    def create_layer_model(self, model_dir, arch_file, layer, stride, attention, aggregation):
        """
        Create the layer model for the given architecture.
        Args:
            model_dir (str): Directory where the model will be stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
        """
        self.__callFunction__(self.functions['CreateLayerModel'], [model_dir, arch_file, layer, stride, attention, aggregation], "Creating weights layer {}:\t".format(layer))

    def encode_layer(self, model_dir, arch_file, layer, stride, attention, aggregation):
        """
        Encode the layer for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
        """
        self.__callFunction__(self.functions['EncodeLayer'], [model_dir, arch_file, layer, stride, attention, aggregation], "Encoding layer {}:\t\t".format(layer))

    def write_activations(self, model_dir, layer, batchsize, skip=None):
        """
        Write the activations for the given layer.
        Args:
            model_dir (str): Directory where the model is stored.
            layer (int): Layer number.
            skip (int, optional): If provided, use skip connections with the given layer. Defaults to None.
        """
        if skip == None:
            self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer, batchsize], "Writing activations layer {}:\t".format(layer))
        else:
            self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer, batchsize, skip], "Writing activations layer {} with skip connection:\t".format(layer))

    def merge_layer_models(self, model_dir, layer):
        """
        Merge the layer models for the given layer.
        Args:
            model_dir (str): Directory where the model is stored.
            layer (int): Layer number.
        """
        self.__callFunction__(self.functions['MergeLayerModels'], [model_dir, layer], "Merging weights layer {}:\t".format(layer))

    def encode_merged_layer(self, model_dir, arch_file, layer, stride, attention, aggregation, dropout=False):
        """
        Encode the merged layer for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
            dropout (bool): Whether to apply dropout.
        """
        self.__callFunction__(self.functions['EncodeMergedLayer'], [model_dir, arch_file, layer, stride, attention, aggregation, 1 if dropout else 0], "Encoding Merged layer {}:\t\t".format(layer))

    def decode_layer(self, model_dir, arch_file, layer, d, stride, batchsize=1):
        """
        Decode the layer for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            d (str): Decoding type ('hybrid', 'graph-adaptive', 'gaussian', 'test', 'tristate').
            stride (str): Stride type ('graph-based', 'none', or 'image-based').
        """
        self.__callFunction__(self.functions['DecodeLayer'], [model_dir, layer, arch_file, d, stride, batchsize], "Decoding layer {}:\t\t".format(layer))

    def convert_seeds(self, model_dir, markers_dir, nfeat_markers, kernel_size, attention, aggregation):
        """
        Convert the seeds for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            markers_dir (str): Directory where the image markers are stored.
            nfeat_markers (int): Number of feature points per marker (if -1, do not perform k-means).
            kernel_size (int): Kernel size.
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
        """
        self.__callFunction__(self.functions['ConvertSeeds'], [model_dir, markers_dir, nfeat_markers, kernel_size, attention, aggregation], 'Converting seeds:\t\t')

    def extract(self, img_dir, graphs_dir, spx_dir, model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, dropout, save_intermediate, layers_save_act, layers_decode, decoders):
        """
        Extract features for the given architecture.
        Args:
            img_dir (str): Directory where the images are stored.
            graphs_dir (str): Path to the graphs directory.
            spx_dir (str): Path to the superpixel directory.
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            nlayers (int): Number of layers.
            batch_size (int): Batch size.
            stride (str): Stride type ('graph-based', 'none', or 'image-based').
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
            dropout (bool): Whether to apply dropout.
            save_intermediate (bool, optional): Whether to save intermediate features and superpixel labels.
            layers_save_act (list): List of layers to save activations.
            layers_decode (list): List of layers to decode.
            decoders (list): List of decoders to use.
        """
        self.__callFunction__(self.functions['Extract'], [img_dir, graphs_dir, spx_dir, model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, 1 if dropout else 0, 1 if save_intermediate else 0, ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1, ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1, ",".join(str(i) for i in decoders)], 'Extracting features:\t\t')

    def extract_multiscale(self, img_dir, graphs_dir, spx_dir, model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, dropout, save_intermediate, layers_save_act, layers_decode, decoders, merge_option, superpixels_list, kernels_map_file):
        """
        Extract features for the given architecture.
        Args:
            img_dir (str): Directory where the images are stored.
            graphs_dir (list of str): Graphs directory list.
            spx_dir (list of str): Superpixel directory list.
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            nlayers (int): Number of layers.
            batch_size (int): Batch size.
            stride (str): Stride type ('graph-based', 'none', or 'image-based').
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN, WCONCAT)
            dropout (bool): Whether to apply dropout.
            save_intermediate (bool, optional): Whether to save intermediate features and superpixel labels.
            layers_save_act (list): List of layers to save activations.
            layers_decode (list): List of layers to decode.
            decoders (list): List of decoders to use.
            merge_option (str): Merge option ('dotprod', 'dotprod_sumscale0', 'dotprod_sumscalei', 'sum').
            superpixels_list (list): List of superpixel values.
            kernels_map_file (str): Path to the kernels map file.
        """
        self.__callFunction__(self.functions['ExtractMultiscale'], [img_dir, ",".join(graphs_dir), ",".join(spx_dir), 
                        model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, 
                        1 if dropout else 0, 1 if save_intermediate else 0, 
                        ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1, 
                        ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1, 
                        ",".join(str(i) for i in decoders),
                        merge_option, ",".join(str(i) for i in superpixels_list), kernels_map_file], 
                        'Extracting features:\t\t')
        
    def write_kernels_correspondence(self, markers_dir_scale0, spx_dir_scale0, spx_dir_scalei, save_file, save_markers_folder=None):
        """
        Write the kernels correspondence between scale 0 and scale i.
        Args:
            markers_dir_scale0 (str): Directory where the markers of scale 0 are stored.
            spx_dir_scale0 (str): Directory where the superpixels of scale 0 are stored.
            spx_dir_scalei (str): Directory where the superpixels of scale i are stored.
            save_file (str): Path to the file where the correspondence will be saved.
            save_markers_folder (str, optional): Directory where the markers will be saved.
        """

        if save_markers_folder is None:
            self.__callFunction__(self.functions['WriteKernelsCorrespondence'], [markers_dir_scale0, spx_dir_scale0, spx_dir_scalei, save_file], 'Writing kernels correspondence:\t')
        else:
            self.__callFunction__(self.functions['WriteKernelsCorrespondence'], [markers_dir_scale0, spx_dir_scale0, spx_dir_scalei, save_file, save_markers_folder], 'Writing kernels correspondence:\t')

    def skip_connection(self, model_dir, images_dir, arch_file, layer):
        """
        Merge scales of different layers as skip connections.
        Args:
            model_dir (str): Directory where the model is stored.
            images_dir (str): Directory where the images are stored.
            arch_file (str): Path to the architecture file.
            layer (int): Current layer.
        """
        self.__callFunction__(self.functions['SkipConnection'], [model_dir, images_dir, arch_file, layer], 'Merging scales:\t\t')

    def merge_scales(self, images_dir, superpixels_list, model_dir, kernels_map_file, layer, merge_option):
        """
        Merge scales of same layer with different superpixels (different hierarchy levels).
        Args:
            images_dir (str): Directory where the images are stored.
            model_dir (str): Directory where the model is stored.
            superpixels_list (str): Comma-separated list of superpixel directories.
            kernels_map_file (str): Path to the kernels map file.
            layer (int): Current layer.
            merge_option (str): Merge option ('dotprod', 'dotprod_sum_scale0', 'dotprod_sum_scalei', 'sum').
        """
        self.__callFunction__(self.functions['MergeScales'], [images_dir, ','.join(str(spx) for spx in superpixels_list), model_dir, kernels_map_file, layer, merge_option], 'Merging scales:\t\t')


class GFLIMModel():
    """
    A class to hold the GFLIM model and its parameters.
    1. Create the graphs if they do not exist.
    2. Train the model.
    3. Extract features from the model.
    """
    def __init__(self, model_dir, arch_file, markers_dir,
                 spx_alg, init_seeds, final_spx, graph_feats, 
                 stride, attention, aggregation, dropout):
        """
        Initialize the GFLIM model with the given parameters.
        Args:
            model_dir (str): Directory where the model will be stored.
            arch_file (str): Path to the architecture file.
            markers_dir (str): Directory where the image markers are stored.
            spx_alg (str): Superpixel algorithm to use ('DISF', 'ERGC', 'SLIC', 'LSC', 'SH').
            init_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
            final_spx (int or list): Final number of superpixels. If spx_alg is 'DISF' or 'ERGC', this is an integer. If spx_alg is 'SLIC', 'LSC', or 'SH', this is a list of integers.
            graph_feats (str): Type of graph features to use ('RGB', 'RGBXY', 'HSV', 'HSVI', 'LAB', 'LABI').
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy ('CONCAT', 'MEAN', 'WMEAN', WCONCAT).
            dropout (bool): Whether to apply dropout.
        """
        
        self.spx_alg = spx_alg
        self.init_seeds = init_seeds
        self.final_spx = final_spx
        self.graph_feats = graph_feats
        self.attention = attention
        self.aggregation = aggregation
        self.dropout = dropout
        self.arch_file = arch_file
        self.stride = stride
        self.markers_dir = markers_dir
        self.model_dir = model_dir

        self.functions = GFLIMFunctions('/home/isabela/Codes/LibIFT/bin')
        self.arch = self.read_architecture(self.arch_file)

    def __train_layer__(self, layer, write_activation, batchsize, decoders=[]):
        """
        Train a single layer of the model.
        Args:
            layer (int): Layer number.
            write_activation (bool): Whether to write the activations.
            decoders (list): List of decoders to use.
        """
        self.functions.create_layer_model(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        
        #if write_activation:
        #    self.functions.encode_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        #    self.functions.write_activations(self.model_dir, layer, batchsize)

        self.functions.merge_layer_models(self.model_dir, layer)
        self.functions.encode_merged_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, self.dropout)
        
        if write_activation:
            self.functions.write_activations(self.model_dir, layer, batchsize)

        for d in decoders: #["hybrid", "graph-adaptive"]:#, "test", "tristate"]:
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def __copyGraphsAndLabels__(self, src_dir, graphs_path_src, spx_label_path_src):
        """
        Copy the graphs and labels from the source directory to the model directory.
        Args:
            src_dir (str): Source directory where the graphs and labels are stored.
        """
        if os.path.exists(os.path.join(self.model_dir, 'layer0')):
            shutil.rmtree(os.path.join(self.model_dir, 'layer0'), ignore_errors=True)
        if os.path.exists(os.path.join(self.model_dir, 'labels0')):
            shutil.rmtree(os.path.join(self.model_dir, 'labels0'), ignore_errors=True)

        os.makedirs(os.path.join(self.model_dir, 'layer0'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'labels0'), exist_ok=True)

        for file_name in os.listdir(src_dir):

            if file_name.endswith('-seeds.txt'):
                graph_name = file_name.split('-')[0]
            else:
                graph_name = file_name.split('.')[0]

            graph_src_file = os.path.join(graphs_path_src, f'{graph_name}.json')
            if not os.path.exists(graph_src_file):
                print(f"Graph source file {graph_src_file} does not exist.")
                exit()
            if not os.path.isfile(os.path.join(self.model_dir, 'layer0', f'{graph_name}.json')):
                os.symlink(graph_src_file, os.path.join(self.model_dir, 'layer0', f'{graph_name}.json'))

            labels_src_file = os.path.join(spx_label_path_src, f'{graph_name}.pgm')
            if not os.path.exists(labels_src_file):
                print(f"Labels source file {labels_src_file} does not exist.")
                exit()
            if not os.path.isfile(os.path.join(self.model_dir, 'labels0', f'{graph_name}.pgm')):
                os.symlink(labels_src_file, os.path.join(self.model_dir, 'labels0', f'{graph_name}.pgm'))

    def get_first_scale(self, layer):
        """
        Get the first scale of the given layer considering skip connections.
        Args:
            layer (int): Layer number.
        Returns:
            int: The first scale of the layer.
        Note: If there are no skip connections, the first scale is the layer itself. 
        If there are skip connections, the first scale is the first layer that does not have a skip connection.
        """
        if 'skip_connection' not in self.arch[f'layer{layer}'].keys():
            return layer
        
        finer_layer = [i for i in range(1, layer+1)]
        for l in range(layer):
            for r in range(l):
                if 'skip_connection' in self.arch['layer'+str(l+1)].keys() and self.arch['layer'+str(l+1)]['skip_connection'][r] > 0 and finer_layer[r] < finer_layer[l]:
                    finer_layer[l] = finer_layer[r]
        return finer_layer[layer-1]

    def skip_connection(self, layer, images_folder, first_scale, write_activation, decoders, batchsize):
        """
        Merge the layers with skip connections.
        Args:
            layer (int): Layer number.
            images_folder (str): Path to the images folder.
            first_scale (int): The first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            write_activation (bool): Whether to write the activations.
        """
        
        # Copy the graphs and labels of the layer to a new directory.
        #shutil.copytree(os.path.join(self.model_dir, f'layer{layer}'), os.path.join(self.model_dir, f'layer{layer}_noSkip'), dirs_exist_ok=True)
        #shutil.copytree(os.path.join(self.model_dir, f'labels{layer}'), os.path.join(self.model_dir, f'labels{layer}_noSkip'), dirs_exist_ok=True)

        self.functions.skip_connection(self.model_dir, images_folder, self.arch_file, layer)

        # write activations of the merged data
        if write_activation:
            shutil.rmtree(os.path.join(self.model_dir, f'activation{layer}'), ignore_errors=True)
            self.functions.write_activations(self.model_dir, layer, batchsize, first_scale)

        for d in decoders: 
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def train_model(self, images_folder, 
                    layers_activations, 
                    layers_decode, decoders, 
                    batchsize):
        """
        Train the GFLIM model.
        Args:
            images_folder (str): Path to the images folder.
            decoders (list): List of decoders to use.
            write_activation (bool): Whether to write the activations.
        Steps:
            1. Copy the architecture file to the model directory.
            2. Convert the seeds using the architecture parameters.
            3. For each layer in the architecture:
                a. Train the layer.
                b. Write the kernels correspondence between layer 1 and the others
                c. If the layer has skip connections, merge.
        """

        # copy arch file to the model directory
        file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
        if os.path.isfile(file): # if it exists, remove it
            os.remove(file)
        shutil.copy(self.arch_file, self.model_dir)
        
        # convert from image seeds to graph seeds and perform k-means
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch['layer1']['conv']["nkernels_per_marker"], self.arch['layer1']['conv']['kernel_size'], self.attention, self.aggregation)
        
        for layer in range(1, self.arch['nlayers'] + 1):
            # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            first_scale = self.get_first_scale(layer)

            self.__train_layer__(layer, 
                                 first_scale == layer or layer in layers_activations, 
                                 batchsize, 
                                 decoders if first_scale == layer or layer in layers_decode else [])

            if layer > 1:
                self.functions.write_kernels_correspondence(os.path.join(self.model_dir, f'markers'),
                                                                os.path.join(self.model_dir, 'labels0'),
                                                                os.path.join(self.model_dir, f'labels{layer-1}'),
                                                                os.path.join(self.model_dir, 'model', 'kernels_map.txt'))

            # merge layers output
            if first_scale != layer: # if there are skip connections
                self.skip_connection(layer, images_folder, first_scale, 
                                     layer in layers_activations, 
                                     decoders if layer in layers_decode else [], batchsize)

    def read_architecture(self, arch_file):
        """
        Read the architecture file.
        Args:
            arch_file (str): Path to the architecture file.
        Returns:
            dict: The architecture parameters.
        """
        with open(arch_file) as f:
            data = json.load(f)
        return data

    def run_model(self, images_folder, process='train', 
                        save_intermediate=False, layers_activations = [-1],
                        layers_decode = [-1], decoders=["hybrid", "graph-adaptive"], 
                        batchsize=64, graphs_path_src=None, spx_label_path_src=None):
        """
        Run (train and/or extract) a GFLIM model.
        Args:
            images_folder (str): Path to the images folder.
            train (bool): Whether to train the model.
            decode_train (bool): Whether to decode the training set.
            activation_train (bool): Whether to write the activations for the training set.
            extract (bool): Whether to extract features from the model.
            decode_extract (bool): Whether to decode the extracted features.
            activation_extract (bool): Whether to write the activations for the extracted features.
            decoders (list): List of decoders to use.
            batchsize (int): Batch size.
            graphs_path_src (str): Path to the graphs directory. If None, graphs will be created.
            spx_label_path_src (str): Path to the superpixel labels directory. If None, labels will be created.
            save_intermediate (bool, optional): Whether to save intermediate features and superpixel labels.
            layers_activations (list): List of layers to save activations.
            layers_decode (list): List of layers to decode.
        
        Steps:
            1. If neither training nor extraction is set, print a message and return.
            2. Remove any existing layer and label directories in the model directory.
            3. If training is set:
                a. Copy the graphs and labels for the training set.
                b. Train the model.
            4. If extraction is set:
                a. Copy the graphs and labels for the entire dataset.
                b. Extract features from the model.
                c. For each layer in the architecture:
                    i. If decoding is set, decode the layer using the specified decoders.
                    ii. If writing activations is set, write the activations for the layer.
                    iii. Remove the layer and label directories for the layer.
        """
        
        if process != 'train' and process != 'extract':
            print("Nothing to do. Please set train and/or extract to True.")
            return

        # remove any existing layer and label directories in the model directory to do not mix train and extract data
        for layer in range(0,self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        print("graphs path:" , graphs_path_src, " spx labels path:", spx_label_path_src)
        if(graphs_path_src is not None and spx_label_path_src is not None and os.path.exists(graphs_path_src) and os.path.exists(spx_label_path_src)):
            # copy the graphs and superpixel labels from the provided directories to the model directory
            self.__copyGraphsAndLabels__(images_folder, graphs_path_src, spx_label_path_src)
        else:
            # create the graphs and superpixel labels if they do not exist
            self.functions.create_graphs(images_folder, [self.final_spx], self.model_dir, self.graph_feats, self.spx_alg, self.init_seeds)
            graphs_path_src = f'{self.model_dir}/layer0'
            spx_label_path_src = f'{self.model_dir}/labels0'

        if process == 'train':
            shutil.rmtree(f'{self.model_dir}/model', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/markers', ignore_errors=True)

            self.train_model(images_folder, layers_activations, layers_decode, decoders, batchsize)

            # remove the layer and label directories to free up space and allow the extraction to be done from scratch
            for layer in range(0,self.arch['nlayers']+1):
                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if process == 'extract':
            # copy the graphs and superpixel labels from the dataset directory to the model directory
            self.functions.extract(images_folder, 
                                   graphs_path_src, 
                                   spx_label_path_src, 
                                   self.model_dir, self.arch_file, self.arch['nlayers'], batchsize, 
                                   self.stride, self.attention, self.aggregation, self.dropout,
                                   save_intermediate, 
                                   layers_activations, 
                                   layers_decode,
                                   decoders)
            

            # decode the output features, write the activations, and remove the layer and label directories to free up space
            #for layer in range(1,self.arch['nlayers']+1):
            #    if decode_extract:
            #        for d in decoders:
            #            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride, batchsize)
            #    if activation_extract:
            #        self.functions.write_activations(self.model_dir, layer)

            #    shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            #    shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

            #shutil.rmtree(f'{self.model_dir}/labels0', ignore_errors=True)
            #shutil.rmtree(f'{self.model_dir}/layer0', ignore_errors=True)

class GFLIMultiScaleUModel(GFLIMModel):
    def __init__(self, model_dir : str, arch_file : str, markers_dir : str, 
                 spx_alg : str, final_spx_list : list, graph_feats : str, 
                 stride : str, attention : str, aggregation : str, dropout : bool,
                 merge_option : str = 'dotprod'):

        if len(final_spx_list) < 1:
            raise ValueError("final_spx_list must contain at least one superpixel scale.")

        super().__init__(model_dir, arch_file, markers_dir,
                         spx_alg, 0, final_spx_list[0], graph_feats, 
                         stride, attention, aggregation, dropout)

        self.final_spx_list = final_spx_list
        self.save_intermediate = False
        self.merge_option = merge_option

    def convert_seeds(self, spx_path0, spx_pathi, graph_seeds_path_src, graph_seeds_path_dst):

        if not os.path.exists(graph_seeds_path_dst):
            os.makedirs(graph_seeds_path_dst, exist_ok=True)

        for train_img in os.listdir(graph_seeds_path_src):
            img_name = train_img.split('-')[0]

            spx_img0 = np.array(Image.open(os.path.join(spx_path0, f'{img_name}.pgm')))
            spx_imgi = np.array(Image.open(os.path.join(spx_pathi, f'{img_name}.pgm')))

            graph_seeds_file_src = os.path.join(graph_seeds_path_src, train_img)
            if not os.path.exists(graph_seeds_file_src):
                raise FileNotFoundError(f"Markers file {graph_seeds_file_src} does not exist.")

            # read graph-based seeds file
            with open(graph_seeds_file_src, 'r') as f:
                lines = f.readlines()
            nmarkers = int(lines[0].split()[0])
            markers = reversed([tuple(map(int, line.split())) for line in lines[1:nmarkers + 1]])

            # create a mapping from superpixel in spx_img0 to superpixel in spx_imgi (from scale 0 to scale i)
            # Vectorized mapping from superpixel in spx_img0 to spx_imgi
            spx_imgi_flat = spx_imgi.flatten()
            spx_map = np.zeros(spx_img0.max() + 1, dtype=int)
            # Find first occurrence indices for each superpixel in spx_img0
            unique_spx, first_idx = np.unique(spx_img0.flatten(), return_index=True)
            spx_map[unique_spx] = spx_imgi_flat[first_idx]

            # create new markers list with the new superpixel indices
            """
            markers_new : list of tuples (spx, m, label) with unique superpixels of scale i
            spx_id_map : dictionary mapping superpixel index to its first occurrence index (to keep order)
            kernels_id_map : list of superpixel indices in the order they were added (to keep order of kernels)
            """
            # Optimize marker mapping using numpy and dictionary comprehension
            # Use numpy for efficient mapping and dictionary comprehension for marker assignment
            markers_new = {
                spx_map[spx]: (m, label)
                for spx, m, label in markers
                if label != 0 or spx_map[spx] not in locals().get('markers_new', {})
            }

            graph_seeds_file_dst = os.path.join(graph_seeds_path_dst, f'{img_name}-seeds_graph.txt')
            # write new markers file
            with open(graph_seeds_file_dst, 'w') as f:
                f.write(f"{len(markers_new)}\n")
                for spx, (m, label) in reversed(list(markers_new.items())):
                    f.write(f"{spx} {m} {label}\n")

    def convert_seeds_bkp2(self, spx_path0, spx_pathi, graph_seeds_path_src, graph_seeds_path_dst):

        if not os.path.exists(graph_seeds_path_dst):
            os.makedirs(graph_seeds_path_dst, exist_ok=True)

        for train_img in os.listdir(graph_seeds_path_src):
            img_name = train_img.split('-')[0]
            
            spx_file0 = os.path.join(spx_path0, f'{img_name}.pgm')
            spx_filei = os.path.join(spx_pathi, f'{img_name}.pgm')

            spx_img0 = np.array(Image.open(spx_file0))
            spx_imgi = np.array(Image.open(spx_filei))

            graph_seeds_file_src = os.path.join(graph_seeds_path_src, train_img)
            if not os.path.exists(graph_seeds_file_src):
                raise FileNotFoundError(f"Markers file {graph_seeds_file_src} does not exist.")

            # read graph-based seeds file
            with open(graph_seeds_file_src, 'r') as f:
                lines = f.readlines()
            
            # first line contains the number of markers, width and height
            nmarkers = int(lines[0].strip().split()[0])
            markers = []
            # subsequent lines contain the x, y coordinates of the markers
            for line in lines[1:nmarkers + 1]:
                spx, m, label = map(int, line.strip().split())
                markers.append((spx, m, label))

            # create a mapping from superpixel in spx_img0 to superpixel in spx_imgi (from scale 0 to scale i)
            # Vectorized mapping from superpixel in spx_img0 to spx_imgi
            spx_img0_flat = spx_img0.flatten()
            spx_imgi_flat = spx_imgi.flatten()
            spx_map = np.zeros(spx_img0.max() + 1, dtype=int)
            # Find first occurrence indices for each superpixel in spx_img0
            unique_spx, first_idx = np.unique(spx_img0_flat, return_index=True)
            spx_map[unique_spx] = spx_imgi_flat[first_idx]

            # create new markers list with the new superpixel indices
            """
            markers_new : list of tuples (spx, m, label) with unique superpixels of scale i
            spx_id_map : dictionary mapping superpixel index to its first occurrence index (to keep order)
            kernels_id_map : list of superpixel indices in the order they were added (to keep order of kernels)
            """
            markers_new  = {}
            for (spx, m,label) in reversed(markers):
                new_spx = spx_map[spx+1] - 1
                if new_spx not in markers_new or label != 0:
                    markers_new[new_spx] = (m, label)
                
            graph_seeds_file_dst = os.path.join(graph_seeds_path_dst, f'{img_name}-seeds_graph.txt')
            # write new markers file
            with open(graph_seeds_file_dst, 'w+') as f:
                f.write(f"{len(markers_new)}\n")
                for spx, (m, label) in reversed(markers_new.items()):
                    f.write(f"{spx} {m} {label}\n")

    def convert_seeds_bkp(self, spx_path0, spx_pathi, graph_seeds_path_src, graph_seeds_path_dst, kernels_map_file):

        kernels_id_map = []
        if not os.path.exists(graph_seeds_path_dst):
            os.makedirs(graph_seeds_path_dst, exist_ok=True)

        for train_img in os.listdir(graph_seeds_path_src):
            img_name = train_img.split('-')[0]
            
            spx_file0 = os.path.join(spx_path0, f'{img_name}.pgm')
            spx_filei = os.path.join(spx_pathi, f'{img_name}.pgm')

            spx_img0 = np.array(Image.open(spx_file0))
            spx_imgi = np.array(Image.open(spx_filei))

            graph_seeds_file_src = os.path.join(graph_seeds_path_src, train_img)
            if not os.path.exists(graph_seeds_file_src):
                raise FileNotFoundError(f"Markers file {graph_seeds_file_src} does not exist.")

            # read graph-based seeds file
            with open(graph_seeds_file_src, 'r') as f:
                lines = f.readlines()
            
            # first line contains the number of markers, width and height
            nmarkers = int(lines[0].strip().split()[0])
            markers = []
            max_label = 0
            # subsequent lines contain the x, y coordinates of the markers
            for line in lines[1:nmarkers + 1]:
                spx, m, label = map(int, line.strip().split())
                markers.append((spx, m, label))
                max_label = max(max_label, label)

            # create a mapping from superpixel in spx_img0 to superpixel in spx_imgi (from scale 0 to scale i)
            spx_img0_flat = spx_img0.flatten()
            spx_imgi_flat = spx_imgi.flatten()
            spx_map = {}
            for p in range(len(spx_img0_flat)):
                spx = spx_img0_flat[p]
                if spx not in spx_map:
                    spx_map[spx] = spx_imgi_flat[p]


            # create new markers list with the new superpixel indices
            """
            markers_new : list of tuples (spx, m, label) with unique superpixels of scale i
            spx_id_map : dictionary mapping superpixel index to its first occurrence index (to keep order)
            kernels_id_map : list of superpixel indices in the order they were added (to keep order of kernels)
            """
            markers_new, spx_id_map,  = {}, {}
            for i, (spx, m,label) in enumerate(markers):

                new_spx = spx_map[spx+1] - 1
                id = spx_id_map[new_spx] if new_spx in spx_id_map else nmarkers - i - 1

                if new_spx not in markers_new or label != 0:
                    markers_new[new_spx] = (m, label)
                    spx_id_map[new_spx] = id # to keep the order of the superpixels
                
                kernels_id_map.append(id) # to keep the order of the kernels

            graph_seeds_file_dst = os.path.join(graph_seeds_path_dst, f'{img_name}-seeds_graph.txt')
            # write new markers file
            with open(graph_seeds_file_dst, 'w+') as f:
                f.write(f"{len(markers_new)}\n")
                for spx, (m, label) in markers_new.items():
                    f.write(f"{spx} {m} {label}\n")

        # append to the kernels map file the ids separated by a space
        with open(kernels_map_file, 'a+') as f:
            f.write(" ".join([str(i) for i in reversed(kernels_id_map)]) + "\n")

    def __train_layer__(self, layer, write_activation, batchsize, decoders=[]):
        self.functions.create_layer_model(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        self.functions.encode_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        
        #if write_activation:
        #    self.functions.write_activations(self.model_dir, layer, batchsize)

        self.functions.merge_layer_models(self.model_dir, layer)
        self.functions.encode_merged_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, self.dropout)
        
        if write_activation:
            self.functions.write_activations(self.model_dir, layer, batchsize)

        for d in decoders: #["hybrid", "graph-adaptive"]:#, "test", "tristate"]:
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def train_model(self, images_folder, 
                    layers_activations, 
                    layers_decode, decoders, 
                    batchsize):
           
        # copy arch file to the model directory
        file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
        if os.path.isfile(file): # if it exists, remove it
            os.remove(file)
        shutil.copy(self.arch_file, self.model_dir)

        spx0 = self.final_spx_list[0]
        shutil.copytree(f'{self.model_dir}/scales/{spx0}/layer0', f'{self.model_dir}/layer0', dirs_exist_ok=True)
        shutil.copytree(f'{self.model_dir}/scales/{spx0}/labels0', f'{self.model_dir}/labels0', dirs_exist_ok=True)
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch[f'layer1']['conv']["nkernels_per_marker"], self.arch[f'layer1']['conv']['kernel_size'], self.attention, self.aggregation)
        
        layer = 1
        self.__train_layer__(layer, 
                            layer in layers_activations, batchsize, 
                            decoders if layer in layers_decode else [])

        self.rename_files_model(layer, spx0, layer in layers_activations, layer in layers_decode, decoders)

        for spx in self.final_spx_list[1:]:
            self.__copyGraphsAndLabels__(images_folder, f'{self.model_dir}/scales/{spx}/layer0', f'{self.model_dir}/scales/{spx}/labels0')
            
            self.functions.write_kernels_correspondence(os.path.join(self.model_dir, 'scales', str(spx0), 'markers'),
                                                        os.path.join(self.model_dir, 'scales', str(spx0), 'labels0'),
                                                        os.path.join(self.model_dir, 'scales', str(spx), 'labels0'),
                                                        os.path.join(self.model_dir, 'model', f'kernels_map_{layer}_scales.txt'),
                                                        os.path.join(self.model_dir, f'markers'))
            
            self.__train_layer__(layer, 
                                layer in layers_activations, batchsize, 
                                decoders if layer in layers_decode else [])
            
            self.rename_files_model(layer, spx, layer in layers_activations, layer in layers_decode, decoders)
            
        # merge scales
        self.functions.merge_scales(images_folder, self.final_spx_list, self.model_dir, os.path.join(self.model_dir, 'model', f'kernels_map_{layer}_scales.txt'), layer, self.merge_option)
        
        # rename the scale 0 folders back to the original names
        folder_list =['markers', 'layer0', 'labels0']
        for folder in folder_list:
            #shutil.rmtree(f'{self.model_dir}/{folder}', ignore_errors=True)
            shutil.copytree(f'{self.model_dir}/scales/{spx0}/{folder}', f'{self.model_dir}/{folder}', dirs_exist_ok=True)
        for graph_name in os.listdir(f'{self.model_dir}/layer{layer}'):
            basename = os.path.join(self.model_dir, 'model', graph_name.split('.')[0])
            for post_name in [f'-conv{layer}-bias.txt', f'-conv{layer}-kernels.npy', f'-conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'{basename}-spx{spx0}-{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'{basename}{post_name}')
                    shutil.copy(src_filename, dst_filename)
        for post_name in [f'conv{layer}-bias.txt', f'conv{layer}-kernels.npy', f'conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'spx{spx0}-{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'{post_name}')
                    shutil.copy(src_filename, dst_filename)

        # write activations and decode the merged data
        if layer in layers_activations:
            shutil.rmtree(os.path.join(self.model_dir, f'activation{layer}'), ignore_errors=True)
            self.functions.write_activations(self.model_dir, layer, batchsize)
        if layer in layers_decode:
            for d in decoders: 
                self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

        for layer in range(2, self.arch['nlayers'] + 1):
            # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            first_scale = self.get_first_scale(layer)

            self.__train_layer__(layer, 
                                 first_scale == layer or layer in layers_activations, 
                                 batchsize, 
                                 decoders if first_scale == layer or layer in layers_decode else [])
            
            if layer > 1:
                self.functions.write_kernels_correspondence(os.path.join(self.model_dir, f'markers'),
                                                                os.path.join(self.model_dir, 'labels0'),
                                                                os.path.join(self.model_dir, f'labels{layer-1}'),
                                                                os.path.join(self.model_dir, 'model', 'kernels_map.txt'))
            
            # merge layers output
            if first_scale != layer: # if there are skip connections
                self.skip_connection(layer, images_folder, first_scale, 
                                     layer in layers_activations, 
                                     decoders if layer in layers_decode else [], batchsize)
        

    def rename_files_model(self, layer, spx, save_activations, save_decode, decoders):
        for graph_name in os.listdir(f'{self.model_dir}/layer{layer}'):
            basename = os.path.join(self.model_dir, 'model', graph_name.split('.')[0])
            for post_name in [f'-conv{layer}-bias.txt', f'-conv{layer}-kernels.npy', f'-conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'{basename}{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'{basename}-spx{spx}-{post_name}')
                    os.rename(src_filename, dst_filename)

        for post_name in [f'conv{layer}-bias.txt', f'conv{layer}-kernels.npy', f'conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'spx{spx}-{post_name}')
                    os.rename(src_filename, dst_filename)

        folder_list =['markers', f'layer{layer}', f'labels{layer}']
        if save_activations:
            folder_list += [f'activation{layer}', f'activationNoMerge{layer}']
        if save_decode:
            folder_list += [f'{d}_{layer}' for d in decoders]

        for folder in folder_list:
            src_path = os.path.join(self.model_dir, folder)
            if os.path.exists(src_path):
                shutil.copytree(src_path, os.path.join(self.model_dir, 'scales', str(spx), folder), dirs_exist_ok=True)
                shutil.rmtree(src_path, ignore_errors=True)

    def run_model(self, images_folder, process='train', 
                        save_intermediate=False, layers_activations = [-1],
                        layers_decode = [-1], decoders=["hybrid", "graph-adaptive"], 
                        batchsize=64, graphs_path_src=[], spx_label_path_src=[]):
        
        if process != 'train' and process != 'extract':
            print("Nothing to do. Please set train and/or extract to True.")
            return

        if len(self.final_spx_list) < 1:
            print("final_spx_list must contain at least one superpixel scale.")
            return

        if len(graphs_path_src) > 0 and len(graphs_path_src) != len(self.final_spx_list):
            print("graphs_path_src must contain the same number of elements as final_spx_list or be empty.")
            return
        
        if len(spx_label_path_src) > 0 and len(spx_label_path_src) != len(self.final_spx_list):
            print("spx_label_path_src must contain the same number of elements as final_spx_list or be empty.")
            return

        # remove any existing layer and label directories in the model directory to do not mix train and extract data
        for layer in range(0,self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if len(graphs_path_src) > 0 and len(spx_label_path_src) > 0:
            # verify if the provided paths exist
            for graphpath, spxpath in zip(graphs_path_src, spx_label_path_src):
                if not os.path.exists(graphpath) or not os.path.exists(spxpath):
                    print(f"Provided graphs_path_src or spx_label_path_src do not exist: {graphpath}, {spxpath}")
                    return
            # copy the graphs and labels from the provided directories to the model directory
            for graphpath, spxpath, spx in zip(graphs_path_src, spx_label_path_src, self.final_spx_list):
                self.__copyGraphsAndLabels__(images_folder, graphpath, spxpath)
                os.makedirs(f'{self.model_dir}/scales/{spx}', exist_ok=True)
                # rename the folder name
                shutil.copytree(f'{self.model_dir}/labels0', f'{self.model_dir}/scales/{spx}/labels0', dirs_exist_ok=True)
                shutil.copytree(f'{self.model_dir}/layer0', f'{self.model_dir}/scales/{spx}/layer0', dirs_exist_ok=True)
                shutil.rmtree(f'{self.model_dir}/labels0', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer0', ignore_errors=True)

        else:
            out_path = f'{self.model_dir}/scales/'
            os.makedirs(out_path, exist_ok=True)
            self.functions.create_graphs(images_folder, self.final_spx_list, out_path, self.graph_feats, self.spx_alg, self.init_seeds)
        
        self.save_intermediate = save_intermediate

        if process == 'train':
            shutil.rmtree(f'{self.model_dir}/model', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/markers', ignore_errors=True)

            self.train_model(images_folder, layers_activations, layers_decode, decoders, batchsize)

            #for spx in self.final_spx_list:
            #    scale_path = os.path.join(self.model_dir, 'scales', str(spx))
            #    for layer in range(0,self.arch['nlayers']+1):
            #        shutil.rmtree(f'{scale_path}/layer{layer}', ignore_errors=True)
            #        shutil.rmtree(f'{scale_path}/labels{layer}', ignore_errors=True)
            
            #for layer in range(0,self.arch['nlayers']+1):
            #    shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            #    shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)
            #shutil.rmtree(f'{self.model_dir}/scales', ignore_errors=True)

        if process == 'extract':
            print("Graphs paths src:", graphs_path_src)
            print("SPX labels paths src:", spx_label_path_src)
            self.functions.extract_multiscale(images_folder, 
                                   graphs_path_src, spx_label_path_src,
                                   self.model_dir, self.arch_file, self.arch['nlayers'], batchsize, 
                                   self.stride, self.attention, self.aggregation, self.dropout,
                                   save_intermediate, 
                                   layers_activations, 
                                   layers_decode,
                                   decoders,
                                   self.merge_option,
                                   self.final_spx_list,
                                   os.path.join(self.model_dir, 'model', 'kernels_map.txt'))

            # decode the output features, write the activations, and remove the layer and label directories to free up space
            #for layer in range(1,self.arch['nlayers']+1):
            #    if decode_extract:
            #        for d in decoders:
            #            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride, batchsize)

            #    if activation_extract:
            #        self.functions.write_activations(self.model_dir, layer, batchsize)

                #shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                #shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)


def getModelPath(ds_name, split_id, markers_name, arch_name, 
                 stride, spx_alg, final_spx : list, graph_feats, 
                 attention, aggregation, dropout, init_seeds=0, model_name='gflim-BoFP', merge_option = 'dotprod'):
    tmp = 'kmeans'
    if dropout:
        tmp += '-dropout'

    spx = ''
    if model_name == 'gflim-BoFP':

        if len(final_spx) != 1:
            raise ValueError("final_spx must be a single integer when model_name is 'gflim-BoFP'.")
        
        if spx_alg == 'DISF':
            spx += f'{init_seeds}-'
            if init_seeds < 1:
                raise ValueError("init_seeds must be positive when spx_alg is 'DISF'.")
        spx += f'{final_spx[0]}'

    elif model_name == 'gflim-BoFP-multiscale':
        if len(final_spx) < 2:
            raise ValueError("final_spx must be a non-empty list when model_name is 'gflim-BoFP-multiscale'.")
        if spx_alg != 'SH':
            raise ValueError("spx_alg must be 'SH' when model_name is 'gflim-BoFP-multiscale'.")
        spx += '-'.join([str(f) for f in final_spx])
        model_name += f'-{merge_option}'
    else:
        raise ValueError("model_name must be either 'gflim-BoFP' or 'gflim-BoFP-multiscale'.")  

    return f'/home/isabela/Codes/LibIFT/Results/{ds_name}/split{split_id}/{markers_name}/{arch_name}/{model_name}/{stride}/{spx_alg}/{spx}/{tmp}/{graph_feats}-{attention}-{aggregation}'


def GFLIMModelExample():
    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/schisto'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'DISF'
    INIT_SEEDS = 20000
    FINAL_SPX = 10000
    GRAPH_FEATS = 'SUM'
    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'interactive_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    for ARCH_NAME in ['arch_schisto_noskip',
                      #'arch_schisto_dotprod_1', 'arch_schisto_dotprod_2', 
                      #'arch_schisto_dotprod_sumscale0_1', 
                      #'arch_schisto_dotprod_sumscale0_2', 
                      #'arch_schisto_dotprod_sumscalei_1', 'arch_schisto_dotprod_sumscalei_2',
                      #'arch_schisto_sum_1', 
                      #'arch_schisto_sum_2'
                      ]:

        for SPLIT_ID in [1]: #range(1, 6):
            print(f'Running GFLIMModelExample with ARCH_NAME={ARCH_NAME} and SPLIT_ID={SPLIT_ID}')
            model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                                    STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                    ATTENTION, AGGREGATION, DROPOUT, INIT_SEEDS)
            
            model = GFLIMModel(model_dir, 
                            f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                            f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                            SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                            STRIDE, ATTENTION, AGGREGATION, DROPOUT)

            images_folder = f'{DATASET_PATH}/interactive_splits/split{SPLIT_ID}/orig/train'
            model.run_model(images_folder, 'train', save_intermediate=False, layers_activations=[1,2,3], 
                            layers_decode=[1,2,3], decoders=['hybrid', 'graph-adaptive'], batchsize=64,
                            graphs_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/graphs',
                            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/labels')

            images_folder = f'{DATASET_PATH}/interactive_splits/split{SPLIT_ID}/orig/val'
            model.run_model(images_folder, 'extract', save_intermediate=False, layers_activations=[], 
                            layers_decode=[1,2,3], decoders=['hybrid', 'graph-adaptive'], batchsize=64,
                            graphs_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/graphs',
                            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{GRAPH_FEATS}/labels')

def GFLIMMultiScaleUModelExample():
    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/schisto'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'SH'
    GRAPH_FEATS = 'SUM'
    ARCH_NAME = 'arch_schisto_noskip'
    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'interactive_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    FINAL_SPX = [10000,8000,6000]

    for SPLIT_ID in [1]: #range(1, 6):
        print(f'Running GFLIMModelExample with ARCH_NAME={ARCH_NAME} and SPLIT_ID={SPLIT_ID}')
        model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                                STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                ATTENTION, AGGREGATION, DROPOUT, model_name='gflim-BoFP-multiscale')
        
        model = GFLIMultiScaleUModel(model_dir, 
                                    f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                                    f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                                    SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                    STRIDE, ATTENTION, AGGREGATION, DROPOUT)
                                        
        images_folder = f'{DATASET_PATH}/interactive_splits/split{SPLIT_ID}/orig/train'
        
        graphs_dir, spx_dir = [], []
        for spx in FINAL_SPX:
            graphs_dir.append(f'{DATASET_PATH}/superpixels/{SPX_ALG}/{spx}/{GRAPH_FEATS}/graphs')
            spx_dir.append(f'{DATASET_PATH}/superpixels/{SPX_ALG}/{spx}/{GRAPH_FEATS}/labels')

        model.run_model(images_folder, 'train', save_intermediate=False, layers_activations=[1,2,3], 
                        layers_decode=[1,2,3], decoders=['hybrid', 'graph-adaptive'], batchsize=64,
                        graphs_path_src=graphs_dir,
                        spx_label_path_src=spx_dir)


def test():
    def get_kernel_linei(k, l, kernels_maps):
        # find which kernel is k in line (or scale) l
        i = 0
        while(i < l):
            k = kernels_maps[i][k]
            i+=1
        return k

    rows_to_keep = [0, 2, 4]

    # contains the mapping from kernels in scale 0 to kernels in each scale i
    lines = ["1 2 3 2 1 6 7 3 9 10\n", "1 2 3 2 1 6 7 3 3 7\n", "1 1 3 1 1 6 1 3 3 7\n", "1 1 1 1 1 6 1 1 1 1\n", "1 1 1 1 1 1 1 1 1 1\n"]

    # process line 0
    ki = 0
    kernels_map = []
    for i,k0 in enumerate(lines[0].strip().split(' ')):
        k0 = int(k0) - 1
        if(k0 == i):
            kernels_map.append(ki) # new kernel
            ki += 1
        else:
            kernels_map.append(kernels_map[k0]) # repeated kernel

    kernels_maps = [kernels_map]
    print(kernels_maps)

    # process each line i
    line_prev = lines[0].strip().split(' ')
    for l, line in enumerate(lines[1:]):
        line = line.strip().split(' ')
        kernels_map = []
        ki = 0
        for i,k0 in enumerate(line):
            k0 = int(k0) - 1 # i-th kernel in scale i indexed as scale 0
            # (e.g., i=5 k0=3 means kernels 3 and 5 in scale 0 are the same in scale i)

            k0_prev = int(line_prev[i]) - 1 # i-th kernel in scale i-1 indexed as scale 0
            # (e.g., i=5 k0_prev=4 means kernels 4 and 5 in scale 0 are the same in scale i-1. 
            # Considering the previous example, kernels 3, 4, and 5 in scale 0 are the same in scale i)

            if(k0_prev == i):
                if(k0 == k0_prev):
                    kernels_map.append(ki)
                    ki+=1
                else: # repeated (merged) kernel. Map from scale i-1 to scale i
                    k = get_kernel_linei(k0, l+1, kernels_maps) # map k0 from scale 0 to scale i-1
                    k = kernels_map[k]
                    kernels_map.append(k)

        line_prev = line.copy()
        kernels_maps.append(kernels_map)

    print('kernels_maps:', kernels_maps)

    # erase scales
    kernels_map_final = [kernels_maps[rows_to_keep[0]].copy()]
    for id,r in enumerate(rows_to_keep[:-1]):
        s = rows_to_keep[id+1]
        r+=1
        mapped_kernels = kernels_maps[r]
        r+=1
        while(r <= s):
            mapped_kernels = [kernels_maps[r][k] for k in mapped_kernels]
            r+=1
        kernels_map_final.append(mapped_kernels)
    print('kernels_map_final:', kernels_map_final)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train and extract GFLIM-BoFP model.")

    # madatory arguments
    parser.add_argument("dataset_path", help="Path to the dataset.", type=str) # example: /home/isabela/Codes/LibIFT/Datasets/schisto
    parser.add_argument("final_spx", help="Final number of superpixels (it can be a list in the form 1,2,3).", type=list_of_ints) # example: 10000
    parser.add_argument("markers_name", help="Name of the markers.", type=str) # example: interactive_markers
    parser.add_argument("arch_name", help="Architecture name.", type=str)
    """
    arch examples : 'arch_schisto_noskip',
        'arch_schisto_dotprod_1', 'arch_schisto_dotprod_2', 
        'arch_schisto_dotprod_sumscale0_1', 'arch_schisto_dotprod_sumscale0_2', 
        'arch_schisto_dotprod_sumscalei_1', 'arch_schisto_dotprod_sumscalei_2',
        'arch_schisto_sum_1', 'arch_schisto_sum_2'
    """
    parser.add_argument("train_val", help="Whether to train or validate the model.", type=str, choices=['train', 'extract', 'both'])
    parser.add_argument("layers_activations", help="Layers activations to use.", type=list_of_ints, default=[1,2,3])
    parser.add_argument("layers_decode", help="Layers to decode.", type=list_of_ints, default=[1,2,3])
    
    # required when spx_alg is 'DISF'
    parser.add_argument("--init_seeds", help="Initial number of seeds.", type=int, default=20000) 

    # optional arguments
    parser.add_argument("--merge_option", help="Merge option to use with multi-scale GFLIM.", type=str, default='dotprod', choices=['dotprod', 'dotprod_sumscale0', 'dotprod_sumscalei', 'sum'])
    parser.add_argument("--spx_alg", help="Superpixel algorithm to use.", type=str, default='DISF', choices=['DISF', 'ERGC','SH'])
    parser.add_argument("--graph_feats", help="Graph features to use.", type=str, default='SUM', choices=['SUM', 'MEAN'])
    parser.add_argument("--aggregation", help="Aggregation method to use.", type=str, default='CONCAT', choices=['CONCAT', 'WCONCAT'])
    parser.add_argument("--stride", help="Stride type to use.", type=str, default='graph-based')
    parser.add_argument("--attention", help="Attention mechanism to use.", type=str, default='NONE', choices=['NONE', 'CLASS_DOMINANCE'])
    parser.add_argument("--dropout", help="Dropout rate to use.", type=int, default=0, choices=[0, 1])
    parser.add_argument("--batch_size", help="Batch size for training.", type=int, default=64)
    parser.add_argument("--split_id", help="Split ID for training.", type=int, default=1)
    parser.add_argument("--decoders", help="Decoders to use.", type=list_of_str, default=['hybrid', 'graph-adaptive'])
    parser.add_argument("--save_intermediate", help="Whether to save intermediate results.", type=int, default=0, choices=[0, 1])
    
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = args.spx_alg
    INIT_SEEDS = args.init_seeds
    FINAL_SPX = args.final_spx
    GRAPH_FEATS = args.graph_feats
    AGGREGATION = args.aggregation
    MARKERS_NAME = args.markers_name
    STRIDE = args.stride
    ATTENTION = args.attention
    DROPOUT = args.dropout
    ARCH_NAME = args.arch_name
    BATCH_SIZE = args.batch_size
    SPLIT_ID = args.split_id
    TRAIN_VAL = args.train_val
    SAVE_INTERMEDIATE = args.save_intermediate
    LAYERS_ACTIVATIONS = args.layers_activations
    LAYERS_DECODE = args.layers_decode
    DECODERS = args.decoders

    if SPX_ALG != 'SH' and len(FINAL_SPX) > 1:
        print("Superpixel algorithm must be SH when FINAL_SPX contains more than 1 element.")
        exit(1)

    print(f'Running GFLIMModelExample with ARCH_NAME={ARCH_NAME} and SPLIT_ID={SPLIT_ID}')
    
    
    graphs_path_src, spx_label_path_src = None, None

    if len(FINAL_SPX) == 1:
        model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                             STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                             ATTENTION, AGGREGATION, DROPOUT, INIT_SEEDS)
        model = GFLIMModel(model_dir, f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                       f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                       SPX_ALG, INIT_SEEDS, FINAL_SPX[0], GRAPH_FEATS, 
                       STRIDE, ATTENTION, AGGREGATION, DROPOUT)
        
        if SPX_ALG == 'DISF':
            graphs_path_src = f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX[0]}/{GRAPH_FEATS}/graphs'
            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX[0]}/{GRAPH_FEATS}/labels'
        else:
            graphs_path_src = f'{DATASET_PATH}/superpixels/{SPX_ALG}/{FINAL_SPX[0]}/{GRAPH_FEATS}/graphs'
            spx_label_path_src=f'{DATASET_PATH}/superpixels/{SPX_ALG}/{FINAL_SPX[0]}/{GRAPH_FEATS}/labels'

    else:
        if len(FINAL_SPX) < 2:
            raise ValueError("FINAL_SPX must contain at least 2 elements when model_name is 'gflim-BoFP-multiscale'.")

        model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME,
                                STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS,
                                ATTENTION, AGGREGATION, DROPOUT, model_name='gflim-BoFP-multiscale', merge_option=args.merge_option)
        model = GFLIMultiScaleUModel(model_dir, f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                       f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                       SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                       STRIDE, ATTENTION, AGGREGATION, DROPOUT, merge_option=args.merge_option)
        graphs_path_src, spx_label_path_src = [], []
        for spx in FINAL_SPX:
            graphs_path_src.append(f'{DATASET_PATH}/superpixels/{SPX_ALG}/{spx}/{GRAPH_FEATS}/graphs')
            spx_label_path_src.append(f'{DATASET_PATH}/superpixels/{SPX_ALG}/{spx}/{GRAPH_FEATS}/labels')

    if args.train_val == 'both' or args.train_val == 'train':
        images_folder = f'{DATASET_PATH}/interactive_splits/split{SPLIT_ID}/orig/train'
        model.run_model(images_folder, 'train', save_intermediate=False, layers_activations=LAYERS_ACTIVATIONS,
                    layers_decode=LAYERS_DECODE, decoders=DECODERS, batchsize=BATCH_SIZE,
                    graphs_path_src=graphs_path_src,
                    spx_label_path_src=spx_label_path_src)

    if args.train_val == 'both' or args.train_val == 'extract':
        images_folder = f'{DATASET_PATH}/interactive_splits/split{SPLIT_ID}/orig/val'
        model.run_model(images_folder, 'extract', save_intermediate=False, layers_activations=[],
                    layers_decode=LAYERS_DECODE, decoders=DECODERS, batchsize=BATCH_SIZE,
                    graphs_path_src=graphs_path_src,
                    spx_label_path_src=spx_label_path_src)