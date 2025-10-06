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

    def merge_scales_dotProd(self, thresh=False, std=False):
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

    def merge_scales_dotProd_sum_scale0(self, thresh=False, std=False):
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

    def merge_scales_dotProd_sum_scalei(self, thresh=False, std=False):
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
            'CreateGraphFromLabels': '{}/iftCreateGFLIMGraphFromLabels'.format(path)
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
        os.system("{} {}".format(function, " ".join([str(i) for i in args])))

    def create_graphs(self, dataset_path, final_spx, graphs_path_src, graph_feats, spx_alg, init_seeds=0):
        """
        Create the graphs for the dataset if they do not exist.
        Args:
            dataset_path (str): images directory
            final_spx (int): Number of superpixels.
            graphs_path_src (str): Path where the graphs and labels will be stored.
            graph_feats (str): Type of graph features to use.
            spx_alg (str): Superpixel algorithm to use.
            init_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
        """
        if spx_alg in ['DISF', 'ERGC']:
            self.__callFunction__(self.functions['CreateGraph'], [f'{dataset_path}/imgs/orig', final_spx, graphs_path_src, graph_feats, spx_alg, init_seeds], 'Creating graphs')
        else:
            self.__callFunction__(self.functions['CreateSpxImageSH'], [f'--img {dataset_path}/imgs/orig', f'--save {graphs_path_src}', f'--node_feat {graph_feats}', f'--k {",".join([str(spx) for spx in final_spx])}'], 'Creating superpixel images')
            for spx in final_spx:
                self.__callFunction__(self.functions['CreateGraphFromLabels'], [f'{dataset_path}/imgs/orig', f'{graphs_path_src}/{spx}/{graph_feats}/labels', graph_feats, f'{graphs_path_src}/{spx}/{graph_feats}/graphs'], 'Creating graphs from superpixel images')

    def create_layer_model(self, model_dir, arch_file, layer, stride, attention, aggregation):
        """
        Create the layer model for the given architecture.
        Args:
            model_dir (str): Directory where the model will be stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN)
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
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN)
        """
        self.__callFunction__(self.functions['EncodeLayer'], [model_dir, arch_file, layer, stride, attention, aggregation], "Encoding layer {}:\t\t".format(layer))

    def write_activations(self, model_dir, layer, skip=None):
        """
        Write the activations for the given layer.
        Args:
            model_dir (str): Directory where the model is stored.
            layer (int): Layer number.
        """
        if skip == None:
            self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer], "Writing activations layer {}:\t".format(layer))
        else:
            self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer, skip], "Writing activations layer {} with skip connection:\t".format(layer))

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
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN)
            dropout (bool): Whether to apply dropout.
        """
        self.__callFunction__(self.functions['EncodeMergedLayer'], [model_dir, arch_file, layer, stride, attention, aggregation, 1 if dropout else 0], "Encoding Merged layer {}:\t\t".format(layer))

    def decode_layer(self, model_dir, arch_file, layer, d, stride):
        """
        Decode the layer for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
            d (str): Decoding type ('hybrid', 'graph-adaptive', 'gaussian', 'test', 'tristate').
            stride (str): Stride type ('graph-based', 'none', or 'image-based').
        """
        self.__callFunction__(self.functions['DecodeLayer'], [model_dir, layer, arch_file, d, stride], "Decoding layer {}:\t\t".format(layer))

    def convert_seeds(self, model_dir, markers_dir, nfeat_markers, kernel_size, attention, aggregation):
        """
        Convert the seeds for the given architecture.
        Args:
            model_dir (str): Directory where the model is stored.
            markers_dir (str): Directory where the image markers are stored.
            nfeat_markers (int): Number of feature points per marker (if -1, do not perform k-means).
            kernel_size (int): Kernel size.
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN)
        """
        self.__callFunction__(self.functions['ConvertSeeds'], [model_dir, markers_dir, nfeat_markers, kernel_size, attention, aggregation], 'Converting seeds:\t\t')

    def extract(self, img_dir, graphs_dir, spx_dir, model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, dropout):
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
            aggregation (str): Aggregation strategy (CONCAT, MEAN, WMEAN)
            dropout (bool): Whether to apply dropout.

        """
        self.__callFunction__(self.functions['Extract'], [img_dir, graphs_dir, spx_dir, model_dir, arch_file, nlayers, batch_size, stride, attention, aggregation, 1 if dropout else 0], 'Extracting features:\t\t')

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


class GFLIMModel():
    """
    A class to hold the GFLIM model and its parameters.
    1. Create the graphs if they do not exist.
    2. Train the model.
    3. Extract features from the model.
    """
    def __init__(self, dataset_path, model_dir, arch_file, markers_dir, split_id,
                 spx_alg, init_seeds, final_spx, graph_feats, 
                 stride, attention, aggregation, dropout):
        """
        Initialize the GFLIM model with the given parameters.
        Args:
            dataset_path (str): Path to the dataset.
            model_dir (str): Directory where the model will be stored.
            arch_file (str): Path to the architecture file.
            markers_dir (str): Directory where the image markers are stored.
            split_id (int): Split ID for training and validation.
            spx_alg (str): Superpixel algorithm to use ('DISF', 'ERGC', 'SLIC', 'LSC', 'SH').
            init_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
            final_spx (int or list): Final number of superpixels. If spx_alg is 'DISF' or 'ERGC', this is an integer. If spx_alg is 'SLIC', 'LSC', or 'SH', this is a list of integers.
            graph_feats (str): Type of graph features to use ('RGB', 'RGBXY', 'HSV', 'HSVI', 'LAB', 'LABI').
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy ('CONCAT', 'MEAN', 'WMEAN').
            dropout (bool): Whether to apply dropout.
        """

        self.dataset_path = dataset_path
        self.ds_name = self.dataset_path.split('/')[-1]
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
        self.split_id = split_id

        self.functions = GFLIMFunctions('/home/isabela/Codes/LibIFT/bin')

        self.graphs_path_src = os.path.join(self.dataset_path, 'superpixels', self.spx_alg, f'{self.init_seeds}-{self.final_spx}' if self.spx_alg == 'DISF' else f'{self.final_spx}', self.graph_feats)
        if not os.path.exists(self.graphs_path_src):
            self.functions.create_graphs(self.dataset_path, self.final_spx, self.graphs_path_src, self.graph_feats, self.spx_alg, self.init_seeds)

        self.arch = self.read_architecture(self.arch_file)

    def __train_layer__(self, layer, write_activation, decoders=[]):
        """
        Train a single layer of the model.
        Args:
            layer (int): Layer number.
            write_activation (bool): Whether to write the activations.
            decoders (list): List of decoders to use.
        """
        self.functions.create_layer_model(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        
        if write_activation:
            self.functions.encode_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
            self.functions.write_activations(self.model_dir, layer)

        self.functions.merge_layer_models(self.model_dir, layer)
        self.functions.encode_merged_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, self.dropout)
        
        if write_activation:
            self.functions.write_activations(self.model_dir, layer)

        for d in decoders: #["hybrid", "graph-adaptive"]:#, "test", "tristate"]:
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def __copyGraphsAndLabels__(self, src_dir):
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

            graph_src_file = os.path.join(self.graphs_path_src, 'graphs', f'{graph_name}.json')
            if not os.path.exists(graph_src_file):
                print(f"Graph source file {graph_src_file} does not exist.")
                exit()
            if not os.path.isfile(os.path.join(self.model_dir, 'layer0', f'{graph_name}.json')):
                os.symlink(graph_src_file, os.path.join(self.model_dir, 'layer0', f'{graph_name}.json'))

            labels_src_file = os.path.join(self.graphs_path_src, 'labels', f'{graph_name}.pgm')
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
        
        scale, parent_scale = layer, -1
        while(parent_scale != scale):
            parent_scale = scale
            for i, skip in enumerate(self.arch[f'layer{scale}']['skip_connection']):
                if skip > 0:
                    scale = i + 1
                    break
        return scale

    def skip_connection(self, layer, first_scale, merge_function, write_activation):
        """
        Merge the layers with skip connections.
        Args:
            layer (int): Layer number.
            first_scale (int): The first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            write_activation (bool): Whether to write the activations.
        Steps:
            1. Copy the graphs and labels of the layer to a new directory.
            2. Compute the kernels maps for the current layer.
            3. For each graph in the layer:
                a. Create a MergeLayers object.
                b. For each skip connection in the layer:
                    i. Add the scale to the MergeLayers object.
                c. Add the current layer to the MergeLayers object.
                d. Merge the scales using the dot product or sum.
                e. Convert the merged features back to a graph.
                f. Save the merged graph.
            4. Replace the labels of the layer with the labels of the first scale.

        Note: the current layer is always the last scale to be merged
        """

        """
        Suponha layer = 9, first scale 2 e len(kernel maps) = 12
        Skips on layer 4 : 2
        Skips on layer 7 : 5 e 6
        Skips on layer 9 : 4 e 7

        kernel_maps = [(1->2), (2->3), (3->4), (4->5), (5->6), (6->7), (7->8), (8->9), (9->10), (10->11), (11->12), (12->13)]

        Merge kernel maps according to skips:
            Layer 4: calcular kernel_maps[0] = (range(kernel_maps[3])) 
                                                i.e, if first_scale == first_scalei or first_scalei == 1, range(kernel_maps[first_scalei-1])
            Layer 7: calcular kernel_maps[1] = (2->5) i.e, (first_scale-1, first_scalei-2)
            Layer 9: calcular kernel_maps[2] = (2->9) i.e, (first_scale-1, layer-2)
        
        """
        
        # Copy the graphs and labels of the layer to a new directory.
        shutil.copytree(os.path.join(self.model_dir, f'layer{layer}'), os.path.join(self.model_dir, f'layer{layer}_noSkip'), dirs_exist_ok=True)

        skip_layers = [i for i, skip in enumerate(self.arch[f'layer{layer}']['skip_connection']) if skip > 0]

        # Compute the kernels maps for the current layer.
        kernel_maps_all = compute_kernels_maps(self.model_dir)
        kernel_maps = []
        for i in skip_layers: # i is layer - 1

            # Since these data (to be merged) was already processed and they may have their own skip connections,
            # we need to ensure that the kernels maps are correctly set to map to the first scale.
            # Note that since these scales are already merged (if they have skip connections), there is NO need to update the nodes map too.
            first_scale_i = self.get_first_scale(i+1)
            
            kernel_maps.append(get_merged_scales_from_kernels_map(kernel_maps_all, first_scale-1, first_scale_i-1))

        kernel_maps.append(get_merged_scales_from_kernels_map(kernel_maps_all, first_scale-1, layer-1))

        # Add data from scales 0 to i-1. Process each image in the layer
        for fileName in os.listdir(os.path.join(self.model_dir, f'layer{layer}')):
            basename = fileName.split('.')[0]
            mergeFunction = MergeLayers(self.model_dir)
            mergeFunction.kernels_maps = kernel_maps
            
            # add data from scales specified in skip connections
            for i in skip_layers:
                graph_path = os.path.join(self.model_dir, f'layer{i+1}', fileName)
                spx_path = os.path.join(self.model_dir, f'labels{i+1}', f'{basename}.pgm')
                mergeFunction.add_scale(spx_path, graph_path)                

            # Add the data from the current layer and merge them all
            # (current layer is always the last scale to be merged)
            graph_path = os.path.join(self.model_dir, f'layer{layer}', fileName)
            spx_path = os.path.join(self.model_dir, f'labels{layer}', f'{basename}.pgm')
            mergeFunction.add_scale(spx_path, graph_path)

            mergeFunction.compute_node_maps()

            if merge_function == 'dotProd':
                merged_features = mergeFunction.merge_scales_dotProd(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif merge_function == 'sum':
                merged_features = mergeFunction.merge_scales_sum(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif merge_function == 'dotProd_sum_scale0':
                merged_features = mergeFunction.merge_scales_dotProd_sum_scale0(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif merge_function == 'dotProd_sum_scalei':
                merged_features = mergeFunction.merge_scales_dotProd_sum_scalei(thresh=False, std=False) # 'merged_features' is a feature matrix
            else:
                raise ValueError(f"Unknown merge function: {merge_function}")

            # convert the merged feature matrix back to graph and save it
            graph_path = os.path.join(self.model_dir, f'layer{first_scale}', fileName)
            graph = mergeFunction.read_graph(graph_path)
            for node in range(graph['num_nodes']):
                graph[f'node{node}']['feats'] = merged_features[node,:].tolist()
            with open(os.path.join(self.model_dir, f'layer{layer}', fileName), 'w') as f:
                json.dump(graph, f, indent=4)
        
        # replace the labels with the ones from the first scale
        shutil.copytree(os.path.join(self.model_dir, f'labels{layer}'), os.path.join(self.model_dir, f'labels{layer}_noSkip'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(self.model_dir, f'labels{first_scale}'), os.path.join(self.model_dir, f'labels{layer}'), dirs_exist_ok=True)

        # write activations of the merged data
        if write_activation:
            shutil.copytree(os.path.join(self.model_dir, f'activation{layer}'), os.path.join(self.model_dir, f'activation{layer}_noSkip'), dirs_exist_ok=True)
            shutil.rmtree(os.path.join(self.model_dir, f'activation{layer}'))
            self.functions.write_activations(self.model_dir, layer, first_scale)

    def train_model(self, img_splits_names, decoders, merge_function, write_activation):
        """
        Train the GFLIM model.
        Args:
            img_splits_names (str): Name of the image splits directory.
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
            self.__train_layer__(layer, write_activation, decoders)

            if layer > 1:
                self.functions.write_kernels_correspondence(os.path.join(self.model_dir, f'markers'),
                                                                os.path.join(self.model_dir, 'labels0'),
                                                                os.path.join(self.model_dir, f'labels{layer-1}'),
                                                                os.path.join(self.model_dir, 'model', 'kernels_map.txt'))

            # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            first_scale = self.get_first_scale(layer)

            # merge layers output
            if first_scale != layer: # if there are skip connections
                self.skip_connection(layer, first_scale, merge_function, write_activation)


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

    def run_model(self, img_splits_names, train=True, decode_train=True, activation_train=True,
                           extract=True, decode_extract=True, activation_extract=True,
                           decoders=["hybrid", "graph-adaptive"],
                           merge_function='dotProd_sum_scalei'):
        """
        Run (train and/or extract) a GFLIM model.
        Args:
            img_splits_names (str): Name of the image splits directory.
            train (bool): Whether to train the model.
            decode_train (bool): Whether to decode the training set.
            activation_train (bool): Whether to write the activations for the training set.
            extract (bool): Whether to extract features from the model.
            decode_extract (bool): Whether to decode the extracted features.
            activation_extract (bool): Whether to write the activations for the extracted features.
            decoders (list): List of decoders to use.
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
        
        if not train and not extract:
            print("Nothing to do. Please set train and/or extract to True.")
            return

        # remove any existing layer and label directories in the model directory to do not mix train and extract data
        for layer in range(0,self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if train:
            # copy the train graphs and superpixel labels from the dataset directory to the model directory
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train')

            self.train_model(img_splits_names, decoders if decode_train else [], merge_function=merge_function, write_activation=activation_train)

            # remove the layer and label directories to free up space and allow the extraction to be done from scratch
            #for layer in range(0,self.arch['nlayers']+1):
            #    shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            #    shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if extract:
            # copy the graphs and superpixel labels from the dataset directory to the model directory
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/imgs/orig')

            self.functions.extract(f'{self.dataset_path}/imgs/orig', f'{self.graphs_path_src}/graphs', f'{self.graphs_path_src}/labels', self.model_dir, self.arch_file, self.arch['nlayers'], 2, self.stride, self.attention, self.aggregation, self.dropout)

            # decode the output features, write the activations, and remove the layer and label directories to free up space
            for layer in range(1,self.arch['nlayers']+1):
                if decode_extract:
                    for d in decoders:
                        self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

                if activation_extract:
                    self.functions.write_activations(self.model_dir, layer)

                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)


class GFLIMultiScaleUModel(GFLIMModel):
    def __init__(self, dataset_path : str, model_dir : str, arch_file : str, markers_dir : str, split_id : int,
                 spx_alg : str, init_seeds : int, final_spx_list : list, graph_feats : str, 
                 stride : str, attention : str, aggregation : str, dropout : bool):

        self.dataset_path = dataset_path
        self.ds_name = self.dataset_path.split('/')[-1]
        self.spx_alg = spx_alg
        self.init_seeds = init_seeds
        self.final_spx_list = final_spx_list
        self.graph_feats = graph_feats
        self.attention = attention
        self.aggregation = aggregation
        self.dropout = dropout
        self.arch_file = arch_file
        self.stride = stride
        self.markers_dir = markers_dir
        self.model_dir = model_dir
        self.split_id = split_id

        self.functions = GFLIMFunctions('/home/isabela/Codes/LibIFT/bin')

        self.graphs_path_src = []
        for spx in self.final_spx_list:
            graphs_path_src = os.path.join(self.dataset_path, 'superpixels', self.spx_alg, f'{self.init_seeds}-{spx}' if self.spx_alg == 'DISF' else f'{spx}', self.graph_feats)
        #    if not os.path.exists(graphs_path_src):
        #        raise ValueError(f"Graphs path source {graphs_path_src} does not exist. Please create the graphs first.")
            self.graphs_path_src.append(graphs_path_src)

        self.arch = self.read_architecture(self.arch_file)


    def __copyGraphsAndLabels__(self, src_dir, index=0):

        if index >= len(self.final_spx_list):
            raise ValueError(f"Index {index} out of range for final_spx_list with length {len(self.final_spx_list)}.")

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

            graph_src_file = os.path.join(self.graphs_path_src[index], 'graphs', f'{graph_name}.json')
            if not os.path.exists(graph_src_file):
                print(f"Graph source file {graph_src_file} does not exist.")
                exit()
            if not os.path.isfile(os.path.join(self.model_dir, 'layer0', f'{graph_name}.json')):
                os.symlink(graph_src_file, os.path.join(self.model_dir, 'layer0', f'{graph_name}.json'))

            # só copia os labels da primeira escala devido aos marcadores serem os mesmos
            labels_src_file = os.path.join(self.graphs_path_src[index], 'labels', f'{graph_name}.pgm')
            if not os.path.exists(labels_src_file):
                print(f"Labels source file {labels_src_file} does not exist.")
                exit()
            if not os.path.isfile(os.path.join(self.model_dir, 'labels0', f'{graph_name}.pgm')):
                os.symlink(labels_src_file, os.path.join(self.model_dir, 'labels0', f'{graph_name}.pgm'))

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


    def __train_layer__(self, layer, write_activation, decoders=[]):
        self.functions.create_layer_model(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        self.functions.encode_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        
        if write_activation:
            self.functions.write_activations(self.model_dir, layer)

        self.functions.merge_layer_models(self.model_dir, layer)
        self.functions.encode_merged_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, self.dropout)
        
        if write_activation:
            self.functions.write_activations(self.model_dir, layer)

        for d in decoders: #["hybrid", "graph-adaptive"]:#, "test", "tristate"]:
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def train_model(self, img_splits_names, decoders, merge_function, write_activation):
        # copy arch file
        file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
        if os.path.isfile(file): # if it exists, remove it
            os.remove(file)
        shutil.copy(self.arch_file, self.model_dir)

        spx0 = self.final_spx_list[0]
        self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train', 0)
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch[f'layer1']['conv']["nkernels_per_marker"], self.arch[f'layer1']['conv']['kernel_size'], self.attention, self.aggregation)
        for layer in range(1, self.arch['nlayers'] + 1):
            self.__train_layer__(layer, write_activation, decoders)
            self.rename_files_model(layer, spx0)
            for i, spx in enumerate(self.final_spx_list[1:]):
                if layer == 1:
                    self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train', i+1)
                    self.functions.write_kernels_correspondence(os.path.join(self.model_dir, f'markers_spx{spx0}'),
                                                                os.path.join(self.graphs_path_src[0], 'labels'),
                                                                os.path.join(self.graphs_path_src[i+1], 'labels'),
                                                                os.path.join(self.model_dir, 'model', f'kernels_map.txt'),
                                                                os.path.join(self.model_dir, f'markers'))
                self.__train_layer__(layer, write_activation, decoders)
                self.rename_files_model(layer, spx)


    def rename_files_model(self, layer, spx):
        for train_graph_name in os.listdir(f'{self.model_dir}/layer{layer}'):
            basename = os.path.join(self.model_dir, 'model', train_graph_name.split('.')[0])
            for post_name in [f'-conv{layer}-bias.txt', f'-conv{layer}-kernels.npy', f'-conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'{basename}{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'{basename}-spx{spx}{post_name}')
                    os.rename(src_filename, dst_filename)

        for post_name in [f'conv{layer}-bias.txt', f'conv{layer}-kernels.npy', f'conv{layer}-labels.txt']:
                src_filename = os.path.join(self.model_dir, 'model', f'{post_name}')
                if os.path.isfile(src_filename):
                    dst_filename = os.path.join(self.model_dir, 'model', f'spx{spx}{post_name}')
                    os.rename(src_filename, dst_filename)

        for folder in [f'layer{layer}', f'labels{layer}', f'activation{layer}', f'activationNoMerge{layer}', 'markers']:
            src_path = os.path.join(self.model_dir, folder)
            if os.path.exists(src_path):
                shutil.copytree(src_path, os.path.join(self.model_dir, f'{folder}_spx{spx}'), dirs_exist_ok=True)
                shutil.rmtree(src_path, ignore_errors=True)


    def run_model(self, img_splits_names, train=True, decode_train=True, activation_train=True,
                           extract=True, decode_extract=True, activation_extract=True,
                           decoders=["hybrid", "graph-adaptive"], merge_function='dotProd_sum'):
        
        if not train and not extract:
            print("Nothing to do. Please set train and/or extract to True.")
            return

        for layer in range(0,self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        # if there are no superpixel images and graphs in the dataset path, create them
        create = False
        for spx in self.final_spx_list:
            graphs_path_src = os.path.join(self.dataset_path, 'superpixels', self.spx_alg, f'{spx}', self.graph_feats, 'labels')
            if not os.path.exists(graphs_path_src) or len(os.listdir(graphs_path_src)) < len(os.listdir(f'{self.dataset_path}/imgs/orig')):
                create = True
                break
        
        if create:
            graphs_path_src = os.path.join(self.dataset_path, 'superpixels', self.spx_alg)
            self.functions.create_graphs(self.dataset_path, self.final_spx_list, graphs_path_src, self.graph_feats, self.spx_alg, self.init_seeds)

        if train:
            self.train_model(img_splits_names, decoders if decode_train else [], write_activation=activation_train)

            for layer in range(0,self.arch['nlayers']+1):
                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if extract:
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/imgs/orig')
            self.functions.extract(f'{self.dataset_path}/imgs/orig', f'{self.graphs_path_src}/graphs', f'{self.graphs_path_src}/labels', self.model_dir, self.arch_file, self.arch['nlayers'], 2, self.stride, self.attention, self.aggregation, self.dropout)

            for layer in range(1,self.arch['nlayers']+1):
                if decode_extract:
                    for d in decoders:
                        self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

                if activation_extract:
                    self.functions.write_activations(self.model_dir, layer)

                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)


def getModelPath(ds_name, split_id, markers_name, arch_name, 
                 stride, spx_alg, final_spx, graph_feats, 
                 attention, dropout, init_seeds=0, model_name='gflim-BoFP'):
    tmp = 'kmeans'
    if dropout:
        tmp += '-dropout'

    spx = ''
    if model_name == 'gflim-BoFP':
        if spx_alg == 'DISF':
            spx += f'{init_seeds}-'
            if init_seeds < 1:
                raise ValueError("init_seeds must be positive when spx_alg is 'DISF'.")
        spx += f'{final_spx}'
    elif model_name == 'gflim-BoFP-multiscale':
        if not isinstance(final_spx, list) or len(final_spx) == 0:
            raise ValueError("final_spx must be a non-empty list when model_name is 'gflim-BoFP-multiscale'.")
        if spx_alg == 'DISF':
            spx += f'{init_seeds}-'
            if init_seeds < 1:
                raise ValueError("init_seeds must be positive when spx_alg is 'DISF'.")
        spx += '-'.join([str(f) for f in final_spx])

    return f'/home/isabela/Codes/LibIFT/Results/{ds_name}/split{split_id}/{markers_name}/{arch_name}/{model_name}/{stride}/{spx_alg}/{spx}/{tmp}/{graph_feats}-{attention}'


def GFLIMModelExample():
    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/cells_dataset'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'DISF'
    INIT_SEEDS = 50000
    FINAL_SPX = 20000
    GRAPH_FEATS = 'SUM'
    ARCH_NAME = 'arch_cells_test'
    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'interactive_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    for merge_function in ['dotProd', 'sum', 'dotProd_sum_scale0', 'dotProd_sum_scalei']:
        for SPLIT_ID in range(1, 6):
            model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                                    STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                    ATTENTION, DROPOUT, INIT_SEEDS)
            model_dir += f'_merge-{merge_function}'

            model = GFLIMModel(DATASET_PATH, model_dir, 
                                            f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                                            f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                                            SPLIT_ID,
                                            SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                                            STRIDE, ATTENTION, AGGREGATION, DROPOUT)

            model.run_model('interactive_splits', extract=False, merge_function=merge_function)

    ARCH_NAME = 'arch_cells_test2'
    for merge_function in ['dotProd', 'sum', 'dotProd_sum_scale0', 'dotProd_sum_scalei']:
        for SPLIT_ID in range(1, 6):
            model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                                    STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                    ATTENTION, DROPOUT, INIT_SEEDS)
            model_dir += f'_merge-{merge_function}'

            model = GFLIMModel(DATASET_PATH, model_dir, 
                                            f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                                            f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                                            SPLIT_ID,
                                            SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                                            STRIDE, ATTENTION, AGGREGATION, DROPOUT)

            model.run_model('interactive_splits', extract=False, merge_function=merge_function)


def GFLIMMultiScaleUModelExample():
    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/schisto'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'DISF'
    INIT_SEEDS = 10000
    SPX = 8000
    GRAPH_FEATS = 'SUM'
    ARCH_NAME = 'arch_cells_test'

    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'interactive_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    FINAL_SPX = [SPX]
    """arch_file = f'{DATASET_PATH}/archs/{ARCH_NAME}.json'
    with open(arch_file) as f:
        arch = json.load(f)
    FINAL_SPX = [SPX]
    last = SPX
    for l in range(1, arch['nlayers']):
        FINAL_SPX.append(last * arch[f'layer{l}']['pooling']['stride'])
        last = FINAL_SPX[-1]
    """

    for SPLIT_ID in range(1, 6):

        model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                                STRIDE, SPX_ALG, FINAL_SPX, GRAPH_FEATS, 
                                ATTENTION, DROPOUT, model_name='gflim-BoFP-multiscale')
        
        model = GFLIMultiScaleUModel(
        #model = GFLIMModel(
                                        DATASET_PATH, model_dir, 
                                        f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                                        f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                                        SPLIT_ID,
                                        SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                                        STRIDE, ATTENTION, AGGREGATION, DROPOUT)
                                        
        model.run_model('interactive_splits', activation_extract=False, activation_train=True, extract=False, decoders=[])


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
    #test()
    GFLIMModelExample()