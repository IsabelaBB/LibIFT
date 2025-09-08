import os
import json
import sys
import shutil
from PIL import Image
import numpy as np
import time



class GFLIMFunctions():
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

    def write_activations(self, model_dir, layer):
        """
        Write the activations for the given layer.
        Args:
            model_dir (str): Directory where the model is stored.
            layer (int): Layer number.
        """
        self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer], "Writing activations layer {}:\t".format(layer))

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
        self.__callFunction__(self.functions['EncodeMergedLayer'], [model_dir, arch_file, layer, stride, attention, aggregation, 1 if dropout else 0], "Encoding layer {}:\t\t".format(layer))

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
        """
        self.__callFunction__(self.functions['WriteKernelsCorrespondence'], [markers_dir_scale0, spx_dir_scale0, spx_dir_scalei, save_file, save_markers_folder], 'Writing kernels correspondence:\t')

class GFLIMModel():
    def __init__(self, dataset_path, model_dir, arch_file, markers_dir, split_id,
                 spx_alg, init_seeds, final_spx, graph_feats, 
                 stride, attention, aggregation, dropout):

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

        self.kernel_size, self.nfeat_markers, self.nlayers = self.read_architecture(self.arch_file)

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

    def __copyGraphsAndLabels__(self, src_dir):
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

    def train_model(self, img_splits_names, decoders, write_activation):
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.nfeat_markers[0], self.kernel_size[0], self.attention, self.aggregation)
        for layer in range(1, self.nlayers + 1):
            self.__train_layer__(layer, write_activation, decoders)

    def read_architecture(self, arch_file):
        with open(arch_file) as f:
            data = json.load(f)
            nlayers = data['nlayers']
            kernel_size, nfeat_markers = [], []
            for layer in range(1, nlayers + 1):
                kernel_size.append(data[f'layer{layer}']['conv']['kernel_size'][0]*data[f'layer{layer}']['conv']['kernel_size'][1])
                nfeat_markers.append(data[f'layer{layer}']['conv']["nkernels_per_marker"])

        return kernel_size, nfeat_markers, nlayers

    def run_model_parallel(self, img_splits_names, train=True, decode_train=True, activation_train=True,
                           extract=True, decode_extract=True, activation_extract=True,
                           decoders=["hybrid", "graph-adaptive"]):
        
        if not train and not extract:
            print("Nothing to do. Please set train and/or extract to True.")
            return

        for layer in range(0,self.nlayers+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if train:
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train')
            self.train_model(img_splits_names, decoders if decode_train else [], write_activation=activation_train)

            for layer in range(0,self.nlayers+1):
                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if extract:
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/imgs/orig')
            self.functions.extract(f'{self.dataset_path}/imgs/orig', f'{self.graphs_path_src}/graphs', f'{self.graphs_path_src}/labels', self.model_dir, self.arch_file, self.nlayers, 2, self.stride, self.attention, self.aggregation, self.dropout)

            for layer in range(1,self.nlayers+1):
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

        self.kernel_size, self.nfeat_markers, self.nlayers = self.read_architecture(self.arch_file)


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

    def train_model(self, img_splits_names, decoders, write_activation):
        spx0 = self.final_spx_list[0]
        self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train', 0)
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.nfeat_markers[0], self.kernel_size[0], self.attention, self.aggregation)
        for layer in range(1, self.nlayers + 1):
            self.__train_layer__(layer, write_activation, decoders)
            self.rename_files_model(layer, spx0)
            for i, spx in enumerate(self.final_spx_list[1:]):
                if layer == 1:
                    self.__copyGraphsAndLabels__(f'{self.dataset_path}/{img_splits_names}/split{self.split_id}/orig/train', i+1)
                    self.functions.write_kernels_correspondence(os.path.join(self.model_dir, f'markers_spx{spx0}'),
                                                                os.path.join(self.graphs_path_src[0], 'labels'),
                                                                os.path.join(self.graphs_path_src[i+1], 'labels'),
                                                                os.path.join(self.model_dir, 'model', f'kernels_map_{layer}.txt'),
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


    def run_model_parallel(self, img_splits_names, train=True, decode_train=True, activation_train=True,
                           extract=True, decode_extract=True, activation_extract=True,
                           decoders=["hybrid", "graph-adaptive"]):
        
        if not train and not extract:
            print("Nothing to do. Please set train and/or extract to True.")
            return

        for layer in range(0,self.nlayers+1):
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

            for layer in range(0,self.nlayers+1):
                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        if extract:
            self.__copyGraphsAndLabels__(f'{self.dataset_path}/imgs/orig')
            self.functions.extract(f'{self.dataset_path}/imgs/orig', f'{self.graphs_path_src}/graphs', f'{self.graphs_path_src}/labels', self.model_dir, self.arch_file, self.nlayers, 2, self.stride, self.attention, self.aggregation, self.dropout)

            for layer in range(1,self.nlayers+1):
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
    ARCH_NAME = 'arch_cells4'

    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'automatic_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    SPLIT_ID = 5

    model_dir = getModelPath(DS_NAME, SPLIT_ID, MARKERS_NAME, ARCH_NAME, 
                            STRIDE, SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                            ATTENTION, DROPOUT)

    model = GFLIMModel(DATASET_PATH, model_dir, 
                                    f'{DATASET_PATH}/archs/{ARCH_NAME}.json', 
                                    f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{SPLIT_ID}', 
                                    SPLIT_ID,
                                    SPX_ALG, INIT_SEEDS, FINAL_SPX, GRAPH_FEATS, 
                                    STRIDE, ATTENTION, AGGREGATION, DROPOUT)

    model.run_model_parallel('interactive_splits')


    """
    STRIDE = 'none'
    ATTENTION = 'CLASS_DOMINANCE'
    DROPOUT = True
    inputs += [(arch_id,split_id, STRIDE, ATTENTION, DROPOUT) for arch_id in [1] for split_id in range(1,6)]
    """

    #with multiprocessing.Pool() as pool:
    #    pool.starmap(run_model_parallel, ((arch_id, split_id, graphs_path_src, STRIDE, ATTENTION, DROPOUT) for (arch_id,split_id, STRIDE, ATTENTION, DROPOUT) in tqdm(inputs)))


def GFLIMMultiScaleUModelExample():
    DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/schisto'
    DS_NAME = DATASET_PATH.split('/')[-1]
    SPX_ALG = 'SH'
    INIT_SEEDS = 0
    FINAL_SPX = [10000,8000,6000]
    GRAPH_FEATS = 'SUM'
    ARCH_NAME = 'arch_schisto1'

    AGGREGATION = 'CONCAT'
    MARKERS_NAME = 'interactive_markers'
    STRIDE = 'graph-based'
    ATTENTION = 'NONE'
    DROPOUT = False

    for SPLIT_ID in range(2, 6):

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
                                        
        model.run_model_parallel('interactive_splits', activation_extract=False, activation_train=True, extract=False, decoders=[])


    """
    STRIDE = 'none'
    ATTENTION = 'CLASS_DOMINANCE'
    DROPOUT = True
    inputs += [(arch_id,split_id, STRIDE, ATTENTION, DROPOUT) for arch_id in [1] for split_id in range(1,6)]
    """

    #with multiprocessing.Pool() as pool:
    #    pool.starmap(run_model_parallel, ((arch_id, split_id, graphs_path_src, STRIDE, ATTENTION, DROPOUT) for (arch_id,split_id, STRIDE, ATTENTION, DROPOUT) in tqdm(inputs)))


if __name__ == "__main__":
    GFLIMMultiScaleUModelExample()