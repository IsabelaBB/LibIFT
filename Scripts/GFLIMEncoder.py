import os
import json
import shutil
from GFLIMFunctions import *

class GFLIMEncoder():
    """
    A class to hold the GFLIM model and its parameters.
    1. Create the graphs if they do not exist.
    2. Train the model.
    3. Extract features from the model.
    """
    def __init__(self, model_dir, arch_name, markers_dir,
                 spx_alg, init_seeds, final_spx, graph_feats):
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
        self.arch_file = os.path.join(ARCH_DIR, f"{arch_name}.json")
        self.markers_dir = markers_dir
        self.model_dir = model_dir

        if(len(final_spx) < 1):
            raise ValueError("GFLIMEncoder: final_spx must have at least one value")
        if(len(final_spx) > 1 and spx_alg != 'SH'):
            raise ValueError("GFLIMEncoder: when final_spx has more than one value, spx_alg must be 'SH'")
        if(len(final_spx) == 1 and final_spx[0] < 1):
            raise ValueError("GFLIMEncoder: final_spx must be at least 1")
        if(init_seeds < 1 and spx_alg == 'DISF'):
            raise ValueError("GFLIMEncoder: init_seeds must be at least 1 when spx_alg is 'DISF'")
        if(init_seeds < final_spx[0] and spx_alg == 'DISF'):
            raise ValueError("GFLIMEncoder: init_seeds must be greater than or equal to final_spx when spx_alg is 'DISF'")

        self.functions = GFLIMFunctions()
        self.arch = self.read_architecture(self.arch_file)

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

    def __createGraphsAndLabels__(self, imgs_src, out_path_parent, use_mask=0):
        """
        Create graphs and superpixel images in out_path_parent using the images in imgs_src.
        Args:
            imgs_src (str): Path to the images directory or a csv file containing the list of images.
            out_path_parent (str): Parent directory where the graphs and labels will be stored.
            use_mask (int): Whether to use a binary mask when creating the graphs (0 or 1). The mask should be in the same directory as the images, in folder 'masks'.
        Returns:
            graphs_path_src (str or list): Path(s) to the created graphs directory(ies).
            spx_label_path_src (str or list): Path(s) to the created superpixel labels directory(ies).
        """
        if(not os.path.exists(imgs_src)):
            raise ValueError("__createGraphsAndLabels__: imgs_src does not exist")

        # create the output path
        if(len(self.final_spx) == 1):
            self.functions.create_graphs(imgs_src, self.final_spx, out_path_parent, self.graph_feats, self.spx_alg, self.init_seeds, use_mask)

        else:
            out_path = os.path.join(out_path_parent, 'superpixels', self.spx_alg, self.graph_feats)
            graphs_path_src, spx_label_path_src = [],[]
            for spx in self.final_spx:
                graphs_path_src.append(os.path.join(out_path, f'{spx}', 'graphs'))
                spx_label_path_src.append(os.path.join(out_path, f'{spx}', 'labels'))

            self.functions.create_graphs(imgs_src, self.final_spx, out_path, self.graph_feats, self.spx_alg, self.init_seeds, use_mask)

    def __train_layer__(self, layer, write_activation, batchsize, decoders=[]):
        """
        Train a single layer of the model.
        Args:
            layer (int): Layer number.
            write_activation (bool): Whether to write the activations.
            batchsize (int): Batch size.
            decoders (list): List of decoders to use.
        """
        self.functions.train_gflim_layer(self.model_dir, self.arch_file, layer)
        if write_activation:
            self.functions.write_activations(self.model_dir, layer, batchsize)
        for d in decoders: 
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, "none", batchsize)

    def prepare_inputs(self, imgs_src, input_path_parent=None, use_mask=0):
        """
        Create/Copy graphs and superpixel labels for training or extraction.
        Args:
            imgs_src (str): Path to the images directory or a csv file containing the list of images.
            input_path_parent (str): Path to the parent directory where the graphs and labels are stored. If None, graphs and labels will be created.
            use_mask (int): Whether to use a binary mask when creating the graphs (0 or 1). The mask should be in the same directory as the images, in folder 'masks'.
        Note:
            If input_path_parent is None, graphs and labels will be created in the model directory.
            Otherwise, input_path_parent must contain the directories layer0 and labels0 containing the graphs and labels, respectively.
            These directories must contain graphs / spx. labels for all the images in imgs_src.
        """

        shutil.rmtree(os.path.join(self.model_dir, 'layer0'), ignore_errors=True)
        shutil.rmtree(os.path.join(self.model_dir, 'labels0'), ignore_errors=True)

        if(input_path_parent is None):
            self.__createGraphsAndLabels__(imgs_src, self.model_dir, use_mask)
        else:
            if(len(self.final_spx) == 1):
                graphs_path_src = os.path.join(input_path_parent, 'layer0')
                spx_label_path_src = os.path.join(input_path_parent, 'labels0')
                copyGraphsAndLabels(imgs_src, graphs_path_src, spx_label_path_src, self.model_dir)
            else:
                for spx in self.final_spx:
                    input_graphs_path = os.path.join(input_path_parent, f'{spx}', 'layer0')
                    input_labels_path = os.path.join(input_path_parent, f'{spx}', 'labels0')
                    copyGraphsAndLabels(imgs_src, input_graphs_path, input_labels_path, os.path.join(self.model_dir, f'scales', f'{spx}'))

    def train(self, images_folder,
              layers_activations,
              layers_decode, decoders,
              batchsize,
              inputs_folder=None
              ):
        """
        Train the GFLIM model.
        Args:
            images_folder (str): Path to the images folder.
            layers_activations (list): List of layers to save activations.
            layers_decode (list): List of layers to decode.
            decoders (list): List of decoders to use.
            batchsize (int): Batch size.
            inputs_folder (str): Path to the parent directory where the graphs and labels are stored. 
                If None, graphs and labels will be created in model_dir/layer0 and model_dir/labels0.
        Steps:
            1. Copy the architecture file to the model directory.
            2. Prepare input files
            3. Convert the seeds using the architecture parameters.
            4. For each layer in the architecture:
                a. Train the layer.
        """
        # delete previous model if exists
        shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # copy arch file to the model directory
        file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
        if os.path.isfile(file): # if it exists, remove it
            os.remove(file)
        shutil.copy(self.arch_file, self.model_dir)

        # prepare input files
        self.prepare_inputs(images_folder, inputs_folder)
        
        # convert from image seeds to graph seeds and perform k-means
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch['layer1']['conv']["nkernels_per_marker"], self.arch['layer1']['conv']['kernel_size'], "NONE", "CONCAT")
        
        for layer in range(1, self.arch['nlayers'] + 1):
            self.__train_layer__(layer, 
                                 layer in layers_activations, 
                                 batchsize, 
                                 decoders if layer in layers_decode else [])

    def encode(self, images_folder, 
                save_intermediate=False, layers_activations = [],
                layers_decode = [], decoders = [], 
                batchsize=64, use_mask=0, input_folder=None):
        """
        Run (train and/or extract) a GFLIM model.
        Args:
            images_folder (str): Path to the images folder.
            save_intermediate (bool, optional): Whether to save intermediate features and superpixel labels.
            layers_activations (list): List of layers to save activations.
            layers_decode (list): List of layers to decode.
            decoders (list): List of decoders to use.
            batchsize (int): Batch size.
            use_mask (int): Whether to use a binary mask when creating the graphs (0 or 1). The mask should be in the same directory as the images, in folder 'masks'.
            input_folder (str): Path to the parent directory where the graphs and labels are stored. 
                If None, graphs and labels will be created in model_dir/layer0 and model_dir/labels0.
                Otherwise, input_folder must contain the directories layer0 and labels0 containing the graphs and labels, respectively.
        Note:
            If using a binary mask, the mask images must be located in the 'masks' subdirectory of the input_folder.

        Steps:
            1. Remove any existing layer and label directories in the model directory except for the last layer (when `save_intermediate` is False).
            2. Prepare input files. Create or copy graphs and superpixel labels.
            3. For each layer in the architecture:
                a. Extract features using the model (and decode if specified).
        """
        # if save_intermediate is true, do not remove the intermediate files
        if not save_intermediate:
            # remove any existing layer and label directories in the model directory except for the last layer
            for layer in range(0,self.arch['nlayers']):
                shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
                shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)

        # prepare input files
        self.prepare_inputs(images_folder, input_folder, use_mask)

        graphs_path_src = os.path.join(self.model_dir, 'layer0')
        spx_label_path_src = os.path.join(self.model_dir, 'labels0')

        # copy the graphs and superpixel labels from the dataset directory to the model directory
        self.functions.extract(images_folder, 
                            graphs_path_src, spx_label_path_src, 
                            self.model_dir, self.arch_file, self.arch['nlayers'], batchsize, 
                            "none", "NONE", "CONCAT", 0,
                            save_intermediate, layers_activations, 
                            layers_decode, decoders)

