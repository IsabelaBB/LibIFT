import os
import shutil
from GFLIMBoFEncoder import GFLIMBoFEncoder

class GFLIMBoFMultiScaleEncoder(GFLIMBoFEncoder):
    """
    A class to hold the GFLIM model and its parameters.
    1. Create the graphs if they do not exist.
    2. Train the model.
    3. Extract features from the model.
    """
    def __init__(self, model_dir, arch_file, markers_dir,
                 spx_alg, final_spx, graph_feats, 
                 stride, attention, aggregation, dropout,
                 merge_option : str = 'dotprod'):
        """
        Initialize the GFLIM model with the given parameters.
        Args:
            model_dir (str): Directory where the model will be stored.
            arch_file (str): Path to the architecture file.
            markers_dir (str): Directory where the image markers are stored.
            spx_alg (str): Superpixel algorithm to use ('DISF', 'ERGC', 'SLIC', 'LSC', 'SH').
            final_spx (list): Final number of superpixels. If spx_alg is 'DISF' or 'ERGC', this is an integer. If spx_alg is 'SLIC', 'LSC', or 'SH', this is a list of integers.
            graph_feats (str): Type of graph features to use ('RGB', 'RGBXY', 'HSV', 'HSVI', 'LAB', 'LABI').
            stride (str): Stride type ('graph-based', 'none', etc.).
            attention (str): Attention type ('NONE', 'CLASS_DOMINANCE', etc.).
            aggregation (str): Aggregation strategy ('CONCAT', 'MEAN', 'WMEAN', WCONCAT).
            dropout (bool): Whether to apply dropout.
        """
        
        super().__init__(model_dir, arch_file, markers_dir,
                         spx_alg, 1, final_spx, graph_feats, 
                         stride, attention, aggregation, dropout)
        self.merge_option = merge_option

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
            2. Convert the seeds using the architecture parameters.
            3. For each layer in the architecture:
                a. Train the layer.
                b. Write the kernels correspondence between layer 1 and the others
                c. If the layer has skip connections, merge.
        """
        # delete previous model if exists
        shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # copy arch file to the model directory
        file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
        if os.path.isfile(file): # if it exists, remove it
            os.remove(file)
        shutil.copy(self.arch_file, self.model_dir)

        # create graphs and labels in model_dir/scales/spx/layer0 and model_dir/scales/spx/labels0
        self.prepare_inputs(images_folder, inputs_folder)

        # copy graphs and superpixel images from final_spx[0] to model_dir/layer0 and model_dir/labels0 to convert markers
        shutil.copytree(os.path.join(self.model_dir, f'scales/{self.final_spx[0]}/layer0'), os.path.join(self.model_dir, 'layer0'))
        shutil.copytree(os.path.join(self.model_dir, f'scales/{self.final_spx[0]}/labels0'), os.path.join(self.model_dir, 'labels0'))

        # convert from image seeds to graph seeds and perform k-means
        self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch['layer1']['conv']["nkernels_per_marker"], self.arch['layer1']['conv']['kernel_size'], self.attention, self.aggregation)
        
        layer = 1
        self.__train_layer__(layer, 
                             layer in layers_activations, 
                             batchsize, 
                             decoders if layer in layers_decode else [])

        spx0 = self.final_spx[0]
        for spx in self.final_spx[1:]:
            shutil.copytree(os.path.join(self.model_dir, f'scales/{spx}/layer0'), os.path.join(self.model_dir, 'layer0'))
            shutil.copytree(os.path.join(self.model_dir, f'scales/{spx}/labels0'), os.path.join(self.model_dir, 'labels0'))

            self.__train_layer__(layer, 
                                 layer in layers_activations, 
                                 batchsize, 
                                 decoders if layer in layers_decode else [])

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
        self.functions.merge_scales(images_folder, self.final_spx, self.model_dir, os.path.join(self.model_dir, 'model', f'kernels_map_{layer}_scales.txt'), layer, self.merge_option)

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
            (not used) use_mask (int): Whether to use a binary mask when creating the graphs (0 or 1). The mask should be in the same directory as the images, in folder 'masks'.
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

        graphs_path_src, spx_label_path_src = [], []
        for spx in self.final_spx:
            graphs_path_src.append(os.path.join(self.model_dir, f'scales/{spx}/layer0'))
            spx_label_path_src.append(os.path.join(self.model_dir, f'scales/{spx}/labels0'))

        # copy the graphs and superpixel labels from the dataset directory to the model directory
        self.functions.extract_multiscale(images_folder, 
                            graphs_path_src, spx_label_path_src, 
                            self.model_dir, self.arch_file, self.arch['nlayers'], batchsize, 
                            self.stride, self.attention, self.aggregation, self.dropout,
                            save_intermediate, layers_activations, 
                            layers_decode, decoders,
                            self.merge_option, self.final_spx,
                            os.path.join(self.model_dir, 'model', 'kernels_map.txt'))

