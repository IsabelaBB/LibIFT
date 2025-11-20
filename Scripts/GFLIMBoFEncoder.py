import os
import shutil
import argparse
from GFLIMEncoder import GFLIMEncoder
from GFLIMFunctions import list_of_ints,list_of_str

class GFLIMBoFEncoder(GFLIMEncoder):
    """
    A class to hold the GFLIM model and its parameters.
    1. Create the graphs if they do not exist.
    2. Train the model.
    3. Extract features from the model.
    """
    def __init__(self, model_dir, arch_name, markers_dir,
                 spx_alg, init_seeds, final_spx, graph_feats, 
                 stride, attention, aggregation, dropout):
        """
        Initialize the GFLIM model with the given parameters.
        Args:
            model_dir (str): Directory where the model will be stored.
            arch_name (str): Name of the architecture file.
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
        
        super().__init__(model_dir, arch_name, markers_dir,
                         spx_alg, init_seeds, final_spx, graph_feats)
        self.attention = attention
        self.aggregation = aggregation
        self.dropout = dropout
        self.stride = stride

    def __train_layer__(self, layer, write_activation, batchsize, decoders=[]):
        """
        Train a single layer of the model.
        Args:
            layer (int): Layer number.
            write_activation (bool): Whether to write the activations.
            batchsize (int): Batch size.
            decoders (list): List of decoders to use.
        """
        self.functions.create_layer_model(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation)
        self.functions.merge_layer_models(self.model_dir, layer)
        self.functions.encode_merged_layer(self.model_dir, self.arch_file, layer, self.stride, self.attention, self.aggregation, dropout=self.dropout, batch=batchsize)
        if write_activation:
            self.functions.write_activations(self.model_dir, layer, batchsize)
        for d in decoders: 
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

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
            self.functions.write_activations(self.model_dir, layer, batchsize, first_scale, self.arch_file)

        for d in decoders: 
            self.functions.decode_layer(self.model_dir, self.arch_file, layer, d, self.stride)

    def train(self, images_folder,
              layers_activations,
              layers_decode, decoders,
              batchsize,
              inputs_folder=None,
              use_mask=0,
              layers=[]
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

        if(len(layers) == 0):
            layers = list(range(1, self.arch['nlayers'] + 1))

        for l in range(layers[0],self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{l}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{l}', ignore_errors=True)

        if(layers[0] == 1):
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
            # remove previous model and markers
            shutil.rmtree(os.path.join(self.model_dir, "model"), ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/markers', ignore_errors=True)

            # copy arch file to the model directory
            file = os.path.join(self.model_dir, self.arch_file.split('/')[-1])
            if os.path.isfile(file): # if it exists, remove it
                os.remove(file)
            shutil.copy(self.arch_file, self.model_dir)

            # prepare input files
            self.prepare_inputs(images_folder, inputs_folder, use_mask)

            # convert from image seeds to graph seeds and perform k-means
            self.functions.convert_seeds(self.model_dir, self.markers_dir, self.arch['layer1']['conv']["nkernels_per_marker"], self.arch['layer1']['conv']['kernel_size'], self.attention, self.aggregation)

        for layer in layers:
            # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
            first_scale = self.get_first_scale(layer)

            self.__train_layer__(layer, 
                                 layer in layers_activations, 
                                 batchsize, 
                                 decoders if layer in layers_decode else [])

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
                batchsize=64, use_mask=0, input_folder=None,
                nlayers=None
                ):
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
        for layer in range(0,self.arch['nlayers']+1):
            shutil.rmtree(f'{self.model_dir}/labels{layer}', ignore_errors=True)
            shutil.rmtree(f'{self.model_dir}/layer{layer}', ignore_errors=True)
        
        # prepare input files
        self.prepare_inputs(images_folder, input_path_parent=input_folder, use_mask=use_mask)

        graphs_path_src = os.path.join(self.model_dir, 'layer0')
        spx_label_path_src = os.path.join(self.model_dir, 'labels0')
        if nlayers is None:
            nlayers = self.arch['nlayers']

        # copy the graphs and superpixel labels from the dataset directory to the model directory
        self.functions.extract(images_folder, 
                            graphs_path_src, 
                            spx_label_path_src, 
                            self.model_dir, self.arch_file, nlayers, batchsize, 
                            self.stride, self.attention, self.aggregation, self.dropout,
                            save_intermediate, 
                            layers_activations, 
                            layers_decode,
                            decoders)

def parse_args():
    parser = argparse.ArgumentParser("GFLIM-BoF Encoder")
    parser.add_argument("op", type=str, help="Operation to perform.", choices=['train', 'encode'])
    parser.add_argument("images_folder", type=str, help="Path to the images folder / csv file.")
    parser.add_argument("model_dir", type=str, help="Directory where the model is stored.")
    parser.add_argument("spx_alg", type=str, help="Superpixel algorithm.", choices=['DISF', 'ERGC'])
    parser.add_argument("final_spx", help="Final number of superpixels (it can be a list in the form 1,2,3).", type=list_of_ints) # example: 10000
    
    parser.add_argument("--graph_feats", type=str, help="Type of graph features to use.", default='SUM', choices=['SUM', 'MEAN'])
    parser.add_argument("--stride", type=str, help="Stride type ('graph-based', 'none', etc.).", default='graph-based', choices=['graph-based', 'none'])
    parser.add_argument("--attention", type=str, help="Attention type ('NONE', 'CLASS_DOMINANCE', etc.).", default='NONE', choices=['NONE', 'CLASS_DOMINANCE'])
    parser.add_argument("--aggregation", type=str, help="Aggregation strategy ('CONCAT', 'MEAN', 'WMEAN', WCONCAT).", default='CONCAT', choices=['CONCAT', 'MEAN', 'WMEAN', 'WCONCAT'])

    parser.add_argument("--arch_name", type=str, help="Name of the architecture file.")
    parser.add_argument("--markers_dir", type=str, help="Directory where the image markers are stored.", default=None)
    parser.add_argument("--init_seeds", type=int, help="Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).", default=20000)
    parser.add_argument("--dropout", help="Dropout rate to use.", type=int, default=0, choices=[0, 1])
    parser.add_argument("--batch_size", help="Batch size for training.", type=int, default=64)
    parser.add_argument("--decoders", help="Decoders to use.", type=list_of_str, default=['hybrid', 'graph-adaptive'])
    parser.add_argument("--save_intermediate", help="Whether to save intermediate results (folders layerX and labelsX in model_dir).", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_mask", help="Whether to use masks during training.", type=int, default=0, choices=[0, 1])

    parser.add_argument("--layers_activations", help="Layers to save activations (it can be a list in the form 1,2,3).", type=list_of_ints, default=[])
    parser.add_argument("--layers_decode", help="Layers to decode (it can be a list in the form 1,2,3).", type=list_of_ints, default=[])
    parser.add_argument("--layers", help="Layers to process (it can be a list in the form 1,2,3). Default: the number of layers in the arch file", type=list_of_ints, default=[])

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    if(len(args.final_spx) > 1):
        raise ValueError("final_spx must be a single integer when using DISF or ERGC as superpixel algorithm.")
    if(args.op == 'train' and args.markers_dir is None):
        raise ValueError("markers_dir must be provided when training the model.")
    if(args.op == 'train' and args.arch_name is None):
        raise ValueError("arch_name must be provided when training the model.")

    # get dataset parent dir
    ds_path = os.path.dirname(args.images_folder)
    print("Dataset path:", ds_path)

    graphs_path = os.path.join(ds_path, 'superpixels', args.spx_alg, args.graph_feats, f'{args.init_seeds}-{args.final_spx[0]}')
    print("Graphs path:", graphs_path)
    model = GFLIMBoFEncoder(args.model_dir, args.arch_name, args.markers_dir,
                            args.spx_alg, args.init_seeds, args.final_spx, args.graph_feats,
                            args.stride, args.attention, args.aggregation, args.dropout)
    
    # verify the specified layers
    if(len(args.layers) > 0):
        for l in args.layers:
            if l < 1 or l > model.arch['nlayers']:
                raise ValueError("Layers must be greater than 0 and less than or equal to the number of layers in the architecture.")
    
        # sort layers in increasing order
        args.layers.sort()
    
    # check whether graph/ spx. images already exist in the dataset folder
    create_graphs = False
    if os.path.exists(os.path.join(graphs_path, 'layer0')) and os.path.exists(os.path.join(graphs_path, 'labels0')):
        create_graphs = False
        """
        if(CheckFilesRelation_dirOrCSV(os.path.join(graphs_path, 'layer0'), os.path.join(graphs_path, 'labels0'), same_extension = False) != 'eq'):
            create_graphs = True
        rel_src = CheckFilesRelation_dirOrCSV(args.images_folder, os.path.exists(os.path.join(graphs_path, 'layer0')), same_extension = False)
        if(rel_src == 'diff' or rel_src == 'gt'):
            create_graphs = True
        """
    else:
        create_graphs = True

    if(len(args.layers) == 0 or args.layers[0] == 1):
        for l in range(0, model.arch['nlayers'] + 1):
            shutil.rmtree(os.path.join(args.model_dir, f'layer{l}'), ignore_errors=True)
            shutil.rmtree(os.path.join(args.model_dir, f'labels{l}'), ignore_errors=True)
            
    if(create_graphs and (len(args.layers) == 0 or args.layers[0] == 1)):
        model.__createGraphsAndLabels__(args.images_folder, graphs_path, args.use_mask)

    # perform the requested operation
    if args.op == 'train':
        model.train(args.images_folder,
                    layers_activations=args.layers_activations,
                    layers_decode=args.layers_decode,
                    decoders=args.decoders,
                    batchsize=args.batch_size,
                    inputs_folder=graphs_path,
                    use_mask=args.use_mask, layers=args.layers)
    elif args.op == 'encode':
        model.encode(args.images_folder,
                     save_intermediate=bool(args.save_intermediate),
                     layers_activations=args.layers_activations,
                     layers_decode=args.layers_decode,
                     decoders=args.decoders,
                     batchsize=args.batch_size,
                     input_folder=graphs_path,
                     use_mask=args.use_mask, nlayers=args.layers[-1] if len(args.layers) > 0 else [])