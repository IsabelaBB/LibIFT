import os
import sys
from setup import *

# To run SH: source ~/.local/share/virtualenvs/SH-NTEVX1e9/bin/activate

class GFLIMFunctions():
    """
    A class to hold the paths to the GFLIM functions.
    """
    def __init__(self):
        self.functions = {
            # Functions for both
            'CreateGraph': '{}/iftCreateGFLIMGraph'.format(EXEC_DIR),
            'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(EXEC_DIR),
            'WriteActivations': '{}/iftGFLIMWriteActivations'.format(EXEC_DIR),
            'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(EXEC_DIR),
            'Extract': '{}/iftGFLIMExtract2'.format(EXEC_DIR),

            # Functions GFLIM
            'TrainLayer': '{}/iftGFLIMTrainLayer'.format(EXEC_DIR),

            # Functions GFLIM Bag of Features
            'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(EXEC_DIR),
            'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(EXEC_DIR),
            'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(EXEC_DIR),
            'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(EXEC_DIR),
            'EncodeMergedLayerBatch': '{}/iftGFLIMEncodeMergedLayerBatch'.format(EXEC_DIR),
            'WriteKernelsCorrespondence': '{}/iftGFLIMWriteKernelsCorrespondence'.format(EXEC_DIR),
            'CreateSpxImageSH': 'python3 /home/isabela/Codes/SH/src/boruvkasupix.py',
            'CreateGraphFromLabels': '{}/iftCreateGFLIMGraphFromLabels'.format(EXEC_DIR),
            'SkipConnection': '{}/iftGFLIMSkipConnection'.format(EXEC_DIR),
            'MergeScales': '{}/iftGFLIMMergeScales'.format(EXEC_DIR),
            'ExtractMultiscale': '{}/iftGFLIMExtractMultiscale'.format(EXEC_DIR)
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

    def create_graphs(self, images_path, final_spx : list, graphs_path_src, graph_feats, spx_alg, init_seeds=0, use_mask=0):
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
            self.__callFunction__(self.functions['CreateGraph'], [images_path, final_spx[0], graphs_path_src, graph_feats, spx_alg, init_seeds, use_mask], 'Creating graphs')
        else:
            self.__callFunction__(self.functions['CreateSpxImageSH'], [f'--img {images_path}', f'--save {graphs_path_src}', f'--node_feat {graph_feats}', f'--k {",".join([str(spx) for spx in final_spx])}'], 'Creating superpixel images')
            for spx in final_spx:
                self.__callFunction__(self.functions['CreateGraphFromLabels'], [f'{images_path}', f'{graphs_path_src}/{spx}/labels0', graph_feats, f'{graphs_path_src}/{spx}/layer0', use_mask], 'Creating graphs from superpixel images')

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

    def write_activations(self, model_dir, layer, batchsize, skip=None, arch=None):
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
            if arch == None:
                self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer, batchsize, skip], "Writing activations layer {} with skip connection:\t".format(layer))
            else:
                self.__callFunction__(self.functions['WriteActivations'], [model_dir, layer, batchsize, skip, arch], "Writing activations layer {} with skip connection:\t".format(layer))

    def merge_layer_models(self, model_dir, layer):
        """
        Merge the layer models for the given layer.
        Args:
            model_dir (str): Directory where the model is stored.
            layer (int): Layer number.
        """
        self.__callFunction__(self.functions['MergeLayerModels'], [model_dir, layer], "Merging weights layer {}:\t".format(layer))

    def encode_merged_layer(self, model_dir, arch_file, layer, stride, attention, aggregation, batch=1, dropout=False):
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
        if batch == 1:
            self.__callFunction__(self.functions['EncodeMergedLayer'], [model_dir, arch_file, layer, stride, attention, aggregation, 1 if dropout else 0], "Encoding Merged layer {}:\t".format(layer))
        else:
            self.__callFunction__(self.functions['EncodeMergedLayerBatch'], [model_dir, arch_file, layer, batch, stride, attention, aggregation, 1 if dropout else 0], "Encoding Merged layer {} (batch={}):\t".format(layer, batch))

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

    def train_gflim_layer(self, model_dir, arch_file, layer):
        """
        Train a GFLIM layer (not Bag of Features).
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
        """
        self.__callFunction__(self.functions['TrainLayer'], [model_dir, arch_file, layer], "Creating weights layer {}:\t".format(layer))