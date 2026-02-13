"""
This file contains classes for GFLIM, GFLIM BoF, MGFLIM BoFP (multiscale), and RGFLIM BoFP (recurrent)
Please be aware that some functionalities may not be implemented in all GFLIM versions.

GFLIM : First version, published in CIARP 2023. 
To create a kernel bank for a layer l, performs a kmeans per marker, 
a kmeans per training image, and a final kmeans. On each kmeans, the centroids are selected as candidate kernels.
It does not have bias, so a z-normalization is performed during training and inference.
It performs pooling, but with stride 1.
The activations can be decoded with the User-guided decoder.

GFLIM Bag of Features (BoFP): It selects seeds per training image with a kmeans per marker. 
On each kmeans, the seeds closest to the centroids are selected as kernels.
Performs a z-normalization and unit normalization to compute the kernels.
Compute a bias vector per layer.
Performs pooling with stride >= 1. The stride can be graph-based or image-based.
The activations can be decoded with the User-guided decoder, Hybrid decoder, or Graph-based decoder.

[in construction]
"""

import os
import sys
import subprocess
import json
from Scripts.finetuneSVM_2 import IFT_dir, run_script
from setup import *
from abc import ABC, abstractmethod

# To run SH: source ~/.local/share/virtualenvs/SH-NTEVX1e9/bin/activate

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

class GFLIMFunction():
    def __init__(self, command_name, tmp_file=None):
        self.command_name = command_name
        self.tmp_file = tmp_file
    
    def __call__(self, args = []):
        """
        Call a command with the given arguments.
        Args:
            args (list): The arguments to pass to the command.
            msg (str, optional): A message to display while calling the function.
        """
        cmd = "{} {}".format(self.command_name, " ".join([str(i) for i in args]))
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if self.tmp_file is not None:
            with open(self.tmp_file, "a+") as f:
                f.write(process.stdout)
        if process.returncode != 0:
            print(f"Error running command: {cmd}")
            print(process.stderr)
            sys.exit(process.returncode)
        return process.stdout

class Dataset():
    def __init__(self, images_path, graphs_path=None, 
                 use_masks=False, markers_path=None, tmp_file=None):
        self.images_path = images_path
        self.graphs_path = graphs_path
        self.use_masks = use_masks
        self.markers_path = markers_path
        self.tmp_file = tmp_file
        self.functions = {
            'CreateGraph': GFLIMFunction(f'{EXEC_DIR}/iftCreateGFLIMGraph', tmp_file),
            'CreateNeighbMatricesAndGraphs': GFLIMFunction(f'{EXEC_DIR}/iftCreateGFLIMNeighbMatricesAndGraphs', tmp_file),
            'CreateSpxImageSH': GFLIMFunction('python3 /home/isabela/Codes/SH/src/boruvkasupix.py', tmp_file),
            'CreateGraphFromLabels': GFLIMFunction(f'{EXEC_DIR}/iftCreateGFLIMGraphFromLabels', tmp_file),
        }

    def create_graphs_and_labels(self, spx_alg, spx, graph_feats, 
                                 graphs_path_src, initial_seeds=0, 
                                 kernel_neighbors='act', 
                                 pool_neighbors='act', 
                                 stride_neighbors='act',
                                 arch_file=None):
        """
        Create the graphs for the dataset if they do not exist.
        Args:
            images_path (str): images directory
            spx_alg (str): Superpixel algorithm to use.
            spx (int): Number of superpixels.
            graph_feats (str): Type of graph features to use.
            graphs_path_src (str): Path where the graphs and labels will be stored.
            initial_seeds (int): Initial number of seeds for the superpixel algorithm (only used when spx_alg=DISF).
            kernel_neighbors (str): Kernel neighbors type (act or color).
            pool_neighbors (str): Pool neighbors type (act or color).
            stride_neighbors (str): Stride neighbors type (act or color).
            arch_file (str): Architecture file (only used when 'color' neighbor types are used).
        """
        if kernel_neighbors not in ['act', 'color']:
            raise ValueError("create_graphs: kernel_neighbors must be 'act' or 'color': {}".format(kernel_neighbors))
        if pool_neighbors not in ['act', 'color']:
            raise ValueError("create_graphs: pool_neighbors must be 'act' or 'color': {}".format(pool_neighbors))
        if stride_neighbors not in ['act', 'color']:
            raise ValueError("create_graphs: stride_neighbors must be 'act' or 'color': {}".format(stride_neighbors))

        if spx_alg in ['DISF', 'ERGC'] and isinstance(spx, list):
            if len(spx) != 1:
                raise ValueError("create_graphs: spx must have only one value when spx_alg is DISF")
            else:
                spx = spx[0]

        if spx_alg == 'DISF':
            if initial_seeds <= 0 or initial_seeds < spx:
                raise ValueError("create_graphs: initial_seeds must be greater than 0 and greater than or equal to final_spx when spx_alg is DISF: {}".format(initial_seeds))
        else:
            if self.use_masks:
                raise ValueError("create_graphs: use_masks can only be used when spx_alg is DISF")

        if pool_neighbors == 'act' and stride_neighbors == 'color' and pool_neighbors == 'act':
            if spx_alg in ['DISF', 'ERGC']:
                self.functions['CreateGraph']([self.images_path, spx, graphs_path_src, graph_feats, spx_alg, initial_seeds, self.use_masks])
            else:
                self.functions['CreateSpxImageSH']([f'--img {self.images_path}', f'--save {graphs_path_src}', f'--node_feat {graph_feats}', f'--k {",".join([str(s) for s in spx])}'])
                for spx in spx:
                    self.functions['CreateGraphFromLabels']([f'{self.images_path}', f'{graphs_path_src}/{spx}/labels0', graph_feats, f'{graphs_path_src}/{spx}/layer0', self.use_masks])
        else:
            if spx_alg != 'DISF':
                raise ValueError("create_graphs: spx_alg must be DISF when using different neighbor types for kernel, pool, and stride: {}".format(spx_alg))
            if arch_file is None:
                raise ValueError("create_graphs: arch_file must be provided when using different neighbor types for kernel, pool, and stride")
            self.functions['NeighbMatricesAndGraphs']([self.images_path, spx, initial_seeds, graphs_path_src, graph_feats, arch_file, self.use_masks, '0', kernel_neighbors, pool_neighbors, stride_neighbors])

        self.graphs_path = graphs_path_src

class GFLIM_ABC(ABC):
    def __init__(self, dataset : Dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 nlayers, tmp_file, 
                 stride_type='graph-based', 
                 kernel_neighbors='act', 
                 pool_neighbors='act', 
                 stride_neighbors='act',
                 aggregation_opt='CONCAT', 
                 attention='NONE',
                 reduce_kernels=0, fixed_spx=0,
                 batch_size=64, 
                 initial_seeds=0):
        
        self.tmp_file = tmp_file
        self.model_dir = model_dir
        self.arch_file = arch_file
        self.dataset = dataset
        self.spx_alg = spx_alg
        self.spx = spx
        self.initial_seeds = initial_seeds
        self.stride_type = stride_type
        self.kernel_neighbors = kernel_neighbors
        self.pool_neighbors = pool_neighbors
        self.stride_neighbors = stride_neighbors
        self.graph_feat = graph_feat
        self.aggregation_opt = aggregation_opt
        self.attention = attention
        self.reduce_kernels = reduce_kernels
        self.fixed_spx = fixed_spx
        self.nlayers = nlayers
        self.batch_size = batch_size

        self.functions = {
            'WriteActivations': GFLIMFunction('{}/iftGFLIMWriteActivations'.format(EXEC_DIR), tmp_file),
        }

        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)

    def write_actiovations(self, layer, skip=None):
        self.functions['WriteActivations'](
            [layer, self.batch_size, 
             skip, self.arch_file])
    
    @abstractmethod
    def train_layer(self, layer):
        pass

    @abstractmethod
    def extract(self, nlayers, save_intermediate=False, layers_save_act=[], layers_decode=[], decoders=[]):
        pass

class GFLIM(GFLIM_ABC):
    def __init__(self, dataset : Dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 nlayers, tmp_file, 
                 stride_type='graph-based',
                 reduce_kernels=0, 
                 batch_size=64, 
                 initial_seeds=0):
        
        super().__init__(dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 nlayers, tmp_file, 
                 stride_type=stride_type, 
                 reduce_kernels=reduce_kernels,
                 batch_size=batch_size,
                 initial_seeds=initial_seeds)
        
        self.functions = {
            'CreateGraph': GFLIMFunction('{}/iftCreateGFLIMGraph'.format(EXEC_DIR), tmp_file),
            'ConvertSeeds': GFLIMFunction('{}/iftGFLIMConvertSeeds'.format(EXEC_DIR), tmp_file),
            'CreateImgSeedsAllPix': GFLIMFunction('python3 {}/Scripts/CreateSeedsFromAllPix_TrainSet.py'.format(IFT_dir), tmp_file),
            'PrecConvertSeeds': GFLIMFunction('{}/iftGFLIMPrecConvertSeeds'.format(EXEC_DIR), tmp_file),
            'WriteActivations': GFLIMFunction('{}/iftGFLIMWriteActivations'.format(EXEC_DIR), tmp_file),
            'DecodeLayer': GFLIMFunction('{}/iftGFLIMDecodeLayer'.format(EXEC_DIR), tmp_file),
            'TrainLayer': GFLIMFunction('{}/iftGFLIMTrainLayer'.format(EXEC_DIR), tmp_file),
            'ExtractFromGraphs': GFLIMFunction('{}/iftGFLIMExtract2'.format(EXEC_DIR), tmp_file),
            'ExtractFromImages': GFLIMFunction('{}/iftGFLIMExtract'.format(EXEC_DIR), tmp_file),
        }

    def create_graph_seeds(self, images_path, markers_dir, use_all_spx=False):
        """
        Create the seeds for the given architecture.
        Args:
            images_path (str): Directory where the training images are stored.
            markers_dir (str): Directory where the image markers are stored.
            use_all_spx (bool): Whether to use all pixels as superpixels.
        """
        if use_all_spx:
            markers_folder = f"{self.model_dir}/img_seeds"
            os.makedirs(markers_folder, exist_ok=True)
            self.functions['CreateImgSeedsAllPix'](
                [images_path, 
                self.dataset.images_path + '/masks',
                markers_folder])

        with open(self.arch_file) as f:
            arch = json.load(f)
            nfeat_markers = arch['layer1']['conv']["nkernels_per_marker"]
            kernel_size = arch['layer1']['conv']['kernel_size']

        if self.kernel_neighbors == 'act':
            self.functions['ConvertSeeds'](
                [self.model_dir, markers_dir, 
                 nfeat_markers, kernel_size, 
                 self.attention, self.aggregation_opt])
        else:
            raise ValueError(f"create_seeds: only kernel_neighbors 'act' is implemented in GFLIM, not {self.kernel_neighbors}")

    def train_layer(self, layer):
        """
        Train a GFLIM layer (not Bag of Features).
        Args:
            model_dir (str): Directory where the model is stored.
            arch_file (str): Path to the architecture file.
            layer (int): Layer number.
        """
        if layer < 1 or layer > self.nlayers:
            raise ValueError("train_layer: layer must be between 1 and nlayers: {}".format(layer))
        self.functions['TrainLayer']([self.model_dir, self.arch_file, layer])

    def extract(self, nlayers, save_intermediate=False, layers_save_act=[], layers_decode=[], decoders=[]):
        if nlayers < 1 or nlayers > self.nlayers:
            raise ValueError("extract: nlayers must be between 1 and self.nlayers: {}".format(nlayers))
        if self.spx_alg != 'DISF' and self.dataset.use_masks:
            raise ValueError("extract: use_masks can only be used when spx_alg is DISF")

        if self.dataset.graphs_path is not None:
            
            if os.path.exists(f"{self.dataset.graphs_path}/layer0") == False or \
               os.path.exists(f"{self.dataset.graphs_path}/labels0") == False:
                raise ValueError("GFLIM.extract: dataset.graphs_path does not contain layer0 and labels0: {}".format(self.dataset.graphs_path))

            self.functions['ExtractFromGraphs']([
                                self.dataset.images_path, f"{self.dataset.graphs_path}/layer0", 
                                f"{self.dataset.graphs_path}/labels0", 
                                self.model_dir, self.arch_file, 
                                nlayers, self.batch_size,
                                self.stride_type, self.attention, 
                                self.aggregation_opt, 
                                1 if self.reduce_kernels else 0,
                                1 if save_intermediate else 0,
                                ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                                ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                                ",".join(str(i) for i in decoders)])
        else:
            args = [self.dataset.images_path, 
                    self.model_dir, self.arch_file, 
                    nlayers, self.batch_size,
                    self.stride_type, self.attention, 
                    self.aggregation_opt, 
                    1 if self.reduce_kernels else 0,
                    self.graph_feat,
                    self.spx_alg,
                    self.spx,
                    1 if save_intermediate else 0,
                    ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                    ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                    ",".join(str(i) for i in decoders)]
            if self.spx_alg == 'DISF':
                args.append(self.initial_seeds)
                if self.dataset.use_masks:
                    args.append(1)
                else:
                    args.append(0)

            self.functions['ExtractFromImages'](args)


class GFLIM_BoFP(GFLIM_ABC):
    """
    GFLIM BoFP with no skip connections or multiscale.
    """
    def __init__(self, dataset : Dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 stride_type, 
                 attention, aggregation_opt, 
                 reduce_kernels,
                 nlayers, batch_size, tmp_file, 
                 kernel_neighbors='act',
                 pool_neighbors='act',
                 stride_neighbors='act',
                 initial_seeds=0):
        
        if stride_type not in ['graph-based', 'none'] and \
            (kernel_neighbors != 'act' or pool_neighbors != 'act' or stride_neighbors != 'act'):
            raise ValueError("GFLIM_BoFP: when using different neighbor types for kernel, pool, and stride, stride_type must be 'graph-based' or 'none'")

        if self.dataset.use_masks and spx_alg != 'DISF':
            raise ValueError("GFLIM_BoFP: use_masks can only be used when spx_alg is DISF")

        super().__init__(dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 nlayers, tmp_file, 
                 stride_type=stride_type,
                 aggregation_opt=aggregation_opt, 
                 attention=attention,
                 reduce_kernels=reduce_kernels,
                 batch_size=batch_size,
                 kernel_neighbors=kernel_neighbors,
                 pool_neighbors=pool_neighbors, 
                 stride_neighbors=stride_neighbors,
                 initial_seeds=initial_seeds)

        self.functions = {
            'CreateGraph': GFLIMFunction('{}/iftCreateGFLIMGraph'.format(EXEC_DIR), tmp_file),
            'ConvertSeeds': GFLIMFunction('{}/iftGFLIMConvertSeeds'.format(EXEC_DIR), tmp_file),
            'CreateImgSeedsAllPix': GFLIMFunction('python3 {}/Scripts/CreateSeedsFromAllPix_TrainSet.py'.format(IFT_dir), tmp_file),
            'WriteActivations': GFLIMFunction('{}/iftGFLIMWriteActivations'.format(EXEC_DIR), tmp_file),
            'DecodeLayer': GFLIMFunction('{}/iftGFLIMDecodeLayer'.format(EXEC_DIR), tmp_file),
            'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(EXEC_DIR),
            'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(EXEC_DIR),
            'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(EXEC_DIR),
            'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(EXEC_DIR),
            'EncodeMergedLayerBatch': '{}/iftGFLIMEncodeMergedLayerBatch'.format(EXEC_DIR),
            'ExtractFromGraphs': GFLIMFunction('{}/iftGFLIMExtract2'.format(EXEC_DIR), tmp_file),
            'ExtractFromImages': GFLIMFunction('{}/iftGFLIMExtract'.format(EXEC_DIR), tmp_file),
            # options for kernel, pool and stride neighbors different than 'act'
            'PrecConvertSeeds': GFLIMFunction('{}/iftGFLIMPrecConvertSeeds'.format(EXEC_DIR), tmp_file),
            'PrecCreateLayerModel': '{}/iftGFLIMPrecCreateLayerModel'.format(EXEC_DIR),
            'ExtractFromImagesColorNeighb': GFLIMFunction('{}/iftGFLIMExtractV3'.format(EXEC_DIR), tmp_file),
        }

    def create_seeds(self, images_path, img_markers_dir=None, use_all_spx=False):
        """
        Create the seeds for the given architecture.
        Args:
            images_path (str): Directory where the training images are stored.
            img_markers_dir (str): Directory where the image markers are stored (optional).
            use_all_spx (bool): Whether to use all pixels as superpixels (optional).
        """
        if use_all_spx:
            img_markers_dir = f"{self.model_dir}/img_seeds"
            os.makedirs(img_markers_dir, exist_ok=True)
            args = [images_path, img_markers_dir]
            if self.dataset.use_masks:
                args.append("--mask_dir")
                args.append(self.dataset.images_path + '/masks')
            self.functions['CreateImgSeedsAllPix'](args)
        
        with open(self.arch_file) as f:
            arch = json.load(f)
            nfeat_markers = arch['layer1']['conv']["nkernels_per_marker"]
            kernel_size = arch['layer1']['conv']['kernel_size']

        if self.kernel_neighbors == 'act':
            self.functions['ConvertSeeds'](
                [self.model_dir, img_markers_dir, 
                 nfeat_markers, kernel_size, 
                 self.attention, self.aggregation_opt])
        else:
            self.functions['PrecConvertSeeds']([
                self.model_dir, img_markers_dir, 
                nfeat_markers, kernel_size, 
                self.aggregation_opt])

    def train_layer(self, layer):
        """
        Train a GFLIM BoFP layer.
        Args:
            layer (int): Layer number.
        """
        if layer < 1 or layer > self.nlayers:
            raise ValueError("train_layer: layer must be between 1 and nlayers: {}".format(layer))

        if self.kernel_neighbors == "color" or self.stride_neighbors == "color":
            self.functions['PrecCreateLayerModel']([self.model_dir,
                    self.arch_file, layer, 
                    self.kernel_neighbors,
                    self.stride_neighbors,
                    self.graph_feat,
                    self.aggregation_opt,
                    self.dataset.images_path])
        else:
            self.functions['CreateLayerModel']([self.model_dir, 
                                           self.arch_file, layer, 
                                           self.stride_type, self.attention, 
                                           self.aggregation_opt])

        self.functions['MergeLayerModels'](self.model_dir, layer)

        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color": 
            self.functions['ExtractFromImagesColorNeighb']([
                self.model_dir, self.arch_file, layer, 
                self.batch_size, self.attention, 
                self.kernel_neighbors, self.pool_neighbors, 
                self.stride_neighbors, self.graph_feat, 
                self.aggregation_opt, self.dataset.images_path
            ])
        else:
            if self.batch_size == 1:
                self.functions['EncodeMergedLayer'](self.model_dir, self.arch_file,
                                                    layer, self.stride_type, 
                                                    self.attention, 
                                                    self.aggregation_opt, 
                                                    self.reduce_kernels)
            else:
                self.functions['EncodeMergedLayerBatch'](self.model_dir, 
                                                        self.arch_file, layer, 
                                                        self.batch_size,
                                                        self.stride_type, 
                                                        self.attention, 
                                                        self.aggregation_opt, 
                                                        self.reduce_kernels)

    def extract(self, nlayers, save_intermediate=False, layers_save_act=[], layers_decode=[], decoders=[]):
        if nlayers < 1 or nlayers > self.nlayers:
            raise ValueError("extract: nlayers must be between 1 and self.nlayers: {}".format(nlayers))
        
        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color":
            self.functions['ExtractFromImagesColorNeighb']([
                self.dataset.images_path, self.model_dir, 
                self.arch_file, nlayers, self.batch_size, 
                self.attention, self.aggregation_opt, 
                self.graph_feat, self.spx_alg, 
                self.spx, self.fixed_spx,
                1 if save_intermediate else 0,
                ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                ",".join(str(i) for i in decoders),
                self.kernel_neighbors, self.pool_neighbors, 
                self.stride_neighbors, 
                self.aggregation_opt, 
                self.initial_seeds,
                self.dataset.use_masks
                ])
        else:

            if self.dataset.graphs_path is not None:
                
                if os.path.exists(f"{self.dataset.graphs_path}/layer0") == False or \
                os.path.exists(f"{self.dataset.graphs_path}/labels0") == False:
                    raise ValueError("GFLIM.extract: dataset.graphs_path does not contain layer0 and labels0: {}".format(self.dataset.graphs_path))

                self.functions['ExtractFromGraphs']([
                                    self.dataset.images_path, f"{self.dataset.graphs_path}/layer0", 
                                    f"{self.dataset.graphs_path}/labels0", 
                                    self.model_dir, self.arch_file, 
                                    nlayers, self.batch_size,
                                    self.stride_type, self.attention, 
                                    self.aggregation_opt, 
                                    1 if self.reduce_kernels else 0,
                                    1 if save_intermediate else 0,
                                    ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                                    ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                                    ",".join(str(i) for i in decoders)])
            else:
                args = [self.dataset.images_path, 
                        self.model_dir, self.arch_file, 
                        nlayers, self.batch_size,
                        self.stride_type, self.attention, 
                        self.aggregation_opt, 
                        1 if self.reduce_kernels else 0,
                        self.graph_feat,
                        self.spx_alg,
                        self.spx,
                        1 if save_intermediate else 0,
                        ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                        ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                        ",".join(str(i) for i in decoders)]
                if self.spx_alg == 'DISF':
                    args.append(self.initial_seeds)
                    if self.dataset.use_masks:
                        args.append(1)
                    else:
                        args.append(0)

                self.functions['ExtractFromImages'](args)

class RGFLIM_BoFP(GFLIM_BoFP):
    """
    GFLIM BoFP with skip connections and no multiscale.
    """
    def __init__(self, dataset : Dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 stride_type, 
                 attention, aggregation_opt, 
                 reduce_kernels,
                 nlayers, batch_size, merge_function,
                 tmp_file, initial_seeds=0):
        
        if self.dataset.use_masks and spx_alg != 'DISF':
            raise ValueError("GFLIM_BoFP: use_masks can only be used when spx_alg is DISF")

        super().__init__(dataset, model_dir, arch_file,
                 spx_alg, spx, graph_feat, 
                 stride_type, 
                 attention, aggregation_opt, 
                 reduce_kernels,
                 nlayers, 
                 batch_size, tmp_file,
                 'act', 'act', 'act',
                 initial_seeds=initial_seeds)

        self.merge_function = merge_function

        self.functions = {
            'CreateGraph': GFLIMFunction('{}/iftCreateGFLIMGraph'.format(EXEC_DIR), tmp_file),
            'ConvertSeeds': GFLIMFunction('{}/iftGFLIMConvertSeeds'.format(EXEC_DIR), tmp_file),
            'CreateImgSeedsAllPix': GFLIMFunction('python3 {}/Scripts/CreateSeedsFromAllPix_TrainSet.py'.format(IFT_dir), tmp_file),
            'WriteActivations': GFLIMFunction('{}/iftGFLIMWriteActivations'.format(EXEC_DIR), tmp_file),
            'DecodeLayer': GFLIMFunction('{}/iftGFLIMDecodeLayer'.format(EXEC_DIR), tmp_file),
            'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(EXEC_DIR),
            'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(EXEC_DIR),
            'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(EXEC_DIR),
            'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(EXEC_DIR),
            'EncodeMergedLayerBatch': '{}/iftGFLIMEncodeMergedLayerBatch'.format(EXEC_DIR),
            'ExtractFromGraphs': GFLIMFunction('{}/iftGFLIMExtract2'.format(EXEC_DIR), tmp_file),
            'ExtractFromImages': GFLIMFunction('{}/iftGFLIMExtract'.format(EXEC_DIR), tmp_file),
            'WriteKernelsCorrespondence' : GFLIMFunction('{}/iftGFLIMWriteKernelsCorrespondence'.format(EXEC_DIR), tmp_file),
        }

    def create_seeds(self, images_path, img_markers_dir=None, use_all_spx=False):
        """
        Create the seeds for the given architecture.
        Args:
            images_path (str): Directory where the training images are stored.
            img_markers_dir (str): Directory where the image markers are stored (optional).
            use_all_spx (bool): Whether to use all pixels as superpixels (optional).
        """
        if use_all_spx:
            img_markers_dir = f"{self.model_dir}/img_seeds"
            os.makedirs(img_markers_dir, exist_ok=True)
            args = [images_path, img_markers_dir]
            if self.dataset.use_masks:
                args.append("--mask_dir")
                args.append(self.dataset.images_path + '/masks')
            self.functions['CreateImgSeedsAllPix'](args)
        
        with open(self.arch_file) as f:
            arch = json.load(f)
            nfeat_markers = arch['layer1']['conv']["nkernels_per_marker"]
            kernel_size = arch['layer1']['conv']['kernel_size']

        if self.kernel_neighbors == 'act':
            self.functions['ConvertSeeds'](
                [self.model_dir, img_markers_dir, 
                 nfeat_markers, kernel_size, 
                 self.attention, self.aggregation_opt])
        else:
            self.functions['PrecConvertSeeds']([
                self.model_dir, img_markers_dir, 
                nfeat_markers, kernel_size, 
                self.aggregation_opt])

    def train_layer(self, layer):
        """
        Train a GFLIM BoFP layer.
        Args:
            layer (int): Layer number.
        """
        super().train_layer(layer)
        self.write_kernels_correspondence(layer)
        # Get the first scale of the layer, which will dictate the number of nodes and features in the merged graph.
        first_scale = self.get_first_scale(layer)
        # merge layers output
        if first_scale != layer: # if there are skip connections
            self.skip_connection(layer, first_scale, write_activation)

    def write_kernels_correspondence(self, layer):
        """
        Write the kernels correspondence between scale 0 and scale i.
        Args:
            layer (int): Layer number.
        """
        self.functions['WriteKernelsCorrespondence'](
            [os.path.join(self.model_dir, f'markers'), 
             os.path.join(self.model_dir, 'labels0'), 
             os.path.join(self.model_dir, f'labels{layer-1}'), 
             os.path.join(self.model_dir, 'model', 'kernels_map.txt')])

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
        with open(self.arch_file) as f:
            self.arch = json.load(f)
        
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

    def skip_connection(self, layer, first_scale, write_activation):
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

            if self.merge_function == 'dotProd':
                merged_features = mergeFunction.merge_scales_dotProd(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif self.merge_function == 'sum':
                merged_features = mergeFunction.merge_scales_sum(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif self.merge_function == 'dotProd_sum_scale0':
                merged_features = mergeFunction.merge_scales_dotProd_sum_scale0(thresh=False, std=False) # 'merged_features' is a feature matrix
            elif self.merge_function == 'dotProd_sum_scalei':
                merged_features = mergeFunction.merge_scales_dotProd_sum_scalei(thresh=False, std=False) # 'merged_features' is a feature matrix
            else:
                raise ValueError(f"Unknown merge function: {self.merge_function}")

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

    def extract(self, nlayers, save_intermediate=False, layers_save_act=[], layers_decode=[], decoders=[]):
        if nlayers < 1 or nlayers > self.nlayers:
            raise ValueError("extract: nlayers must be between 1 and self.nlayers: {}".format(nlayers))
        
        if self.kernel_neighbors == "color" or self.pool_neighbors == "color" or self.stride_neighbors == "color":
            self.functions['ExtractFromImagesColorNeighb']([
                self.dataset.images_path, self.model_dir, 
                self.arch_file, nlayers, self.batch_size, 
                self.attention, self.aggregation_opt, 
                self.graph_feat, self.spx_alg, 
                self.spx, self.fixed_spx,
                1 if save_intermediate else 0,
                ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                ",".join(str(i) for i in decoders),
                self.kernel_neighbors, self.pool_neighbors, 
                self.stride_neighbors, 
                self.aggregation_opt, 
                self.initial_seeds,
                self.dataset.use_masks
                ])
        else:

            if self.dataset.graphs_path is not None:
                
                if os.path.exists(f"{self.dataset.graphs_path}/layer0") == False or \
                os.path.exists(f"{self.dataset.graphs_path}/labels0") == False:
                    raise ValueError("GFLIM.extract: dataset.graphs_path does not contain layer0 and labels0: {}".format(self.dataset.graphs_path))

                self.functions['ExtractFromGraphs']([
                                    self.dataset.images_path, f"{self.dataset.graphs_path}/layer0", 
                                    f"{self.dataset.graphs_path}/labels0", 
                                    self.model_dir, self.arch_file, 
                                    nlayers, self.batch_size,
                                    self.stride_type, self.attention, 
                                    self.aggregation_opt, 
                                    1 if self.reduce_kernels else 0,
                                    1 if save_intermediate else 0,
                                    ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                                    ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                                    ",".join(str(i) for i in decoders)])
            else:
                args = [self.dataset.images_path, 
                        self.model_dir, self.arch_file, 
                        nlayers, self.batch_size,
                        self.stride_type, self.attention, 
                        self.aggregation_opt, 
                        1 if self.reduce_kernels else 0,
                        self.graph_feat,
                        self.spx_alg,
                        self.spx,
                        1 if save_intermediate else 0,
                        ",".join(str(i) for i in layers_save_act) if len(layers_save_act) > 0 else -1,
                        ",".join(str(i) for i in layers_decode) if len(layers_decode) > 0 else -1,
                        ",".join(str(i) for i in decoders)]
                if self.spx_alg == 'DISF':
                    args.append(self.initial_seeds)
                    if self.dataset.use_masks:
                        args.append(1)
                    else:
                        args.append(0)

                self.functions['ExtractFromImages'](args)


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
        #os.system("{} {}".format(function, " ".join([str(i) for i in args])))

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
        self.__callFunction__(self.functions['TrainLayer'], [model_dir, arch_file, layer], "Creating weights layer {}:\t".format(layer))    
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