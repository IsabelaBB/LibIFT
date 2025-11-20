"""
This program implements functions to train and extract features using the GFLIM-BoFP and multi-scale GFLIM-BoFP.

"""

import os
import shutil
from PIL import Image
import numpy as np
import argparse
from Scripts_GFLIM.GFLIMFunctions import GFLIMFunctions
from Scripts_GFLIM.GFLIMModel import GFLIMModel

# source ~/.local/share/virtualenvs/SH-NTEVX1e9/bin/activate

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_str(arg):
    return list(map(str, arg.split(',')))


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

        for d in decoders: 
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

    def run_model(self, images_folder, 
                        graphs_path_src, spx_label_path_src,
                        process='train', 
                        save_intermediate=False, layers_activations = [],
                        layers_decode = [], decoders=["hybrid", "graph-adaptive"], 
                        batchsize=64, use_mask=0):
        
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
            img_files = []
            if(os.path.isdir(images_folder)):
                img_files = [os.path.basename(file).split('.')[0] for file in os.listdir(images_folder)]
            else:
                if(os.path.isfile(images_folder) == False):
                    raise ValueError("run_model: images_folder must be a directory or a file containing the list of images")
                with open(images_folder, 'r') as f:
                    img_files = [os.path.basename(file).split('.')[0] for file in f.read().splitlines()]
            
            create_graphs = False
            # verify if the provided paths exist and contain the graphs and labels for all images
            for graphpath, spxpath in zip(graphs_path_src, spx_label_path_src):
                for file in img_files:
                    if not os.path.isfile(f'{graphpath}/{file}.json') or not os.path.isfile(f'{spxpath}/{file}.json'):
                        create_graphs = True
                        break
                if create_graphs:
                    break

            if(create_graphs):
                out_path = f'{self.model_dir}/scales/'
                os.makedirs(out_path, exist_ok=True)
                self.functions.create_graphs(images_folder, self.final_spx_list, out_path, self.graph_feats, self.spx_alg, self.init_seeds, use_mask)
                for spx in self.final_spx_list:
                    shutil.copytree(os.path.join(out_path, f'{spx}', 'layer0'), os.path.join(out_path, f'{spx}', 'graphs'), dirs_exist_ok=True)
                    shutil.copytree(os.path.join(out_path, f'{spx}', 'labels0'), os.path.join(out_path, f'{spx}', 'labels'), dirs_exist_ok=True)
                    shutil.rmtree(os.path.join(out_path, f'{spx}', 'layer0'), ignore_errors=True)
                    shutil.rmtree(os.path.join(out_path, f'{spx}', 'labels0'), ignore_errors=True)
            else:
                # copy the graphs and labels from the provided directories to the model directory
                for graphpath, spxpath, spx in zip(graphs_path_src, spx_label_path_src, self.final_spx_list):
                    self.__copyGraphsAndLabels__(images_folder, graphpath, spxpath)
                    os.makedirs(f'{self.model_dir}/scales/{spx}', exist_ok=True)
                    # rename the folder name
                    shutil.copytree(f'{self.model_dir}/labels0', f'{self.model_dir}/scales/{spx}/labels0', dirs_exist_ok=True)
                    shutil.copytree(f'{self.model_dir}/layer0', f'{self.model_dir}/scales/{spx}/layer0', dirs_exist_ok=True)
                    shutil.rmtree(f'{self.model_dir}/labels0', ignore_errors=True)
                    shutil.rmtree(f'{self.model_dir}/layer0', ignore_errors=True)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train and extract GFLIM-BoFP model.")

    # madatory arguments
    parser.add_argument("dataset_path", help="Path to the dataset or a csv file with the images paths. " \
                                            "The dataset directory must have a folder 'images' with the original images, " \
                                            "a path 'superpixels/spx_alg/final_spx/graph_feats/graphs' with graphs, "
                                            "a path 'superpixels/spx_alg/final_spx/graph_feats/labels' with superpixel images, "
                                            "and a folder 'masks' (optional) with binary masks for each image, in which black pixels are ignored.", type=str) 
                                        # example: /home/isabela/Codes/LibIFT/Datasets/schisto

    parser.add_argument("final_spx", help="Final number of superpixels (it can be a list in the form 1,2,3).", type=list_of_ints) # example: 10000
    parser.add_argument("markers_path", help="Path to the image markers or a csv file with the file paths.", type=str) # example: interactive_markers
    parser.add_argument("arch_path", help="Architecture file path.", type=str)

    parser.add_argument("train_val", help="Whether to train or validate the model.", type=str, choices=['train', 'extract'])
    parser.add_argument("model_dir", help="Output directory to save the model.", type=str)

    # required when spx_alg is 'DISF'
    parser.add_argument("--init_seeds", help="Initial number of seeds.", type=int, default=20000) 

    # optional arguments
    parser.add_argument("--layers_activations", help="Layers activations to use.", type=list_of_ints, default=[])
    parser.add_argument("--layers_decode", help="Layers to decode.", type=list_of_ints, default=[])
    parser.add_argument("--merge_option", help="Merge option to use with multi-scale GFLIM.", type=str, default='dotprod', choices=['dotprod', 'dotprod_sumscale0', 'dotprod_sumscalei', 'sum'])
    parser.add_argument("--spx_alg", help="Superpixel algorithm to use.", type=str, default='DISF', choices=['DISF', 'ERGC','SH'])
    parser.add_argument("--graph_feats", help="Graph features to use.", type=str, default='SUM', choices=['SUM', 'MEAN'])
    parser.add_argument("--aggregation", help="Aggregation method to use.", type=str, default='CONCAT', choices=['CONCAT', 'WCONCAT'])
    parser.add_argument("--stride", help="Stride type to use.", type=str, default='graph-based')
    parser.add_argument("--attention", help="Attention mechanism to use.", type=str, default='NONE', choices=['NONE', 'CLASS_DOMINANCE'])
    parser.add_argument("--dropout", help="Dropout rate to use.", type=int, default=0, choices=[0, 1])
    parser.add_argument("--batch_size", help="Batch size for training.", type=int, default=64)
    parser.add_argument("--decoders", help="Decoders to use.", type=list_of_str, default=['hybrid', 'graph-adaptive'])
    parser.add_argument("--save_intermediate", help="Whether to save intermediate results (folders layerX and labelsX in model_dir).", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_mask", help="Whether to use mask files. Only if spx_alg is 'DISF'", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    if(os.path.exists(DATASET_PATH)):
        DS_NAME = DATASET_PATH.split('/')[-1]
    else:
        DS_NAME = DATASET_PATH.split('/')[-2] # remove the filename and the parent directory
    
    SPX_ALG = args.spx_alg
    INIT_SEEDS = args.init_seeds
    FINAL_SPX = args.final_spx
    GRAPH_FEATS = args.graph_feats
    AGGREGATION = args.aggregation
    MARKERS_PATH = args.markers_path
    STRIDE = args.stride
    ATTENTION = args.attention
    DROPOUT = args.dropout
    ARCH_PATH = args.arch_path
    BATCH_SIZE = args.batch_size
    TRAIN_VAL = args.train_val
    SAVE_INTERMEDIATE = args.save_intermediate
    LAYERS_ACTIVATIONS = args.layers_activations
    LAYERS_DECODE = args.layers_decode
    DECODERS = args.decoders
    MODEL_DIR = args.model_dir

    if SPX_ALG != 'SH' and len(FINAL_SPX) > 1:
        print("Superpixel algorithm must be SH when FINAL_SPX contains more than 1 element.")
        exit(1)

    print(f'Running GFLIMModelExample with ARCH_PATH={ARCH_PATH} and OUTPUT_DIR={MODEL_DIR}')

    graphs_path_src, spx_label_path_src = None, None

    graphs_out_dir = DATASET_PATH
    if(os.path.isfile(DATASET_PATH)):
        graphs_out_dir = os.path.dirname(DATASET_PATH)

    if len(FINAL_SPX) == 1:
        model = GFLIMModel(MODEL_DIR, ARCH_PATH,
                       MARKERS_PATH,
                       SPX_ALG, INIT_SEEDS, FINAL_SPX[0], GRAPH_FEATS,
                       STRIDE, ATTENTION, AGGREGATION, DROPOUT)
        graphs_path_src,spx_label_path_src = model.CreateGraphsAndLabels(DATASET_PATH, FINAL_SPX, GRAPH_FEATS, SPX_ALG, INIT_SEEDS, graphs_out_dir, use_mask=args.use_mask)

    else:
        if len(FINAL_SPX) < 2:
            raise ValueError("FINAL_SPX must contain at least 2 elements when model_name is 'gflim-BoFP-multiscale'.")

        model = GFLIMultiScaleUModel(MODEL_DIR, ARCH_PATH,
                       MARKERS_PATH,
                       SPX_ALG, FINAL_SPX, GRAPH_FEATS,
                       STRIDE, ATTENTION, AGGREGATION, DROPOUT, merge_option=args.merge_option)
        graphs_path_src,spx_label_path_src = model.CreateGraphsAndLabels(DATASET_PATH, FINAL_SPX, GRAPH_FEATS, SPX_ALG, INIT_SEEDS, graphs_out_dir, use_mask=args.use_mask)
        
    if args.train_val == 'train':
        model.run_model(DATASET_PATH, graphs_path_src, spx_label_path_src,
                    'train', save_intermediate=SAVE_INTERMEDIATE, layers_activations=LAYERS_ACTIVATIONS,
                    layers_decode=LAYERS_DECODE, decoders=DECODERS, batchsize=BATCH_SIZE,
                    use_mask=args.use_mask)

    if args.train_val == 'extract':
        graphs_path_src,spx_label_path_src = model.CreateGraphsAndLabels(DATASET_PATH, FINAL_SPX, GRAPH_FEATS, SPX_ALG, INIT_SEEDS, graphs_out_dir, use_mask=args.use_mask)
        model.run_model(DATASET_PATH, graphs_path_src, spx_label_path_src,
                    'extract', save_intermediate=SAVE_INTERMEDIATE, layers_activations=LAYERS_ACTIVATIONS,
                    layers_decode=LAYERS_DECODE, decoders=DECODERS, batchsize=BATCH_SIZE,
                    use_mask=args.use_mask)