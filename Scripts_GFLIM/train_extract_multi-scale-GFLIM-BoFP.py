import os
import json
import sys
import shutil
import pickle
from train_extract_GFLIMBoFP import *
from PIL import Image
import numpy as np
import heapq

def CallFunction(function, args, msg=None):
    if(msg is not None):
        sys.stdout.write(msg)
        sys.stdout.flush()
    os.system("{} {}".format(function, " ".join([str(i) for i in args])))


def train_layer(model_dir, arch_file, layer, stride, attention, dropout):
    CallFunction(functions['CreateLayerModel'], [model_dir, arch_file, layer, stride, attention, AGGREGATION], "Creating weights layer {}:\t".format(layer))
    
    CallFunction(functions['EncodeLayer'], [model_dir, arch_file, layer, stride, attention, AGGREGATION], "Encoding layer {}:\t\t".format(layer))
    CallFunction(functions['WriteActivations'], [model_dir, layer], "Writing activations layer {}:\t".format(layer))
    
    #CallFunction(functions['MergeLayerModels'], [model_dir, layer], "Merging weights layer {}:\t".format(layer))
    #CallFunction(functions['EncodeMergedLayer'], [model_dir, arch_file, layer, stride, attention, AGGREGATION, 1 if dropout else 0], "Encoding layer {}:\t\t".format(layer))

    #for d in ["hybrid", "graph-adaptive", "gaussian", "test", "tristate"]:
    #    CallFunction(functions['DecodeLayer'], [model_dir, layer, arch_file, d, stride], "Decoding layer {}:\t\t".format(layer))

def copyGraphsAndLabels(markers_dir, model_dir, graphs_path_src):
    if os.path.exists(os.path.join(model_dir, 'layer0')):
        shutil.rmtree(os.path.join(model_dir, 'layer0'), ignore_errors=True)
    if os.path.exists(os.path.join(model_dir, 'labels0')):
        shutil.rmtree(os.path.join(model_dir, 'labels0'), ignore_errors=True)

    os.makedirs(os.path.join(model_dir, 'layer0'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'labels0'), exist_ok=True)

    for file_name in os.listdir(markers_dir):

        if file_name.endswith('-seeds.txt'):
            graph_name = file_name.split('-')[0]
        else:
            graph_name = file_name.split('.')[0]

        graph_src_file = os.path.join(graphs_path_src, 'graphs', f'{graph_name}.json')
        if not os.path.exists(graph_src_file):
            print(f"Graph source file {graph_src_file} does not exist.")
            exit()
        if not os.path.isfile(os.path.join(model_dir, 'layer0', f'{graph_name}.json')):
            os.symlink(graph_src_file, os.path.join(model_dir, 'layer0', f'{graph_name}.json'))

        labels_src_file = os.path.join(graphs_path_src, 'labels', f'{graph_name}.pgm')
        if not os.path.exists(labels_src_file):
            print(f"Labels source file {labels_src_file} does not exist.")
            exit()
        if not os.path.isfile(os.path.join(model_dir, 'labels0', f'{graph_name}.pgm')):
            os.symlink(labels_src_file, os.path.join(model_dir, 'labels0', f'{graph_name}.pgm'))

def train_model(model_dir, markers_dir, arch_file, nlayers, nfeat_markers, kernel_size, stride, attention, dropout):
    CallFunction(functions['ConvertSeeds'], [model_dir, markers_dir, nfeat_markers, kernel_size, attention, AGGREGATION], 'Converting seeds:\t\t')
    for layer in range(1,nlayers+1):
        train_layer(model_dir, arch_file, layer, stride, attention, dropout)

def read_architecture(arch_file):
    with open(arch_file) as f:
        data = json.load(f)
        kernel_size = data['layer1']['conv']['kernel_size']
        nfeat_markers = data['layer1']['conv']["nkernels_per_marker"]
        nlayers = data['nlayers']
    return kernel_size, nfeat_markers, nlayers

def decoder(model_dir, arch_file, layer, d, stride):
    CallFunction(functions['DecodeLayer'], [model_dir, layer, arch_file, d, stride], "Decoding layer {}:\t\t".format(layer))

def extract(model_dir, img_dir, graphs_path_src, arch_file, nlayers, batch_size, stride, attention, dropout):
    vals = [img_dir, graphs_path_src+'/graphs', graphs_path_src+'/labels', model_dir, arch_file, nlayers, GRAPH_FEATS, SPX_ALG, FINAL_SPX, batch_size, stride, attention, AGGREGATION, 1 if dropout else 0, INIT_SEEDS]
    line = "{} {}".format("{}/iftGFLIMExtract2".format(path), " ".join([str(i) for i in vals]))
    os.system(line)
    #print("Running command: ", line)

def run_model_parallel(arch_file, split_id, graphs_dir, stride, attention, dropout):
    
    kernel_size, nfeat_markers, nlayers = read_architecture(arch_file)

    if stride == 'graph-based':
        STRIDE_NAME = 'graphStride'
    elif stride == 'none':  
        STRIDE_NAME = 'noStride'
    else:
        print(f"Unknown stride type: {stride}")
        exit()
    
    tmp = 'kmeans'
    if dropout:
        tmp += '-dropout'
    spx_str = ''
    for spx in FINAL_SPX_LIST:
        spx_str += f'{spx}-'
    spx_str = spx_str[:-1]  # Remove the last dash

    model_dir = f'/home/isabela/Codes/LibIFT/Results/{DS_NAME}/split{split_id}/{MARKERS_NAME}/{arch_name}/gflim-BoFP/{STRIDE_NAME}/{SPX_ALG}/{spx_str}/{tmp}/{GRAPH_FEATS}-{attention}'
    markers_dir = f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{split_id}'
    
    for i,spx in enumerate(FINAL_SPX_LIST):

        if i == 0:
            continue

        graphs_dir = os.path.join(graphs_dir, f'{spx}', GRAPH_FEATS)
        if not os.path.exists(graphs_dir):
            print(f"Graph features directory {graphs_dir} does not exist.")
            exit()

        copyGraphsAndLabels(f'{DATASET_PATH}/interactive_splits/split{split_id}/orig/train', model_dir, graphs_dir)
        train_model(model_dir, markers_dir, arch_file, nlayers, nfeat_markers, kernel_size, stride, attention, dropout)
        
        #os.rename(f'{model_dir}/layer0', f'{model_dir}/layer0{i}')
        #os.rename(f'{model_dir}/layer1', f'{model_dir}/layer1{i}')
        #os.rename(f'{model_dir}/labels0', f'{model_dir}/labels0{i}')
        #os.rename(f'{model_dir}/labels1', f'{model_dir}/labels1{i}')
        os.rename(f'{model_dir}/activationNoMerge1', f'{model_dir}/activationNoMerge1{i}')
        os.rename(f'{model_dir}/graphadaptive_1', f'{model_dir}/graphadaptive_1{i}')
        os.rename(f'{model_dir}/hybrid_1', f'{model_dir}/hybrid_1{i}')
        os.rename(f'{model_dir}/gaussian_1', f'{model_dir}/gaussian_1{i}')
        os.rename(f'{model_dir}/test_1', f'{model_dir}/test_1{i}')

        shutil.rmtree(f'{model_dir}/labels0', ignore_errors=True)
        shutil.rmtree(f'{model_dir}/layer0', ignore_errors=True)
        shutil.rmtree(f'{model_dir}/labels1', ignore_errors=True)
        shutil.rmtree(f'{model_dir}/layer1', ignore_errors=True)


path = '/home/isabela/Codes/LibIFT/bin'
functions = {
        'CreateGraph': '{}/iftCreateGFLIMGraph'.format(path),
        'ConvertSeeds': '{}/iftGFLIMConvertSeeds'.format(path),
        'ConvertSeeds2': '{}/iftGFLIMConvertSeeds_v2'.format(path),
        'CreateLayerModel': '{}/iftGFLIMCreateLayerModel'.format(path),
        'EncodeLayer': '{}/iftGFLIMEncodeLayer'.format(path),
        'WriteActivations': '{}/iftGFLIMWriteActivations'.format(path),
        'MergeLayerModels': '{}/iftGFLIMMergeLayerModels'.format(path),
        'EncodeMergedLayer': '{}/iftGFLIMEncodeMergedLayer'.format(path),
        'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path)
    }

DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/cells2'
DS_NAME = DATASET_PATH.split('/')[-1]
SPX_ALG = 'SH'
FINAL_SPX_LIST = [20000, 40000, 80000]
GRAPH_FEATS = 'SUM'
AGGREGATION = 'CONCAT'
arch_file = f'{DATASET_PATH}/archs/arch_schisto_test.json'

for spx in FINAL_SPX_LIST:
    graphs_path_src = os.path.join(DATASET_PATH, 'superpixels', SPX_ALG, f'{spx}', GRAPH_FEATS)
    if not os.path.exists(graphs_path_src):
        print(f"Graph features directory {graphs_path_src} does not exist.")
        #exit()


MARKERS_NAME = 'test'

STRIDE = 'graph-based'
ATTENTION = 'NONE'
DROPOUT = False
graphs_path_src = os.path.join(DATASET_PATH, 'superpixels', SPX_ALG)
run_model_parallel(2, 1, graphs_path_src, STRIDE, ATTENTION, DROPOUT)


