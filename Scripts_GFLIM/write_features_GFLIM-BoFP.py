import os
import json
import sys
import shutil

def CallFunction(function, args, msg=None):
    if(msg is not None):
        sys.stdout.write(msg)
        sys.stdout.flush()
    os.system("{} {}".format(function, " ".join([str(i) for i in args])))


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


def read_architecture(arch_file):
    with open(arch_file) as f:
        data = json.load(f)
        kernel_size = data['layer1']['conv']['kernel_size'][0]*data['layer1']['conv']['kernel_size'][1]
        nfeat_markers = data['layer1']['conv']["nkernels_per_marker"]
        nlayers = data['nlayers']
    return kernel_size, nfeat_markers, nlayers


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
        'DecodeLayer': '{}/iftGFLIMDecodeLayer'.format(path),
        'iftWriteSeedsVectors': '{}/iftWriteSeedsVectors'.format(path),
    }

DATASET_PATH = '/home/isabela/Codes/LibIFT/Datasets/schisto'
DS_NAME = DATASET_PATH.split('/')[-1]
MARKERS_NAME = 'interactive_markers'
SPX_ALG = 'DISF'
INIT_SEEDS = 20000
FINAL_SPX = 8000
GRAPH_FEATS = 'SUM'
AGGREGATION = 'CONCAT'


graphs_path_src = os.path.join(DATASET_PATH, 'superpixels', SPX_ALG, f'{INIT_SEEDS}-{FINAL_SPX}', GRAPH_FEATS)
if not os.path.exists(graphs_path_src):
    print(f"Graph features directory {graphs_path_src} does not exist.")
    exit()


def run_model_parallel(arch_id, split_id, graphs_path_src, stride, attention, dropout):
    
    arch_name = f'arch_schisto{arch_id}'
    arch_file = f'{DATASET_PATH}/archs/{arch_name}.json'
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
    model_dir = f'/home/isabela/Codes/LibIFT/Results/{DS_NAME}/split{split_id}/{MARKERS_NAME}/{arch_name}/gflim-BoFP/{STRIDE_NAME}/{SPX_ALG}/{INIT_SEEDS}-{FINAL_SPX}/{tmp}/{GRAPH_FEATS}-{attention}'
    markers_dir = f'{DATASET_PATH}/markers/{MARKERS_NAME}/split{split_id}'
    
    for layer in range(0,nlayers+1):
        shutil.rmtree(f'{model_dir}/labels{layer}', ignore_errors=True)
        shutil.rmtree(f'{model_dir}/layer{layer}', ignore_errors=True)

    copyGraphsAndLabels(f'{DATASET_PATH}/interactive_splits/split{split_id}/orig/train', model_dir, graphs_path_src)
    CallFunction(functions['iftWriteSeedsVectors'], [model_dir, markers_dir, nfeat_markers, kernel_size, attention, AGGREGATION], 'Converting seeds:\t\t')
    
    for layer in range(0,nlayers+1):
        shutil.rmtree(f'{model_dir}/labels{layer}', ignore_errors=True)
        shutil.rmtree(f'{model_dir}/layer{layer}', ignore_errors=True)
    

import multiprocessing
from tqdm import tqdm


STRIDE = 'graph-based'
ATTENTION = 'NONE'
DROPOUT = False

inputs = [(arch_id, split_id, STRIDE, ATTENTION, DROPOUT) for arch_id in [1] for split_id in range(1,6)]

with multiprocessing.Pool() as pool:
    pool.starmap(run_model_parallel, ((arch_id, split_id, graphs_path_src, STRIDE, ATTENTION, DROPOUT) for (arch_id,split_id, STRIDE, ATTENTION, DROPOUT) in tqdm(inputs)))
