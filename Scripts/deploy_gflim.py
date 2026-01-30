import os
import sys
import argparse
from setup import *

def parse_args():
    parse = argparse.ArgumentParser("GFLIM-BoFP Classification Classify Script")
    parse.add_argument("encode_path", type=str, help="encode path")
    parse.add_argument("arch_name", type=str, help="Architecture name")
    parse.add_argument("model_dir", type=str, help="model directory")
    parse.add_argument("--spx_alg", type=str, help="Superpixel algorithm", default="DISF")
    parse.add_argument("--final_spx", type=str, help="Final superpixels", default="50")
    parse.add_argument("--stride", type=str, help="Technique for distance calculation", 
                       choices=["graph-based", "none"], default="graph-based")
    parse.add_argument("--node_feat", type=str, help="Node feature", default="MEAN")
    parse.add_argument("--init_seeds", type=str, help="Initial seeds", default="200")
    parse.add_argument("--nlayers", type=int, help="Number of layers", default=3)
    parse.add_argument("--split_set", type=str, help="Split set to classify", default="val")
    parse.add_argument("--sorted_vector", type=int, help="Use sorted vector for training", 
                       default=0, choices=[0,1])
    args = parse.parse_args()
    return args

# ==================================================

def call_func(function, args, msg=None):
    if(msg is not None):
        sys.stdout.write(msg)
        sys.stdout.flush()
    os.system("{} {}".format(function, args))

def train_gflim_bofp(args):
        #for layer in range(1, args.nlayers + 1):
        
        # train GFLIM-BoF model
        call_func(f"python3 GFLIMBoFEncoder.py", 
                f"encode {args.encode_path} {args.model_dir} " \
                f"{args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} " \
                f"--stride {args.stride} " \
                f"--attention NONE " \
                f"--aggregation CONCAT " \
                f"--arch_name {args.arch_name} " \
                f"--init_seeds {args.init_seeds} " \
                "--dropout 0 " \
                "--batch_size 64 " \
                "--save_intermediate 1 " \
                "--use_mask 1 " \
                f"--layers {args.nlayers}", 
                f"Encoding GFLIM-BoF model for layer {args.nlayers}...\n")

if __name__ == "__main__":
    args = parse_args()
    train_gflim_bofp(args)