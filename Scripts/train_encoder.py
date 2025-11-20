import os
import sys
import argparse
from setup import *

def parse_args():
    parse = argparse.ArgumentParser("GFLIM-BoFP Classification Training Script")
    parse.add_argument("train_path", type=str, help="train path")
    parse.add_argument("arch_name", type=str, help="Architecture name")
    parse.add_argument("markers_folder", type=str, help="markers folder")
    parse.add_argument("model_dir", type=str, help="model directory")
    parse.add_argument("--spx_alg", type=str, help="Superpixel algorithm", default="DISF")
    parse.add_argument("--final_spx", type=str, help="Final superpixels", default="50")
    parse.add_argument("--stride", type=str, help="Technique for distance calculation", 
                       choices=["graph-based", "none"], default="graph-based")
    parse.add_argument("--node_feat", type=str, help="Node feature", default="MEAN")
    parse.add_argument("--init_seeds", type=str, help="Initial seeds", default="200")
    parse.add_argument("--nlayers", type=int, help="Number of layers", default=3)
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
                f"train {args.train_path} {args.model_dir} " \
                f"{args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} " \
                f"--stride {args.stride} " \
                f"--attention NONE " \
                f"--aggregation CONCAT " \
                f"--arch_name {args.arch_name} " \
                f"--markers_dir {args.markers_folder} " \
                f"--init_seeds {args.init_seeds} " \
                "--dropout 0 " \
                "--batch_size 64 " \
                "--save_intermediate 1 " \
                "--use_mask 1 " \
                f"--layers {','.join([str(l) for l in range(1, args.nlayers + 1)])}", 
                f"Training GFLIM-BoF model for layer {args.nlayers}...\n")

        # move train layer files to avoid overwrite during encoding
        call_func("cp -r", f"{args.model_dir}/layer{args.nlayers} {args.model_dir}/layer{args.nlayers}_train", 
                  f"Copying layer {args.nlayers} for training...\n")
        call_func("cp -r", f"{args.model_dir}/labels{args.nlayers} {args.model_dir}/labels{args.nlayers}_train", 
                  f"Copying labels for layer {args.nlayers} for training...\n")

        """
        if args.sorted_vector == 0:
            # transform mimg images into a ift dataset
            call_func(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{args.model_dir}/layer{args.nlayers}_train {args.train_path} 1", 
                    f"Training SVM for layer {args.nlayers}...\n")

        else:
            # transform mimg images into a ift dataset
            call_func(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{args.model_dir}/layer{args.nlayers}_train {args.train_path} 1 1",
                  f"Transform mimg images into a ift dataset\n")
        
        # train a SVM classifier
        call_func(f"{EXEC_DIR}/iftSupTrainBySVM",
                  f"{args.model_dir}/layer{args.nlayers}_train.zip 0 0 1e2 0 {args.model_dir}/svm_{args.nlayers}.zip",
                  f"SVM training for layer {args.nlayers}...\n")
        """
        # -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train_gflim_bofp(args)