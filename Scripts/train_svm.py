import os
import sys
import argparse
from setup import *

def parse_args():
    parse = argparse.ArgumentParser("GFLIM-BoFP Classification Training Script")
    parse.add_argument("train_path", type=str, help="Train path")
    parse.add_argument("activations_dir", type=str, help="Activations directory with graph files")
    parse.add_argument("model_dir", type=str, help="model directory")
    parse.add_argument("svm_file", type=str, help="SVM model file to save")
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

def train_svm(args):

        if args.sorted_vector == 0:
            # transform mimg images into a ift dataset
            call_func(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{args.activations_dir} {args.train_path} 1", 
                    f"Transform mimg images into a ift dataset\n")

        else:
            # transform mimg images into a ift dataset
            call_func(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{args.activations_dir} {args.train_path} 1 1",
                  f"Transform mimg images into a ift dataset\n")
        
        # train a SVM classifier
        call_func(f"{EXEC_DIR}/iftSupTrainBySVM",
                  f"{args.activations_dir}.zip 0 0 1e2 0 {args.svm_file}",
                  f"SVM training ...\n")
        # -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train_svm(args)