import sys
import os
import subprocess
import shutil
from tqdm import tqdm
import time
import argparse
import json
from setup import *

"""
Train a SVM using the superpixel features with no GFLIM
"""

def parse_args():
    parser = argparse.ArgumentParser("GFLIM Iterative Training Pipeline")

    # dataset setup
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("split", type=int, help="Split identifier (integer)")
    parser.add_argument("random_state", type=int, help="Random state for reproducibility")
    
    parser.add_argument("--dataset_name", type=str, help="Dataset name", default="")
    parser.add_argument("--nclass", type=int, default=9, help="Number of classes in the dataset")

    # superpixels setup
    parser.add_argument("--init_seeds", type=int, help="Initial seeds", default=200)
    parser.add_argument("--final_spx", type=list_of_ints, help="Final superpixels", default=[50])
    parser.add_argument("--node_feat", type=str, help="Node feature", default="MEAN")
    parser.add_argument("--spx_alg", type=str, help="Superpixel algorithm", default="DISF")
    
    # svm parameters
    parser.add_argument("--sorted_vector", type=int, help="Use sorted vector for training",
                       default=0, choices=[0,1])
    
    args = parser.parse_args()
    return args

def run_script(script_name, args, description="", type="exec"):
    """Runs a Python script with arguments and returns the result."""

    if type == "exec":
        script_path = os.path.join(EXEC_DIR, script_name)
        cmd = f"{script_path} {args}"
    elif type == "python":
        cmd = f"python3 {script_name} {args}"
    else:
        raise ValueError("run_script: type must be 'exec' or 'python'")

    print(f"🔄 Running: {description} ..", end = " ")
    sys.stdout.flush()
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    print(f"✅ Finished in {elapsed_time:.2f}s")
    return result

def experiment_pipeline(args, results_path, results_file, nspxels, spx_path):
    """Runs the experiment pipeline for a given technique and random_state."""
    best_result = [0.0, 0.0, 0]  # best kappa, global accuracy, nspxels
    print(f"🔬 Experiment with {nspxels} superpixels, random state {args.random_state}, split {args.split}")

    img_svm_deploy_file = os.path.join(results_path, 'build', 'splits', f'test{args.split}.csv')
    img_svm_train_file = os.path.join(results_path, 'build', 'splits', f'train-val{args.split}.csv')

    model_dir = os.path.join(results_path, f"{args.init_seeds}-{nspxels}spx")
    
    # copy the train-val graphs in img_svm_train_file from spx_path/layer0 to model_dir/layer0_trainSVM
    activation_dir = os.path.join(model_dir, 'layer0_trainSVM')
    os.makedirs(activation_dir, exist_ok=True)
    files = []
    with open(img_svm_train_file, 'r') as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    nimages = len(files)
    for file in files:
        basename = os.path.basename(file).split('.')[0]
        os.symlink(f"{spx_path}/layer0/{basename}.json", f"{activation_dir}/{basename}.json")

    # transform graphs into a ift dataset
    if args.sorted_vector == 0:
        run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                f"{activation_dir} {img_svm_train_file} 1", 
                f"Transform graphs into a ift dataset")
    else:
        run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
            f"{activation_dir} {img_svm_train_file} 1 1",
            f"Transform graphs into a ift dataset")

    if not os.path.isfile(f"{activation_dir}.zip"):
        raise ValueError(f"Error: Expected {activation_dir}.zip to be created.")

    svm_file = f"{model_dir}/svm_0_{args.split}.zip"

    # train a SVM classifier
    run_script(f"{EXEC_DIR}/iftSupTrainBySVM",
                f"{activation_dir}.zip 0 0 1e2 0 {svm_file}",
                f"Train SVM")
    
    if not os.path.isfile(svm_file):
        raise ValueError(f"Error: Expected SVM file {svm_file} to be created.")

    shutil.rmtree(activation_dir, ignore_errors=True)
    os.remove(f"{activation_dir}.zip")

    # copy the test graphs in img_svm_deploy_file from spx_path/layer0 to model_dir/layer0_test
    activation_dir = os.path.join(model_dir, 'layer0_test')
    os.makedirs(activation_dir, exist_ok=True)
    files = []
    with open(img_svm_deploy_file, 'r') as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    for file in files:
        basename = os.path.basename(file).split('.')[0]
        os.symlink(f"{spx_path}/layer0/{basename}.json", f"{activation_dir}/{basename}.json")

    # transform graphs into a ift dataset
    if args.sorted_vector == 0:
        run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                f"{activation_dir} {img_svm_deploy_file} 2", 
                f"Transform graphs into a ift dataset")
    else:
        run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                f"{activation_dir} {img_svm_deploy_file} 2 1",
                f"Transform graphs into a ift dataset")

    if not os.path.isfile(f"{activation_dir}.zip"):
        raise ValueError(f"Error: Expected {activation_dir}.zip to be created.")

    # deploy SVM classifier
    run_script(f"{EXEC_DIR}/iftClassifyBySVM",
            f"{activation_dir}.zip " \
            f"{svm_file} " \
            f"{model_dir}/results_layer0_test",
            f"Classify test images with SVM")

    # compute the number of features in SVM
    file = os.listdir(activation_dir)[0]
    nfeatures = 0
    with open(os.path.join(activation_dir, file), 'r') as f:
        graph = json.load(f)
        nfeatures = graph['num_nodes'] * graph['num_feats']

    shutil.rmtree(activation_dir, ignore_errors=True)
    os.remove(f"{activation_dir}.zip")

    # read the results file and append to results list
    model_results = os.path.join(model_dir, f'results_layer0_test', 'results.csv')
    if not os.path.isfile(model_results):
        print(f"Warning: Results file {model_results} does not exist.")
        return best_result
                
    with open(model_results, mode='r') as file:
        lines = file.readlines()
            
    # if the number of lines is diff. than 2, print an error message and skip
    if len(lines) < 2:
        print(f"Warning: Results file {model_results} has {len(lines)} lines. Expected 2 lines.")
        return best_result

    # concatenate the lines and convert to float
    values = [float(value.strip()) for value in (lines[0]+lines[1]).split(';')]
    values = [nspxels, 0, nimages, args.random_state] + values + [nfeatures]

    best_result = [values[-3], values[-2], nspxels]

    # write to the main results file
    with open(results_file, 'a') as f:
        f.write(','.join([str(v) for v in values]) + '\n')

    return best_result

def main():
    # Experiment parameters
    args = parse_args()
    total_experiments = len(args.final_spx)
    experiments_done = 0

    if not os.path.isdir(os.path.join(args.dataset_path)):
        raise FileNotFoundError(f"Folder not found in {args.dataset_path}. Please run the dataset preparation script first.")

    print(f"🚀 Starting {total_experiments} experiments")

    # same parent dir to summary the results
    results_path = os.path.join(RESULTS_DIR, 'gflim_basic_params', 
                        args.dataset_path.split('/')[-1], 
                        f"SVMbaseline-{args.spx_alg}-{args.node_feat}-{args.nclass}classes-sorted{args.sorted_vector}",
                        f'split{args.split}', f'seed{args.random_state}')

    # copy build folder to results_path to have a copy of the img. sets and markers
    if os.path.isdir(os.path.join(results_path, 'build')):
        shutil.rmtree(os.path.join(results_path, 'build'), ignore_errors=True)
    shutil.copytree(os.path.join(args.dataset_path, 'build'), 
                    os.path.join(results_path, 'build'), dirs_exist_ok=True)

    # initialize results file
    results_file = os.path.join(results_path, f'results{args.split}.csv')
    if not os.path.isfile(results_file):
        header = "superpixel,nlayers,img_per_class,seed," # nlayers will be 0 for SVM baseline
        for c in range(1, args.nclass + 1):
            header += f'class{c}_accuracy,'
        header = header + "kappa,global_accuracy,nfeats\n"  
        with open(results_file, 'w') as f:
            f.write(header)
    
    best_result = [0.0, 0.0, 0]  # best kappa, global accuracy, nspxels
    with tqdm(total=total_experiments, position=0, leave=True) as pbar:
        for nspxels in args.final_spx:
            pbar.set_description_str(f"{nspxels} Superpixels")
            pbar.refresh()

            spx_path = os.path.join(args.dataset_path, 'superpixels', args.spx_alg, args.node_feat, f'{args.init_seeds}-{nspxels}')
            if not os.path.isdir(spx_path):
                raise FileNotFoundError(f"Warning: Superpixel path does not exist: {spx_path}")

            best_i = experiment_pipeline(
                args, results_path, results_file, 
                nspxels,
                spx_path
            )
            if best_i[0] > best_result[0]:
                best_result = best_i
            experiments_done += 1
            percent = int((experiments_done / total_experiments) * 100)
            pbar.n = experiments_done

            pbar.set_postfix_str(f"{percent}%, Best Kappa: {best_result[0]:.4f} (Acc: {best_result[1]:.4f}, Spx: {best_result[2]})")
            pbar.refresh()
    
    print("🎉 All experiments completed!")

if __name__ == "__main__":
    main()
