
import json
import os
import subprocess
import shutil
from tqdm import tqdm
import time
import argparse
from setup import *

def parse_args():
    parser = argparse.ArgumentParser("GFLIM Iterative Training Pipeline")

    # dataset setup
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("split", type=int, help="Split identifier (integer)")
    
    parser.add_argument("--dataset_name", type=str, help="Dataset name", default="")
    parser.add_argument("--nclass", type=int, default=9, help="Number of classes in the dataset")

    # architecture setup
    parser.add_argument("arch_name", type=str, help="Architecture name")
    parser.add_argument("nlayers", type=int, help="Number of layers")

    # gflim setup
    parser.add_argument("--stride", type=str, help="Technique for distance calculation", choices=["graph-based", "none"], default="graph-based")
    parser.add_argument("--init_seeds", type=int, help="Initial seeds", default=200)
    parser.add_argument("--final_spx", type=int, help="Final superpixels", default=50)
    parser.add_argument("--node_feat", type=str, help="Node feature", default="MEAN")
    parser.add_argument("--spx_alg", type=str, help="Superpixel algorithm", default="DISF")
    
    # results
    parser.add_argument("--nimages_final", type=int, default=5, help="Final number of images to select per class")
    parser.add_argument("--random_states", type=int, default=42, help="Random seed for reproducibility")

    # svm parameters
    parser.add_argument("--sorted_vector", type=int, help="Use sorted vector for training",
                       default=0, choices=[0,1])
    
    args = parser.parse_args()
    return args

def run_script(script_name, args, description="", type="exec"):
    """Runs a Python script with arguments and returns the result."""

    if type == "exec":
        script_path = os.path.join(EXEC_DIR, script_name)
        if script_name == 'iftPrototypeEvaluation':
            print(f"Running iftPrototypeEvaluation from {script_path}")
        cmd = f"{script_path} {args}"
    elif type == "python":
        cmd = f"python3 {script_name} {args}"
        print(f"Running python script: {cmd}")
    else:
        raise ValueError("run_script: type must be 'exec' or 'python'")

    print(f"🔄 Running: {description} - {script_name}")
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    print(f"✅ Finished in {elapsed_time:.2f}s: {description}")
    
    return result

def rename_and_move_results(args, graph_dir, model_dir):
    """Renames result files, adds nfeat, and moves them to the target directory."""
    
    graph_file = os.listdir(graph_dir)[0]
    nfeat = "unknown"
    # open json file
    with open(os.path.join(graph_dir, graph_file), 'r') as f:
        graph = json.load(f)
        nfeat = graph['num_feats'] * graph['num_nodes']

    for layer in range(1, args.nlayers + 1):
        results_dir = f"{model_dir}/../results_{layer}_train-val"
        results_csv = f'results.csv'

        # Append nfeat to results file
        results_path = os.path.join(results_dir, results_csv)
        if os.path.isfile(results_path):
            with open(results_path, 'a') as f:
                f.write(f"\nnfeat: {nfeat}")
        else:
            print(f"Warning: Results file {results_path} does not exist.")

        shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
        shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)
        shutil.rmtree(os.path.join(model_dir, f'layer{layer}_train'), ignore_errors=True)
        shutil.rmtree(os.path.join(model_dir, f'labels{layer}_train'), ignore_errors=True)
        shutil.rmtree(os.path.join(model_dir, f'layer{layer}_train-val'), ignore_errors=True)
        shutil.rmtree(os.path.join(model_dir, f'labels{layer}_train-val'), ignore_errors=True)

    shutil.rmtree(os.path.join(model_dir, f'layer0'), ignore_errors=True)
    shutil.rmtree(os.path.join(model_dir, f'labels0'), ignore_errors=True)

def experiment_pipeline(args, random_state, technique, distance_flag):
    """Runs the experiment pipeline for a given technique and random_state."""
    nimages = 1
    # print(f"🔷 Starting experiments for random_state {random_state} with {technique} distance")
    
    # Initial image selection and training
    run_script('choose_images_gflim_train.py', 
               f"{args.dataset_path} {args.nclass} {args.split} {nimages} {random_state}", 
               f"Image selection (random_state={random_state})", type = "python")

    img_gflim_train_file = os.path.join(args.dataset_path, 'build', f'train{args.split}.csv')
    img_svm_deploy_file = os.path.join(args.dataset_path, 'build', f'test{args.split}.csv')
    img_svm_train_file = os.path.join(args.dataset_path, 'build', f'train-val{args.split}.csv')
    img_train_val_file = os.path.join(args.dataset_path, 'build', f'train-val{args.split}.csv')

    # copy superpixel graphs and images to build folder
    shutil.copytree(os.path.join(args.dataset_path, 'superpixels'), 
                    os.path.join(args.dataset_path, 'build', 'superpixels'), dirs_exist_ok=True)

    # create marker files
    markers_folder = os.path.join(args.dataset_path, 'build', f'markers{args.split}')
    run_script('CreateSeedsFromAllPix_TrainSet.py', 
            f"{img_gflim_train_file} " \
            f"{markers_folder} " \
            f"--mask_dir {os.path.join(args.dataset_path, 'masks')}",
            "Creating marker files", type="python")

    while nimages <= args.nimages_final:

        model_dir = os.path.join(RESULTS_DIR, args.dataset_path.split('/')[-1], 
                            f'models_incremental{random_state}', 
                            f'split{args.split}', 
                            f"gflimBoFP-{technique}-{args.arch_name}-{args.stride}-{args.spx_alg}-{args.init_seeds}-{args.final_spx}-{args.node_feat}",
                            f'{nimages}_images')
    
        # train GFLIM-BoF model
        run_script('GFLIMBoFEncoder.py', 
                f"train {img_gflim_train_file} {model_dir} {args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} --stride {args.stride} " \
                f"--attention NONE --aggregation CONCAT " \
                f"--arch_name {args.arch_name} --markers_dir {markers_folder} " \
                f"--init_seeds {args.init_seeds} --dropout 0 --batch_size 64 " \
                f"--save_intermediate 1 --use_mask 1 " \
                f"--layers {','.join([str(l) for l in range(1, args.nlayers + 1)])}",
                f"Train GFLIM with {nimages} images", type = "python")

        # remove train layer files to avoid overwrite during encoding
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # deploy GFLIM-BoF model with the images to train the svm
        run_script('GFLIMBoFEncoder.py', 
                f"encode {img_svm_train_file} {model_dir} " \
                f"{args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} --stride {args.stride} " \
                f"--attention NONE --aggregation CONCAT " \
                f"--arch_name {args.arch_name} " \
                f"--init_seeds {args.init_seeds} --dropout 0 " \
                f"--batch_size 64 --save_intermediate 1 --use_mask 1 " \
                f"--layers {args.nlayers}",
                "Deploy GFLIM with svm train images", type = "python")

        activations_dir = os.path.join(model_dir, f'layer{args.nlayers}_trainSVM')
        svm_file = os.path.join(model_dir, f'svm_{args.nlayers}.zip')

        shutil.copytree(os.path.join(model_dir, f'layer{args.nlayers}'), 
                        os.path.join(model_dir, f'layer{args.nlayers}_trainSVM'), dirs_exist_ok=True)

        # remove train layer files to avoid overwrite during encoding
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # transform mimg images into a ift dataset
        if args.sorted_vector == 0:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{activations_dir} {img_svm_train_file} 1", 
                    f"Transform mimg images into a ift dataset\n")
        else:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{activations_dir} {img_svm_train_file} 1 1",
                  f"Transform mimg images into a ift dataset\n")
        
        # train a SVM classifier
        run_script(f"{EXEC_DIR}/iftSupTrainBySVM",
                  f"{activations_dir}.zip 0 0 1e2 0 {svm_file}",
                  f"SVM training ...\n")

        shutil.rmtree(activations_dir, ignore_errors=True)

        # deploy GFLIM in test set
        run_script('GFLIMBoFEncoder.py', 
                f"encode {img_svm_deploy_file} {model_dir} " \
                f"{args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} --stride {args.stride} " \
                f"--attention NONE --aggregation CONCAT " \
                f"--arch_name {args.arch_name} " \
                f"--init_seeds {args.init_seeds} --dropout 0 " \
                f"--batch_size 64 --save_intermediate 1 --use_mask 1 " \
                f"--layers {args.nlayers}",
                "Deploy GFLIM with svm deploy images", type = "python")

        # rename deploy layer files to avoid overwrite during encoding
        activations_dir = os.path.join(model_dir, f'layer{args.nlayers}_test')
        shutil.copytree(os.path.join(model_dir, f'layer{args.nlayers}'), os.path.join(model_dir, f'layer{args.nlayers}_test'), dirs_exist_ok=True)
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # transform mimg images into a ift dataset
        if args.sorted_vector == 0:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{activations_dir} {img_svm_deploy_file} 2", 
                    f"Transform mimg images into a ift dataset\n")
        else:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{activations_dir} {img_svm_deploy_file} 2 1",
                  f"Transform mimg images into a ift dataset\n")
        
        # deploy SVM classifier
        run_script(f"{EXEC_DIR}/iftClassifyBySVM",
                  f"{activations_dir}.zip "\
                  f"{svm_file} " \
                  f"{model_dir}/../results_{args.nlayers}_test",
                  f"SVM classification ...\n")

        shutil.rmtree(activations_dir, ignore_errors=True)

        # deploy GFLIM in train-val set for prototype evaluation
        run_script('GFLIMBoFEncoder.py', 
                f"encode {img_train_val_file} {model_dir} " \
                f"{args.spx_alg} {args.final_spx} " \
                f"--graph_feats {args.node_feat} --stride {args.stride} " \
                f"--attention NONE --aggregation CONCAT " \
                f"--arch_name {args.arch_name} " \
                f"--init_seeds {args.init_seeds} --dropout 0 " \
                f"--batch_size 64 --save_intermediate 1 --use_mask 1 " \
                f"--layers {args.nlayers}",
                "Deploy GFLIM with train-val images", type = "python")

        # rename deploy layer files to avoid overwrite during encoding
        activations_dir = os.path.join(model_dir, f'layer{args.nlayers}_train-val')
        shutil.copytree(os.path.join(model_dir, f'layer{args.nlayers}'), os.path.join(model_dir, f'layer{args.nlayers}_train-val'), dirs_exist_ok=True)
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # transform mimg images into a ift dataset
        if args.sorted_vector == 0:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{activations_dir} {img_train_val_file} 1", 
                    f"Transform mimg images into a ift dataset\n")
        else:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{activations_dir} {img_train_val_file} 1 1",
                  f"Transform mimg images into a ift dataset\n")

        missclassified_file = f"{model_dir}/../results_{args.nlayers}_test/missclassified.csv"
        run_script('iftPrototypeEvaluation', 
                   f"{activations_dir}.zip {img_train_val_file} {missclassified_file}", 
                   f"Prototype evaluation ({technique})")
        
        shutil.rmtree(activations_dir, ignore_errors=True)

        if nimages < args.nimages_final:
            run_script('add_images_train_gflim.py', 
                    f"{args.dataset_path} {args.split} " \
                    f"{missclassified_file} " \
                    f"{args.nclass} ", 
                    "Add most similar images",
                    type="python")

            nimages += 1

            print(f"🔄 Re-training SVM with {nimages} images")

            # create marker files
            run_script('CreateSeedsFromAllPix_TrainSet.py', 
                f"{img_gflim_train_file} {markers_folder} --mask_dir {os.path.join(args.dataset_path, 'masks')}", 
                f"Creating marker files for {nimages} images", type="python")
        else:
            nimages += 1

def main():
    # Experiment parameters
    args = parse_args()

    techniques = [("cossine", 0)]
    total_experiments = len([args.random_states]) * len(techniques) * args.nimages_final
    experiments_done = 0

    print(f"🚀 Starting {total_experiments} experiments for Cosine distances")

    with tqdm(total=total_experiments, desc="Experiments", position=0, leave=True) as pbar:
        for i, random_state in enumerate([args.random_states]):
            for j, (technique, flag) in enumerate(techniques):
                experiment_pipeline(
                    args,
                    random_state=random_state,
                    technique=technique,
                    distance_flag=flag
                )
                experiments_done += 1
                percent = int((experiments_done / total_experiments) * 100)
                pbar.n = experiments_done
                pbar.set_postfix_str(f"{percent}%")
                pbar.refresh()
                print(f"Experiments remaining: {total_experiments - experiments_done}")

    print("🎉 All experiments completed!")

if __name__ == "__main__":
    main()
