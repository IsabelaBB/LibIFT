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
It uses 50% of the data for training + validation and 50% for testing
Train GFLIM models with 1 to 5 images per class and train a SVM in the train-val set.
The SVM classification is evaluated in the test set after extracting features using GFLIM.

The images are randomly selected at each iteration. 
The images selection is done using several random indices.
"""

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
    parser.add_argument("--log_file", type=str, help="Log file to store the results", default=None)

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

def experiment_pipeline(args, random_state):
    """Runs the experiment pipeline for a given technique and random_state."""
    nimages = 1
    best_result = [0.0, 0.0, 0]  # best kappa, global accuracy, nimages
    print(f"🔬 Experiment with random state {random_state}, split {args.split}")

    # same parent dir to summary the results
    results_path = os.path.join(RESULTS_DIR, 'gflim_iterative_pipeline_050trainSVM', 
                        args.dataset_path.split('/')[-1], 
                        f"GFLIMBoFP-{args.arch_name}-{args.stride}-{args.spx_alg}-{args.init_seeds}-{args.final_spx}-{args.node_feat}-{args.nclass}classes-sorted{args.sorted_vector}_{args.nlayers}layer",
                        f'split{args.split}', 
                        f"seed{random_state}")
    
    # copy build folder to results_path to have a copy of the img. sets and markers
    if os.path.isdir(os.path.join(results_path, 'build')):
        shutil.rmtree(os.path.join(results_path, 'build'), ignore_errors=True)
    shutil.copytree(os.path.join(args.dataset_path, 'build'), 
                    os.path.join(results_path, 'build'), dirs_exist_ok=True)

    if not os.path.exists(os.path.join(results_path, 'build', 'superpixels')):
        os.symlink(os.path.join(args.dataset_path, 'superpixels'), 
            os.path.join(results_path, 'build', 'superpixels'),
            target_is_directory=True)

    img_gflim_train_file = os.path.join(results_path, 'build', f'train{args.split}.csv')
    img_svm_deploy_file = os.path.join(results_path, 'build', f'test{args.split}.csv')
    img_svm_train_file = os.path.join(results_path, 'build', f'train-val{args.split}.csv')

    # initialize results file
    results_file = os.path.join(results_path, f'results{args.split}.csv')
    if not os.path.isfile(results_file):
        header = "superpixel,img_per_class,seed,"
        for c in range(1, args.nclass + 1):
            header += f'class{c}_accuracy,'
        header = header + "kappa,global_accuracy,nfeats\n"  
        with open(results_file, 'w') as f:
            f.write(header)

    while nimages <= args.nimages_final:

        model_dir = os.path.join(results_path, f'{nimages}_images')

        # Image selection
        run_script('choose_images_gflim_train.py', 
                f"{results_path} {args.nclass} {args.split} {nimages} {random_state}", 
                f"Image selection", type = "python")

        imgs_file = f"{results_path}/build/train{args.split}.csv"
        # count the number of images in the file
        if os.path.isfile(imgs_file):
            with open(imgs_file, 'r') as f:
                selected_images = f.readlines()
            nselected_images = len(selected_images)
            if nselected_images != nimages * args.nclass:
                raise ValueError(f"Error: Expected {nimages * args.nclass} images, but found {nselected_images} images in {imgs_file}.")
        else:
            raise FileNotFoundError(f"Error: File {imgs_file} does not exist.")

        # create marker files
        markers_folder = os.path.join(results_path, 'build', f'markers{args.split}')
        run_script('CreateSeedsFromAllPix_TrainSet.py', 
                f"{img_gflim_train_file} " \
                f"{os.path.join(args.dataset_path, 'masks')} " \
                f"{markers_folder}", 
                "Creating marker files", type="python")

        # count the number of files in the markers directory
        if os.path.isdir(markers_folder):
            if len(os.listdir(markers_folder)) != nimages * args.nclass:
                raise ValueError(f"Error: Expected {nimages * args.nclass} marker files, but found {len(os.listdir(markers_folder))} files in {markers_folder}.")
        else:
            raise FileNotFoundError(f"Error: Markers directory {markers_folder} does not exist.")


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

        nsamples = 0
        with open(img_gflim_train_file, 'r') as f:
            nsamples = len(f.readlines())
        layer_dir = os.path.join(model_dir, f'layer{args.nlayers}')
        labels_dir = os.path.join(model_dir, f'labels{args.nlayers}')
        if not os.path.isdir(layer_dir) or len(os.listdir(layer_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} layer files, " \
                            f"but found {len(os.listdir(layer_dir)) if os.path.isdir(layer_dir) else 0} files in {layer_dir}.")
        if not os.path.isdir(labels_dir) or len(os.listdir(labels_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} label files, " \
                            f"but found {len(os.listdir(labels_dir)) if os.path.isdir(labels_dir) else 0} files in {labels_dir}.")
         
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
                "Deploy GFLIM with SVM train images", type = "python")

        nsamples = 0
        with open(img_svm_train_file, 'r') as f:
            nsamples = len(f.readlines())
        layer_dir = os.path.join(model_dir, f'layer{args.nlayers}')
        labels_dir = os.path.join(model_dir, f'labels{args.nlayers}')
        if not os.path.isdir(layer_dir) or len(os.listdir(layer_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} layer files, " \
                            f"but found {len(os.listdir(layer_dir)) if os.path.isdir(layer_dir) else 0} files in {layer_dir}.")
        if not os.path.isdir(labels_dir) or len(os.listdir(labels_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} label files, " \
                            f"but found {len(os.listdir(labels_dir)) if os.path.isdir(labels_dir) else 0} files in {labels_dir}.")
         
        activations_dir = os.path.join(model_dir, f'layer{args.nlayers}_trainSVM')
        svm_file = os.path.join(model_dir, f'svm_{args.nlayers}.zip')

        shutil.copytree(os.path.join(model_dir, f'layer{args.nlayers}'), 
                        os.path.join(model_dir, f'layer{args.nlayers}_trainSVM'), dirs_exist_ok=True)

        # remove train layer files to avoid overwrite during encoding
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # transform graphs into a ift dataset
        if args.sorted_vector == 0:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{activations_dir} {img_svm_train_file} 1", 
                    f"Transform graphs into a ift dataset")
        else:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{activations_dir} {img_svm_train_file} 1 1",
                  f"Transform graphs into a ift dataset")

        if not os.path.isfile(f"{activations_dir}.zip"):
            raise ValueError(f"Error: Expected {activations_dir}.zip to be created.")

        # train a SVM classifier
        run_script(f"{EXEC_DIR}/iftSupTrainBySVM",
                  f"{activations_dir}.zip 0 0 1e2 0 {svm_file}",
                  f"Train SVM")

        if not os.path.isfile(svm_file):
            raise ValueError(f"Error: Expected SVM file {svm_file} to be created.")

        shutil.rmtree(activations_dir, ignore_errors=True)
        os.remove(f"{activations_dir}.zip")

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
                "Deploy GFLIM with SVM deploy images", type = "python")

        nsamples = 0
        with open(img_svm_deploy_file, 'r') as f:
            nsamples = len(f.readlines())
        layer_dir = os.path.join(model_dir, f'layer{args.nlayers}')
        labels_dir = os.path.join(model_dir, f'labels{args.nlayers}')
        if not os.path.isdir(layer_dir) or len(os.listdir(layer_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} layer files, " \
                            f"but found {len(os.listdir(layer_dir)) if os.path.isdir(layer_dir) else 0} files in {layer_dir}.")
        if not os.path.isdir(labels_dir) or len(os.listdir(labels_dir)) != nsamples:
            raise ValueError(f"Error: Expected {nsamples} label files, " \
                            f"but found {len(os.listdir(labels_dir)) if os.path.isdir(labels_dir) else 0} files in {labels_dir}.")
        
        # rename deploy layer files to avoid overwrite during encoding
        activations_dir = os.path.join(model_dir, f'layer{args.nlayers}_test')
        shutil.copytree(os.path.join(model_dir, f'layer{args.nlayers}'), 
                        os.path.join(model_dir, f'layer{args.nlayers}_test'), 
                        dirs_exist_ok=True)
        for layer in range(0, args.nlayers + 1):
            shutil.rmtree(os.path.join(model_dir, f'layer{layer}'), ignore_errors=True)
            shutil.rmtree(os.path.join(model_dir, f'labels{layer}'), ignore_errors=True)

        # transform graphs into a ift dataset
        if args.sorted_vector == 0:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet", 
                    f"{activations_dir} {img_svm_deploy_file} 2", 
                    f"Transform graphs into a ift dataset")
        else:
            run_script(f"{EXEC_DIR}/iftGFLIMActivDataSet",
                  f"{activations_dir} {img_svm_deploy_file} 2 1",
                  f"Transform graphs into a ift dataset")

        if not os.path.isfile(f"{activations_dir}.zip"):
            raise ValueError(f"Error: Expected {activations_dir}.zip to be created.")

        # deploy SVM classifier
        run_script(f"{EXEC_DIR}/iftClassifyBySVM",
                  f"{activations_dir}.zip " \
                  f"{svm_file} " \
                  f"{model_dir}/results_layer{args.nlayers}_test",
                  f"Classify test images with SVM")

        # compute the number of features in SVM
        file = os.listdir(activations_dir)[0]
        nfeatures = 0
        with open(os.path.join(activations_dir, file), 'r') as f:
            graph = json.load(f)
            nfeatures = graph['num_nodes'] * graph['num_feats']

        shutil.rmtree(activations_dir, ignore_errors=True)
        os.remove(f"{activations_dir}.zip")

        # read the results file and append to results list
        model_results = os.path.join(model_dir, f'results_layer{args.nlayers}_test', 'results.csv')
        if not os.path.isfile(model_results):
            print(f"Warning: Results file {model_results} does not exist.")
            nimages += 1
            continue
            
        with open(model_results, mode='r') as file:
            lines = file.readlines()
            
        # if the number of lines is diff. than 2, print an error message and skip
        if len(lines) != 2:
            print(f"Warning: Results file {model_results} has {len(lines)} lines. Expected 2 lines.")
            nimages += 1
            continue

        # concatenate the lines and convert to float
        values = [float(value.strip()) for value in (lines[0]+lines[1]).split(';')]
        values = [args.final_spx, nimages, random_state] + values + [nfeatures]

        if values[-3] > best_result[0]:
            best_result = [values[-3], values[-2], nimages]

        # write to the main results file
        with open(results_file, 'a') as f:
            f.write(','.join([str(v) for v in values]) + '\n')

        nimages += 1

    
    shutil.rmtree(os.path.join(results_path, 'build', 'superpixels'), 
                  ignore_errors=True)
    
    return best_result

def main():
    # Experiment parameters
    args = parse_args()
    random_indices = [42, 2735, 6854, 7580, 8900, 123, 456, 789, 1011, 1213]
    #random_indices = [42, 8900]
    total_experiments = len(random_indices) * args.nimages_final
    experiments_done = 0

    if not os.path.isdir(os.path.join(args.dataset_path, 'build')):
        raise FileNotFoundError(f"Build folder not found in {args.dataset_path}. Please run the dataset preparation script first.")

    print(f"🚀 Starting {total_experiments} experiments")

    best_result = [0.0, 0.0, 0]  # best kappa, global accuracy, nimages, random_state
    with tqdm(total=total_experiments, position=0, leave=True) as pbar:
        for i, random_state in enumerate(random_indices):
            pbar.set_description_str(f"Random State {random_state}")
            pbar.refresh()

            best_i = experiment_pipeline(
                args,
                random_state=random_state,
            )
            if best_i[0] > best_result[0]:
                best_result = best_i
            experiments_done += args.nimages_final
            percent = int((experiments_done / total_experiments) * 100)
            pbar.n = experiments_done
            
            pbar.set_postfix_str(f"{percent}%, Best Kappa: {best_result[0]:.4f} (Acc: {best_result[1]:.4f}, Images: {best_result[2]})")
            pbar.refresh()

    print("🎉 All experiments completed!")

if __name__ == "__main__":
    main()
