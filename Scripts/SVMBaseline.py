
import os

DS_NAME="helminth-eggs"
init_seeds="200"
final_spx="50"
node_feat="MEAN"
spx_alg="DISF"
graphs_folder=f"/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/{DS_NAME}/superpixels/{spx_alg}/{node_feat}/{init_seeds}-{final_spx}/layer0"
FOLD="3"
model_dir=f"/home/isabela/Codes/LibIFT/Results/{DS_NAME}/fold{FOLD}/SVM_baseline/{spx_alg}-{init_seeds}-{final_spx}-{node_feat}/"

os.makedirs(model_dir, exist_ok=True)

# ==================================================
print("Starting SVM Baseline training...")

images_folder=f"/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/{DS_NAME}/fold{FOLD}_train.csv"
files = []
with open(images_folder, 'r') as f:
    files = f.readlines()
files = [x.strip() for x in files]

os.makedirs(f"{model_dir}/layer0_train/", exist_ok=True)

for file in files:
    basename = os.path.basename(file).split('.')[0]
    graph_name = f"{graphs_folder}/{basename}.json"
    # copy graph files to model dir
    os.system(f"cp {graph_name} {model_dir}/layer0_train/")

# transform mimg images into a ift dataset
os.system(f"/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet "
          f"{model_dir}/layer0_train/ "
          f"{images_folder} "
          f"1")

# train a SVM classifier
os.system(f"/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM "
          f"{model_dir}/layer0_train.zip "
          f"0 0 1e2 0 {model_dir}/svm_0_{FOLD}.zip")

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
os.system(f"cp -r {model_dir}/layer0_train {model_dir}/layer0_train_sort")

# transform mimg images into a ift dataset
os.system(f"/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet "
          f"{model_dir}/layer0_train_sort "
          f"{images_folder} "
          f"1 1")

# train a SVM classifier
os.system(f"/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM "
          f"{model_dir}/layer0_train_sort.zip "
          f"0 0 1e2 0 {model_dir}/svm_0_{FOLD}_sort.zip")

# ==================================================

print("SVM Baseline training completed.")

images_folder=f"/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/{DS_NAME}/fold{FOLD}_val.csv"
files = []
with open(images_folder, 'r') as f:
    files = f.readlines()
files = [x.strip() for x in files]

os.makedirs(f"{model_dir}/layer0_val/", exist_ok=True)

for file in files:
    basename = os.path.basename(file).split('.')[0]
    graph_name = f"{graphs_folder}/{basename}.json"
    # copy graph files to model dir
    os.system(f"cp {graph_name} {model_dir}/layer0_val/")

# transform mimg images into a ift dataset
os.system(f"/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet "
          f"{model_dir}/layer0_val/ "
          f"{images_folder} "
          f"2")

# test SVM classifier
os.system(f"/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM "
          f"{model_dir}/layer0_val.zip "
          f"{model_dir}/svm_0_{FOLD}.zip "
          f"{model_dir}/results_0_{FOLD}_val")

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
os.system(f"cp -r {model_dir}/layer0_val {model_dir}/layer0_val_sort")

# transform mimg images into a ift dataset
os.system(f"/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet "
          f"{model_dir}/layer0_val_sort "
          f"{images_folder} "
          f"2 1")

# test SVM classifier
os.system(f"/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM "
          f"{model_dir}/layer0_val_sort.zip "
          f"{model_dir}/svm_0_{FOLD}_sort.zip "
          f"{model_dir}/results_0_{FOLD}_val_sort")
