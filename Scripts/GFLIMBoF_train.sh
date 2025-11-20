#!/bin/bash

DS_NAME="helminth-eggs"
ARCH_NAME="9-1-1-50-max-3-125_3_noSkip"

arch_folder="/home/isabela/Codes/LibIFT/Datasets/archs/${ARCH_NAME}.json"
stride="graph-based"
init_seeds="200"
final_spx="50"
node_feat="MEAN"
spx_alg="DISF"

FOLD="$1"
images_folder="/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/${DS_NAME}/fold${FOLD}_train.csv"
model_dir="/home/isabela/Codes/LibIFT/Results/${DS_NAME}/fold${FOLD}/${ARCH_NAME}/gflim-BoFP/${stride}/${spx_alg}/${init_seeds}-${final_spx}/kmeans/${node_feat}-NONE-CONCAT/"
markers_folder="/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/${DS_NAME}/markers/fold${FOLD}/"

# ==================================================
last_layer="1"
# train GFLIM-BoF model
python3 GFLIMBoFEncoder.py train ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --markers_dir ${markers_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train \
    ${images_folder} \
    1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}.zip

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train_sort
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train_sort \
    ${images_folder} \
    1 1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train_sort.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip

# ==================================================
last_layer="2"
# train GFLIM-BoF model
python3 GFLIMBoFEncoder.py train ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --markers_dir ${markers_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train \
    ${images_folder} \
    1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}.zip

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train_sort
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train_sort \
    ${images_folder} \
    1 1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train_sort.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip

# ==================================================

last_layer="3"
# train GFLIM-BoF model
python3 GFLIMBoFEncoder.py train ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --markers_dir ${markers_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train \
    ${images_folder} \
    1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}.zip

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_train_sort
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_train_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_train_sort \
    ${images_folder} \
    1 1

# train a SVM classifier
/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
    ${model_dir}/layer${last_layer}_train_sort.zip \
    0 0 1e2 0 ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip

# ==================================================