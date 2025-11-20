#!/bin/bash

DS_NAME="helminth-eggs"
ARCH_NAME="9-1-1-50-max-3-125_3_noSkip"
split_set="val"

arch_folder="/home/isabela/Codes/LibIFT/Datasets/archs/${ARCH_NAME}.json"
stride="graph-based"
init_seeds="200"
final_spx="50"
node_feat="MEAN"
spx_alg="DISF"

FOLD="$1"
images_folder="/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/${DS_NAME}/fold${FOLD}_${split_set}.csv"
model_dir="/home/isabela/Codes/LibIFT/Results/${DS_NAME}/fold${FOLD}/${ARCH_NAME}/gflim-BoFP/${stride}/${spx_alg}/${init_seeds}-${final_spx}/kmeans/${node_feat}-NONE-CONCAT/"

last_layer="1"
python3 GFLIMBoFEncoder.py encode ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set} \
    ${images_folder} \
    2

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
mv ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}_sort
mv ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set}_sort \
    ${images_folder} \
    2 1

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}_sort.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}_sort

# ==================================================

last_layer="2"
python3 GFLIMBoFEncoder.py encode ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set} \
    ${images_folder} \
    2

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
mv ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}_sort
mv ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set}_sort \
    ${images_folder} \
    2 1

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}_sort.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}_sort
# ==================================================

last_layer="3"
python3 GFLIMBoFEncoder.py encode ${images_folder} ${model_dir} ${spx_alg} ${final_spx} \
        --graph_feats ${node_feat} \
        --stride ${stride} \
        --attention "NONE" \
        --aggregation "CONCAT" \
        --arch_file ${arch_folder} \
        --init_seeds ${init_seeds} \
        --dropout 0 \
        --batch_size 64 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers ${last_layer}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
cp -r ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}
cp -r ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set} \
    ${images_folder} \
    2

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}

# -------------------------------------------------

# move train layer files to avoid overwrite during encoding
mv ${model_dir}/layer${last_layer} ${model_dir}/layer${last_layer}_${split_set}_sort
mv ${model_dir}/labels${last_layer} ${model_dir}/labels${last_layer}_${split_set}_sort

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftGFLIMActivDataSet \
    ${model_dir}/layer${last_layer}_${split_set}_sort \
    ${images_folder} \
    2 1

# test SVM classifier
/home/isabela/Codes/LibIFT/bin/iftClassifyBySVM \
        ${model_dir}/layer${last_layer}_${split_set}_sort.zip \
        ${model_dir}/svm_${last_layer}_${FOLD}_sort.zip \
        ${model_dir}/results_${last_layer}_${FOLD}_${split_set}_sort

# ==================================================