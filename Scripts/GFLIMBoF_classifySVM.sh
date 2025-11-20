#!/bin/bash

DS_NAME="helminth-eggs"
DS_PATH="/home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/${DS_NAME}"
ARCH_NAME="9-1-1-64-max-3-125_9-1-1-64-max-3-125_9-1-1-64-max-3-125_noSkip"

arch_folder="/home/isabela/Codes/LibIFT/Datasets/archs/${ARCH_NAME}.json"
stride="graph-based"
init_seeds="200"
final_spx="50"
node_feat="SUM"
spx_alg="DISF"
last_layer="3"

FOLD="1"
images_folder="${DS_PATH}/fold${FOLD}_train.csv"
model_dir="/home/isabela/Codes/LibIFT/Results/${DS_NAME}/fold${FOLD}/${ARCH_NAME}/gflim-BoFP/${stride}/${spx_alg}/${init_seeds}-${final_spx}/kmeans/${node_feat}-NONE-CONCAT/"

# transform graphs into mimg images
/home/isabela/Codes/LibIFT/bin/iftGraphToMimg \
    ${model_dir}/layer${last_layer} \
    ${model_dir}/labels${last_layer} \
    ${model_dir}/mimg${last_layer}

# transform mimg images into a ift dataset
/home/isabela/Codes/LibIFT/bin/iftActivDataSet \
    ${model_dir}/mimg${last_layer} \
    ${images_folder} \
    train \
    0 # border size to be cropped (e.g., 5, 10)

# train SVM classifier
#/home/isabela/Codes/LibIFT/bin/iftSupTrainBySVM \
#    /home/isabela/Codes/LibIFT/Results/helminth-eggs/fold1/arch_9-1-1-64-max-3-1.25_9-1-1-64-max-3-1.25_9-1-1-64-max-3-1.zip \
#    0 0 1e2 0 ./svm_3_1.zip