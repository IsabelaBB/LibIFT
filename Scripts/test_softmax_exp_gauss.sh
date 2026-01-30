#!/bin/bash 

#OAR -l cpu=4,walltime=36:00:00

# chmod u+x ~/LibIFT/Scripts/build_basic_params2.sh
# oarsub -n MEAN_s2_32k_300 -S ~/LibIFT/Scripts/MEAN_s2_32k_300.sh

#if [ $# -lt 5 ]; then 
#    echo "Invalid number of arguments. Usage: bash build_basic_params.sh <split> <node_feat> <init_seeds> <superpixels_list> <arch_name>"
#    exit -1
#fi

cd ~/LibIFT/Scripts; 

IFT_dir="/home/iborlido/LibIFT"

run_exp () {
   
    model_dir="${IFT_dir}/Results/gflim_basic_params/helminth-eggs/test_softmax/split1/16kernels_softmax_expGauss${sigma}_relu/1_image/"
    arch_name="9-1-1-16-softmax-${sigma}-125_3_noSkip"

    echo "sigma: ${sigma}"

    python3 GFLIMBoFEncoder.py train \
        ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/train1.csv \
        ${model_dir} \
        DISF 250 \
        --graph_feats MEAN \
        --stride graph-based \
        --attention NONE \
        --aggregation CONCAT \
        --arch_name ${arch_name} \
        --markers_dir ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/markers1 \
        --init_seeds 300 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers 1,2,3 \
        --layers_activations 1,2,3

    python3 GFLIMBoFEncoder.py encode \
        ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/train-val1.csv \
        ${model_dir} \
        DISF 250 \
        --graph_feats MEAN \
        --stride graph-based \
        --attention NONE \
        --aggregation CONCAT \
        --arch_name ${arch_name} \
        --markers_dir ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/markers1 \
        --init_seeds 300 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers 1,2,3

    for layer in 1 2 3; do
        activations_dir="${model_dir}/layer${layer}_trainSVM"
        svm_file="${model_dir}/svm_${layer}.zip"
        cp -r ${model_dir}/layer${layer} ${activations_dir}

        ${IFT_dir}/bin/iftGFLIMActivDataSet \
            ${activations_dir} \
            ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/train-val1.csv 1 1

        ${IFT_dir}/bin/iftSupTrainBySVM \
            ${activations_dir}.zip 0 0 1e2 0 ${svm_file}
    done;

    rm -r ${activations_dir}
    rm -f ${activations_dir}.zip
    for layer in 1 2 3; do
        rm -r ${model_dir}/layer${layer}
        rm -r ${model_dir}/labels${layer}
    done;

    python3 GFLIMBoFEncoder.py encode \
        ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/test1.csv \
        ${model_dir} \
        DISF 250 \
        --graph_feats MEAN \
        --stride graph-based \
        --attention NONE \
        --aggregation CONCAT \
        --arch_name ${arch_name} \
        --markers_dir ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/markers1 \
        --init_seeds 300 \
        --save_intermediate 1 \
        --use_mask 1 \
        --layers 1,2,3
        
    for layer in 1 2 3; do
        activations_dir="${model_dir}/layer${layer}_test"
        svm_file="${model_dir}/svm_${layer}.zip"
        cp -r ${model_dir}/layer${layer} ${activations_dir}

        ${IFT_dir}/bin/iftGFLIMActivDataSet \
            ${activations_dir} \
            ${IFT_dir}/Datasets/intestinal-parasites/helminth-eggs/build/test1.csv 1 1

        ${IFT_dir}/bin/iftClassifyBySVM \
                    ${activations_dir}.zip \
                    ${svm_file} \
                    ${model_dir}/results_layer${layer}_test
    done;
}

for sigma in 1 3 9; do
    run_exp
done;