#!/bin/bash 

#OAR -l cpu=2,walltime=36:00:00

# chmod u+x ~/LibIFT/Scripts/build_basic_params2.sh
# oarsub -n MEAN_s2_32k_300 -S ~/LibIFT/Scripts/MEAN_s2_32k_300.sh

#if [ $# -lt 5 ]; then 
#    echo "Invalid number of arguments. Usage: bash build_basic_params.sh <split> <node_feat> <init_seeds> <superpixels_list> <arch_name>"
#    exit -1
#fi

node_feat=MEAN
ds_name="helminth-eggs"
stride="graph-based"
spx_alg="DISF"
nclasses=9
nlayers=3
nimages=1
random_state=42

cd ~/LibIFT/Scripts; 

for split in 1 2 3; do

for attention in "NONE" "CLASS_DOMINANCE"; do

nkernels=64
arch_name="9-1-1-${nkernels}-max-3-125_3_noSkip_noStride"

init_seeds=200
superpixels_list="175,200"

echo "Split: ${split}"
echo "Node features: ${node_feat}"
echo "Init seeds: ${init_seeds}"
echo "Superpixels list: ${superpixels_list}"
echo "Arch name: ${arch_name}"
echo "Attention: ${attention}"

python3 gflim_pipeline_basic_params.py \
    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
    ${split} ${random_state} ${arch_name} ${nlayers} --dataset_name ${ds_name} --nclass ${nclasses} \
    --stride ${stride} --init_seeds ${init_seeds} --final_spx ${superpixels_list} --node_feat ${node_feat} \
    --spx_alg ${spx_alg} --nimages ${nimages} --sorted_vector 1 --attention ${attention}

init_seeds=300
superpixels_list="250,300"

echo "Split: ${split}"
echo "Node features: ${node_feat}"
echo "Init seeds: ${init_seeds}"
echo "Superpixels list: ${superpixels_list}"
echo "Arch name: ${arch_name}"
echo "Attention: ${attention}"

python3 gflim_pipeline_basic_params.py \
    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
    ${split} ${random_state} ${arch_name} ${nlayers} --dataset_name ${ds_name} --nclass ${nclasses} \
    --stride ${stride} --init_seeds ${init_seeds} --final_spx ${superpixels_list} --node_feat ${node_feat} \
    --spx_alg ${spx_alg} --nimages ${nimages} --sorted_vector 1 --attention ${attention}

done
done