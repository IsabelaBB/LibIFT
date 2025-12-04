#!/bin/bash 

# oarsub -t deploy -r now -q p1 -n name -l walltime=6:00 "bash ~/LibIFT/Scripts/build_baseline_model.sh 1 MEAN"

# install dependencies and compile LibIFT
sudo-g5k apt-get install -y cmake libblas-dev liblapack-dev libopenblas-dev libomp-dev python3-pip # when not using kadeploy3
cd ~/LibIFT/build; cmake ..; make; cp libift.a ../lib/;
export NEWIFT_DIR="~/LibIFT/"
cd ~/LibIFT/demo/Classification; bash compile.sh 0;
cd ~/LibIFT/demo/GFLIM; bash compile.sh 0;

ds_name="helminth-eggs"
init_seeds=200
final_spx=15,25,50,100,150,175,200
spx_alg="DISF"
nclasses=9
random_state=42

split=$1
node_feat=$2

cd ~/LibIFT/Scripts; 
results_dir="/home/iborlido/LibIFT/Results/gflim_basic_params/${ds_name}/SVMbaseline-${spx_alg}-${node_feat}-${nclasses}classes-sorted1/split${split}/";

mkdir -p ${results_dir}

python3 baseline_model.py \
    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
    ${split} ${random_state} --dataset_name ${ds_name} --nclass ${nclasses} \
    --init_seeds ${init_seeds} --final_spx ${final_spx} --node_feat ${node_feat} \
    --spx_alg ${spx_alg} --sorted_vector 1 >> \
    ${results_dir}/seed${random_state}.log 2>&1

