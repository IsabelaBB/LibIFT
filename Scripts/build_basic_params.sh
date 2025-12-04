#!/bin/bash 

# oarsub -t deploy -r now -q p1 -n name -l walltime=12:00 "bash ~/LibIFT/Scripts/build_basic_params.sh 1 MEAN 100,150,175,200"

if [ $# -lt 2 ]; then 
    echo "Invalid number of arguments. Usage: bash build_basic_params.sh <split> <node_feat> [<superpixels_list>]"
    exit -1
fi

# install dependencies and compile LibIFT
sudo-g5k apt-get install -y cmake libblas-dev liblapack-dev libopenblas-dev libomp-dev python3-pip # when not using kadeploy3
cd ~/LibIFT/build; cmake ..; make; cp libift.a ../lib/;
export NEWIFT_DIR="~/LibIFT/"
cd ~/LibIFT/demo/Classification; bash compile.sh 0;
cd ~/LibIFT/demo/GFLIM; bash compile.sh 0;

arch_name="9-1-1-50-max-3-125_3_noSkip"
ds_name="helminth-eggs"
stride="graph-based"
init_seeds=200
spx_alg="DISF"
nclasses=9
nlayers=3
nimages=1
random_state=42
superpixels_list="100,150,175,200"

split=$1
node_feat=$2

if [ $# -gt 2 ]; then 
    superpixels_list=$3
fi

cd ~/LibIFT/Scripts; 
python3 gflim_pipeline_basic_params.py \
    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
    ${split} ${random_state} ${arch_name} ${nlayers} --dataset_name ${ds_name} --nclass ${nclasses} \
    --stride ${stride} --init_seeds ${init_seeds} --final_spx ${superpixels_list} --node_feat ${node_feat} \
    --spx_alg ${spx_alg} --nimages ${nimages} --sorted_vector 1