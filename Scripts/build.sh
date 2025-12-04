#!/bin/bash 

# useful:
# ssh rennes.g5k
# oarsub -I -t deploy
# oarsub -I
# oarstat -u -> check your jobs
# rsync -avzP /home/isabela/Codes/LibIFT/<..>/ rennes.g5k:/home/iborlido/LibIFT/<..>
# kadeploy3 ubuntu2404-nfs
# oarsub -r now -l nodes=2,walltime=4:00
# oarstat -fj 3357071 -> check job details
# tar -zcvf folder.tar.gz folder -> compress folder
# tar -xf folder.tar.gz -> unzip folder

# chmod +x experiment.sh; oarsub -t deploy -r now "./experiment.sh"
# oarsub -t deploy -r now -q p1 -n name -l walltime=6:00 "bash ~/LibIFT/Scripts/build.sh 1 1213"

# https://slices-fr.gitlabpages.inria.fr/tech/docs/users-doc/docs/tutorials/precursors/grid5000/getting-started-g5k/
# https://www.grid5000.fr/w/GNU_Parallel

# screen:
# screen -S libift_session -> create a new screen session named libift_session
# screen -r libift_session -> reattach to the libift_session screen session
# screen -ls -> list all screen sessions
# Ctrl+A D -> detach from the current screen session
# Ctrl+A K -> kill the current screen session

# funk -m date -r lille,rennes,taurus,suno -w 2:00:00 -> display nodes avaible for 2 hours in the clusters lille, rennes, taurus, suno

# install dependencies and compile LibIFT
sudo-g5k apt-get install -y cmake libblas-dev liblapack-dev libopenblas-dev libomp-dev python3-pip # when not using kadeploy3
cd ~/LibIFT/build; cmake ..; make; cp libift.a ../lib/;
export NEWIFT_DIR="~/LibIFT/"
cd ~/LibIFT/demo/Classification; bash compile.sh 0;
cd ~/LibIFT/demo/GFLIM; bash compile.sh 0;

# unzip dataset files
#cd ~/LibIFT/Datasets/intestinal-parasites/helminth-eggs; 
#tar -xf images.tar.bz2; 
#mkdir masks; cd masks; cp ../masks.tar.bz2 ./; tar -xf masks.tar.bz2; rm -r ./masks.tar.bz2;

arch_name="9-1-1-16-max-3-125_3_noSkip"
ds_name="helminth-eggs"
stride="graph-based"
init_seeds=200
final_spx=50
node_feat="MEAN"
spx_alg="DISF"
nclasses=9
nlayers=2

split=$1
random_state=$2
cd ~/LibIFT/Scripts; 
results_dir="/home/iborlido/LibIFT/Results/gflim_iterative_pipeline_050trainSVM_server/${ds_name}/GFLIMBoFP-${arch_name}-${stride}-${spx_alg}-${init_seeds}-${final_spx}-${node_feat}-${nclasses}classes-sorted1/split${split}/";

mkdir -p ${results_dir}

python3 gflim_iterative_pipeline_050trainSVM_server.py \
    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
    ${split} ${random_state} ${arch_name} ${nlayers} --dataset_name ${ds_name} --nclass ${nclasses} \
    --stride ${stride} --init_seeds ${init_seeds} --final_spx ${final_spx} --node_feat ${node_feat} \
    --spx_alg ${spx_alg} --nimages_final 5 --sorted_vector 1 \
    --init_img 1 >> \
    ${results_dir}/seed${random_state}_layer${nlayers}.log 2>&1

#nlayers=3
#python3 gflim_iterative_pipeline_050trainSVM_server.py \
#    /home/iborlido/LibIFT/Datasets/intestinal-parasites/${ds_name} \
#    ${split} ${random_state} ${arch_name} ${nlayers} --dataset_name ${ds_name} --nclass ${nclasses} \
#    --stride ${stride} --init_seeds ${init_seeds} --final_spx ${final_spx} --node_feat ${node_feat} \
#    --spx_alg ${spx_alg} --nimages_final 5 --sorted_vector 1 >> \
#    ${results_dir}/seed${random_state}_layer${nlayers}.log 2>&1
