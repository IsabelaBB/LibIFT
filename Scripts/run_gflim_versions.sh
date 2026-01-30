#!/bin/bash 

if [ $# -lt 11 ]; then 
    echo "Invalid number of arguments. "
    echo "Usage: bash $0 <spx> <initial_seeds> <kernel_size> <kernel_neighbors>" \
            "<pool_size> <pool_neighbors> <pool_name> " \
            "<fixed_spx> <n_kernels> <stride_size> <stride_neighbors>" 
    exit -1
fi

spx=$1
initial_seeds=$2
kernel_size=$3
kernel_neighbors=$4
pool_size=$5
pool_neighbors=$6
pool_name=$7
fixed_spx=$8
n_kernels=$9
stride_size=${10}
stride_neighbors=${11}

graph_feat="MEAN_CONCAT_STD_SIZE"
aggregation_opt="CONCAT"
attention="CLASS_DOMINANCE"

echo "SPX: ${spx}"
echo "Kernel size: ${kernel_size}"
echo "Kernel neighbors: ${kernel_neighbors}"
echo "Pool size: ${pool_size}"
echo "Pool neighbors: ${pool_neighbors}"
echo "Pool name: ${pool_name}"
echo "Fixed SPX: ${fixed_spx}"
echo "Number of kernels: ${n_kernels}"
echo "Stride size: ${stride_size}"
echo "Stride neighbors: ${stride_neighbors}"

IFT_dir="/home/isabela/Codes/LibIFT"
arch_name="${kernel_size}-1-1-${n_kernels}-${pool_name}-${pool_size}-${stride_size}_3_noSkip"

# check if arch file exists
arch_file="${IFT_dir}/Datasets/archs/${arch_name}.json"
if [ ! -f "${arch_file}" ]; then
    echo "Architecture file ${arch_file} does not exist. Exiting."
    exit -1
fi

# set model directory
model_dir="${IFT_dir}/Results/helminth-eggs/gflimBoFP_precomputedGraph/split1/${arch_name}/${initial_seeds}-${spx}spx_${kernel_neighbors}-kernel_${pool_neighbors}-pool_${stride_neighbors}-stride"
if [ "${fixed_spx}" -eq 1 ]; then 
    model_dir="${model_dir}/fixedSPX_${graph_feat}_${aggregation_opt}_${attention}"
else
    model_dir="${model_dir}/notFixedSPX_${graph_feat}_${aggregation_opt}_${attention}"
fi

# create model directory if it does not exist
mkdir -p "${model_dir}"

# create graphs and spx labels
if [ "${pool_neighbors}" = "color" ] || [ "${stride_neighbors}" = "color" ] || [ "${kernel_neighbors}" = "color" ]; then 
    /home/isabela/Codes/LibIFT/bin/iftGFLIMCreateNeighbMatricesAndGraphs \
        /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/train1.csv \
        ${spx} ${initial_seeds} \
        ${model_dir} ${graph_feat} \
        ${arch_file} 1 ${fixed_spx} \
        ${kernel_neighbors} ${pool_neighbors} ${stride_neighbors}
else
    /home/isabela/Codes/LibIFT/bin/iftCreateGFLIMGraph \
        /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/train1.csv \
        ${spx} \
        ${model_dir} ${graph_feat} DISF ${initial_seeds} 1
fi

if [ "${kernel_neighbors}" = "color" ]; then 
	/home/isabela/Codes/LibIFT/bin/iftGFLIMPrecConvertSeeds \
        ${model_dir} \
        /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/markers1 \
        ${n_kernels} ${kernel_size} ${aggregation_opt}
else
	/home/isabela/Codes/LibIFT/bin/iftGFLIMConvertSeeds \
        ${model_dir} \
        /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/markers1 \
        ${n_kernels} ${kernel_size} \
        ${attention} ${aggregation_opt}
fi

for layer in 1 2 3; do

if [ "${kernel_neighbors}" = "color" ] || [ "${stride_neighbors}" = "color" ]; then 
	/home/isabela/Codes/LibIFT/bin/iftGFLIMPrecCreateLayerModel \
        ${model_dir} \
        ${arch_file} ${layer} \
        ${kernel_neighbors} ${stride_neighbors} \
        ${graph_feat} ${aggregation_opt} /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/images
else
	/home/isabela/Codes/LibIFT/bin/iftGFLIMCreateLayerModel \
        ${model_dir} \
        ${arch_file} ${layer} \
        graph-based ${attention} ${aggregation_opt}
fi

/home/isabela/Codes/LibIFT/bin/iftGFLIMMergeLayerModels \
    ${model_dir} ${layer}

if [ "${kernel_neighbors}" = "color" ] || [ "${pool_neighbors}" = "color" ] || [ "${stride_neighbors}" = "color" ]; then 
	/home/isabela/Codes/LibIFT/bin/iftGFLIMEncodeMergedLayerBatchV3 \
        ${model_dir} ${arch_file} ${layer} \
        64 ${attention} ${kernel_neighbors} \
        ${pool_neighbors} ${stride_neighbors} \
        ${graph_feat} ${aggregation_opt} /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/images
else
	/home/isabela/Codes/LibIFT/bin/iftGFLIMEncodeMergedLayerBatch \
        ${model_dir} ${arch_file} ${layer} 64 graph-based \
        ${attention} ${aggregation_opt}
fi

/home/isabela/Codes/LibIFT/bin/iftGFLIMWriteActivations \
    ${model_dir} ${layer} 64

done


# run this with the following:
/home/isabela/Codes/LibIFT/bin/iftGFLIMExtractV3 \
    /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/train-val1.csv \
    ${model_dir} 1 ${spx} ${initial_seeds} ${fixed_spx} ${graph_feat} \
    ${arch_file} 3 64 ${attention} ${aggregation_opt} \
    1 -1 -1 graph-adaptive \
    ${kernel_neighbors} ${pool_neighbors} ${stride_neighbors}

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

rm -r ${model_dir}/layer0
rm -r ${model_dir}/labels0

for layer in 1 2 3; do
	activations_dir="${model_dir}/layer${layer}_trainSVM"
    rm -r ${activations_dir}
    rm -f ${activations_dir}.zip
    rm -r ${model_dir}/layer${layer}
    rm -r ${model_dir}/labels${layer}
done;

/home/isabela/Codes/LibIFT/bin/iftGFLIMExtractV3 \
    /home/isabela/Codes/LibIFT/Datasets/intestinal-parasites/helminth-eggs/build/test1.csv \
    ${model_dir} 1 ${spx} ${initial_seeds} ${fixed_spx} ${graph_feat} \
    ${arch_file} 3 64 ${attention} ${aggregation_opt} \
    1 -1 -1 graph-adaptive \
    ${kernel_neighbors} ${pool_neighbors} ${stride_neighbors}

# run this with the following:
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

for layer in 1 2 3; do
	activations_dir="${model_dir}/layer${layer}_test"
    rm -r ${activations_dir}
    rm -f ${activations_dir}.zip
    rm -r ${model_dir}/layer${layer}
    rm -r ${model_dir}/labels${layer}
done;