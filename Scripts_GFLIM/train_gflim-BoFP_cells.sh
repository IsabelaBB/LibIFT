#bin/bash

function usage {
	echo "Usage: $0 [-a] [-m nkernels_per_label] [-s split] [-i init_seeds] IMGS_DIR MARKERS_DIR GRAPH_FEAT SPX_ALG ARCH NLAYERS STRIDE_TYPE DECODER ATTENTION AGGREGATION"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Example: ../Datasets/cells_dataset/splits "
	echo "  MARKERS_DIR:          Image markers directory. Ex: ../Datasets/cells_dataset/markers/gt_markers_erosion9/gt_markers_erosion9"
	echo "  GRAPH_FEAT:           Graph features option (See iftGFLIMGraphFeature at iftGFLIMGraph.h). Examples: MEAN, SUM"
	echo "  SPX_ALG:              Superpixel segmentation option (see iftGFLIMGraphType at iftGFLIMGraph.h). Options: DISF, ERGC"
	echo "  ARCH:                 Architecture (.json) file. Examples: arch1.json, arch1_noStride.json"
	echo "  NLAYERS               Number of layers"
	echo "  STRIDE_TYPE:          Stride type (see iftGFLIMStrideType at iftGFLIM_BoFP.h). Options: none, image-based, graph-based"
	echo "  DECODER:              Decode option (see iftGFLIMDecoderType at iftGFLIM_BoFP.h). Example: hybrid "
	echo "  ATTENTION:            Attention option (see iftGFLIMGraphAttention at iftGFLIMGraph.h). Examples: NONE, CLASS_DOMINANCE"
	echo "  AGGREGATION:		  Aggregation option (see iftGFLIMAggregationType at iftGFLIMGraph.h). Options: CONCAT, MEAN, WMEAN"
	echo "Optional arguments:"
	echo "  -a:                   write the activations of the training set per layer"
	echo "  -m:                   perform a second kmeans per label"
	echo "  -s:                   split (1, 2, or 3). Default: 3 splits"
	echo "  -i:                   number of initial seeds. Needed if SPX_ALG is DISF)"
	exit 1
}

# Optional arguments
ACTIVATIONS=0;
KMEANS=1;
SPLIT=0;
INIT_SEEDS="0";
KERNEL_REMOVAL="0";
DROPOUT=0;
while getopts "adm:s:i:" opt
do
    case "${opt}" in
        a) ACTIVATIONS=1;;
		d) DROPOUT=1;;
		m) KERNEL_REMOVAL=${OPTARG};;
        s) SPLIT=${OPTARG};;
		i) INIT_SEEDS=${OPTARG};;
		*) exit 1;;
    esac
done
shift $((OPTIND -1))

if [ ${SPLIT} != "0" ] && [ ${SPLIT} != "1" ] && [ ${SPLIT} != "2" ] && [ ${SPLIT} != "3" ]; then
	echo "Invalid split value ${SPLIT}."
	exit 1
fi

if [ "$#" -lt 10 ]; then
	echo "num args: $#" 
	usage
fi

# Mandatory arguments
IMGS_DIR=$1;
MARKERS_DIR=$2; 
GRAPH_FEAT=$3; 
SPX_ALG=$4; 
ARCH=$5; 
NLAYERS=$6;
STRIDE_TYPE=$7; 
DECODER=$8;
ATTENTION=${9};
AGGREGATION=${10};
MARKERS_NAME=$(basename ${MARKERS_DIR}); 

# Set stride folder name
case $STRIDE_TYPE in
	"none")
		STRIDE_FOLDER_NAME="noStride";;
	"image-based")
		STRIDE_FOLDER_NAME="imageStride";;
	"graph-based")
		STRIDE_FOLDER_NAME="graphStride";;
	*)
		echo "Invalid STRIDE_TYPE: $STRIDE_TYPE"
		exit 1;;
esac

function execGFLIM {
	IMG_PATH="${IMGS_DIR}/split${s}/orig/train/" # images directory
	MARKERS_PATH="${MARKERS_DIR}/split${s}/" # markers directory
	
	tmp=""
	if [ "$SPX_ALG" == "DISF" ]; then
		tmp+="${INIT_SEEDS}-"
	fi
	tmp+="${spx}/"
	if [ ${KMEANS} -eq 0 ]; then
		tmp+="noKmeans";
	else 
		tmp+="kmeans"
		if [ ${KERNEL_REMOVAL} != "0" ]; then
			tmp+="-reduce${KERNEL_REMOVAL}";
		fi
		if [ ${DROPOUT} -eq 1 ]; then
			tmp+="-dropout";
		fi
	fi
	if [ ${AGGREGATION} != "CONCAT" ]; then
		tmp+="-agg${AGGREGATION}"
	fi
	
	MODEL_DIR="../Results/cells_dataset/split${s}/${MARKERS_NAME}/gflim-BoFP/${STRIDE_FOLDER_NAME}/${SPX_ALG}/${tmp}/${GRAPH_FEAT}-${ATTENTION}/";	# model directory

	ARGS="${IMG_PATH} ${MARKERS_PATH} ${ARCH} ${NLAYERS} ${MODEL_DIR} ${STRIDE_TYPE} ${spx} ${GRAPH_FEAT} ${SPX_ALG} ${DECODER} ${ATTENTION} ${AGGREGATION} ";	

	[ "$SPX_ALG" == "DISF" ] && ARGS+="-init_seeds ${INIT_SEEDS} ";	# init seeds (DISF)
	[ ${KERNEL_REMOVAL} != "0" ] && ARGS+="-reduce_kernels ${KERNEL_REMOVAL} ";
	[ ${DROPOUT} -eq 1 ] && ARGS+="-dropout ";
	[ ${ACTIVATIONS} -eq 1 ] && ARGS+="-act ";	# write activations
	
	echo python3 ../demo/GFLIM/GFLIM-BoFP/GFLIM-BoFP-train.py ${ARGS};
}

# Assign superpixels based on the stride type and split
case ${STRIDE_TYPE} in
	"none")
		if [ ${KMEANS} -eq 0 ]; then
			superpixels=(10000 10000 10000)
		else
			superpixels=(10000 10000 10000)
		fi
		;;
	"image-based")
		superpixels=(10000 10000 10000);
		;;
	"graph-based")
		if [ ${KMEANS} -eq 0 ]; then
			superpixels=(20000 20000 20000)
		else
			superpixels=(20000 20000 20000);
		fi
		;;
	*)
		echo "Invalid STRIDE_TYPE: $STRIDE_TYPE"
		exit 1;;
esac

if [ ${SPLIT} == "0" ]; then
	for s in 1 2 3 4 5; do
		spx=${superpixels[${s}-1]};
		execGFLIM;
	done;
else
	s=$((${SPLIT} + 0));
	spx=${superpixels[${s}-1]};
	execGFLIM;
fi


