#bin/bash

function usage {
	echo "Usage: $0 [-m nkernels_per_label] [-s split] [-i init_seeds] IMGS_DIR DS_NAME MARKERS_NAME GRAPH_FEAT SPX_ALG ARCH NLAYERS STRIDE_TYPE DECODER ATTENTION AGGREGATION NUM_SUPERPIXELS BATCH_SIZE"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Example: ../Datasets/schisto/image-splits2d "
	echo "  DS_NAME:              Dataset name. Example: schisto "
	echo "  MARKERS_NAME          Name of the image markers folder. Example: stroke_markers"
	echo "  GRAPH_FEAT:           Graph features option (See iftGFLIMGraphFeature at iftGFLIMGraph.h). Examples: MEAN, SUM"
	echo "  SPX_ALG:              Superpixel segmentation option (see iftGFLIMGraphType at iftGFLIMGraph.h). Options: DISF, ERGC"
	echo "  ARCH:                 Architecture (.json) file. Examples: arch1.json, arch1_noStride.json"
	echo "  NLAYERS               Number of layers"
	echo "  STRIDE_TYPE:          Stride type (see iftGFLIMStrideType at iftGFLIM_BoFP.h). Options: none, image-based, graph-based"
	echo "  DECODER:              Decode option (see iftGFLIMDecoderType at iftGFLIM_BoFP.h). Example: hybrid "
	echo "  ATTENTION:            Attention option (see iftGFLIMGraphAttention at iftGFLIMGraph.h). Examples: NONE, CLASS_DOMINANCE"
	echo "  AGGREGATION:		  Aggregation option (see iftGFLIMAggregationType at iftGFLIMGraph.h). Options: CONCAT, MEAN, WMEAN"
	echo "  NUM_SUPERPIXELS:      Number of superpixels (> 0)"
	echo "  BATCH_SIZE            Batch size (> 0). Example: 64"
	echo "Optional arguments:"
	echo "  -m:                   perform a second kmeans per label"
	echo "  -s:                   split (1, 2, or 3). Default: 3 splits"
	echo "  -i:                   initial seeds (needed if SPX_ALG is DISF"
	exit 1
}

# Optional arguments
KMEANS=1;
SPLIT=0;
INIT_SEEDS="0";
KERNEL_REMOVAL="0";
DROPOUT=0;
while getopts "dm:s:i:" opt
do
    case "${opt}" in 
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

if [ "$#" -lt 13 ]; then
	echo "num args: $#" 
	usage
fi

# Mandatory arguments
IMGS_DIR=$1; 
DS_NAME=$2;
MARKERS_NAME=$3; 
GRAPH_FEAT=$4; 
SPX_ALG=$5; 
ARCH=$6; 
NLAYERS=$7;
STRIDE_TYPE=$8; 
DECODER=$9; 
ATTENTION=${10};
AGGREGATION=${11};
NUM_SUPERPIXELS=${12};
BATCH_SIZE=${13};

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
	IMG_PATH="${IMGS_DIR}/split${s}/orig/val/"; # images directory
	tmp=""
	if [ "$SPX_ALG" == "DISF" ]; then
		tmp+="${INIT_SEEDS}-"
	fi
	tmp+="${NUM_SUPERPIXELS}/"
	if [ ${KMEANS} -eq 0 ]; then # not used
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

	MODEL_DIR="../Results/${DS_NAME}/split${s}/${MARKERS_NAME}/gflim-BoFP/${STRIDE_FOLDER_NAME}/${SPX_ALG}/${tmp}/${GRAPH_FEAT}-${ATTENTION}/";	# model directory

	ARGS="${IMG_PATH} ${ARCH} ${NLAYERS} ${MODEL_DIR} ${STRIDE_TYPE} ${NUM_SUPERPIXELS} ${GRAPH_FEAT} ${SPX_ALG} ${DECODER} ${ATTENTION} ${BATCH_SIZE} ${AGGREGATION} ";
	
	[ ${DROPOUT} -eq 1 ] && ARGS+="-dropout ";
	[ "$SPX_ALG" == "DISF" ] && ARGS+="-init_seeds ${INIT_SEEDS} ";

	python3 ../demo/GFLIM/GFLIM-extract.py ${ARGS};
}

if [ ${SPLIT} == "0" ]; then
	for s in 1 2 3; do
		execGFLIM;
	done;
else
	s=$((${SPLIT} + 0));
	execGFLIM;
fi
