#bin/bash

function usage {
	echo "Usage: $0 [-a] [-m nkernels_per_label] [-s split] [-i init_seeds] IMGS_DIR MARKERS_DIR MARKERS_NAME GRAPH_FEAT SPX_ALG ARCH NLAYERS STRIDE_TYPE DECODER ATTENTION AGGREGATION"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Ex: ../../datasets/baseline_flim/schisto/image-splits2d"
	echo "  MARKERS_DIR:          Image markers directory. Ex: ../../datasets/baseline_flim/schisto"
	echo "  MARKERS_NAME          Name of the image markers folder. Ex: stroke_markers"
	echo "  GRAPH_FEAT:           How to compute graph nodes features. Options: MEAN, SUM, SUM_MULT_RELSIZE, MEAN_MULT_RELSIZE, MEAN_CONCAT_RELSIZE, MEAN_CONCAT_RELSIZE_NEIGHB, MEAN_CONCAT_RELSIZE_HARALICK, SUM_MULT_RELSIZE_CONCAT_HARALICK, SUM_CONCAT_HARALICK (See iftFLIMGraph.c)"
	echo "  SPX_ALG:              DISF, ERGC"
	echo "  ARCH:                 architecture file (.json)"
	echo "  NLAYERS               Number of layers"
	echo "  STRIDE_TYPE:          none, image-based, graph-based"
	echo "  DECODER:              tristate, hybrid, graph-adaptive, gaussian"
	echo "  ATTENTION:            Attention option (NONE, NODE_RELSIZE, NODE_SIZE, NEIGHB_SIMIL, COSINE_SIMILARITY, #NEIGHB_SIZE, CLASS_DOMINANCE_TEST, CLASS_DOMINANCE, CLASS_DOMINANCE_WINNER_POWER2, CLASS_DOMINANCE_SOFTMAX_NEIGHB)"
	echo "  AGGREGATION:          Aggregation option (CONCAT, MEAN, WMEAN)"
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

if [ "$#" -lt 11 ]; then
	echo "num args: $#" 
	usage
fi

# Mandatory arguments
IMGS_DIR=$1; # ../../datasets/baseline_flim/schisto/image-splits2d
MARKERS_DIR=$2; # ../../datasets/baseline_flim/schisto/
MARKERS_NAME=$3; # stroke_markers
GRAPH_FEAT=$4; # MEAN, SUM, SUM_MULT_RESIZE, MEAN_MULT_RESIZE, MEAN_CONCAT_RESIZE, ..
SPX_ALG=$5; # DISF, ERGC
ARCH=$6; # .json file path
NLAYERS=$7;
STRIDE_TYPE=$8; # none, image-based, graph-based
DECODER=$9; # tristate, hybrid, graph-adaptive, gaussian
ATTENTION=${10};
AGGREGATION=${11};

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

function AssignVariables {
	# Assign variables
	IMG_PATH="${IMGS_DIR}/split${s}/orig/train/"	# images directory
	MARKERS_PATH="${MARKERS_DIR}/${MARKERS_NAME}/split${s}/" # markers directory
	
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
	
	MODEL_DIR="./schisto/split${s}/${MARKERS_NAME}/gflim-BoFP/${STRIDE_FOLDER_NAME}/${SPX_ALG}/${tmp}/${GRAPH_FEAT}-${ATTENTION}/";	# model directory
}

function execGFLIM {
	ARGS="${IMG_PATH} ${MARKERS_PATH} ${ARCH} ";
	ARGS+="${NLAYERS} "; # nlayers
	ARGS+="${MODEL_DIR} ${STRIDE_TYPE} ${spx} ${GRAPH_FEAT} ${SPX_ALG} ${DECODER} ${ATTENTION} ${AGGREGATION} ";	

	[ "$SPX_ALG" == "DISF" ] && ARGS+="-init_seeds ${INIT_SEEDS} ";	# init seeds (DISF)
	[ ${KERNEL_REMOVAL} != "0" ] && ARGS+="-reduce_kernels ${KERNEL_REMOVAL} ";
	[ ${DROPOUT} -eq 1 ] && ARGS+="-dropout ";
	[ ${ACTIVATIONS} -eq 1 ] && ARGS+="-act ";	# write activations
	
	python3 ../demo/GFLIM/GFLIM-BoFP/GFLIM-BoFP-train.py ${ARGS};
}


case ${STRIDE_TYPE} in
	"none")
		if [ ${KMEANS} -eq 0 ]; then
			superpixels=(3000 4000 5000)
		else
			superpixels=(6000 4000 5000)
		fi
		;;
	"image-based")
		superpixels=(4000 4000 5000);
		;;
	"graph-based")
		if [ ${KMEANS} -eq 0 ]; then
			superpixels=(5000 8000 8000)
		else
			superpixels=(6000 7000 7000);
		fi
		;;
	*)
		echo "Invalid STRIDE_TYPE: $STRIDE_TYPE"
		exit 1;;
esac

if [ ${SPLIT} == "0" ]; then
	for s in 1 2 3; do
		spx=${superpixels[${s}-1]};
		AssignVariables;
		execGFLIM;
	done;
else
	s=$((${SPLIT} + 0));
	spx=${superpixels[${s}-1]};
	AssignVariables;
	execGFLIM;
fi


