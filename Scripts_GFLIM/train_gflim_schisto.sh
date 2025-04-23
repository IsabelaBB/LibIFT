#bin/bash

function usage {
	echo "Usage: $0 [-a] [-s split] [-i init_seeds] IMGS_DIR MARKERS_DIR GRAPH_FEAT SPX_ALG ARCH NLAYERS"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Example: ../Datasets/schisto/image-splits2d "
	echo "  MARKERS_DIR:          Image markers directory. Ex: ../Datasets/schisto/stroke_markers"
	echo "  GRAPH_FEAT:           Graph features option (See iftFLIMGraphFeature at iftFLIMGraph.h). Examples: MEAN, SUM"
	echo "  SPX_ALG:              Superpixel segmentation option (see iftFLIMGraphType at iftFLIMGraph.h). Options: DISF, ERGC"
	echo "  ARCH:                 Architecture (.json) file. Examples: arch1.json, arch1_noStride.json"
	echo "  NLAYERS               Number of layers"
	echo "Optional arguments:"
	echo "  -a:                   write the activations of the training set per layer"
	echo "  -s:                   split (1, 2, or 3). Default: 3 splits"
	echo "  -i:                   number of initial seeds. Needed if SPX_ALG is DISF)"
	exit 1
}

# Optional arguments
ACTIVATIONS=0;
KMEANS=1;
SPLIT=0;
INIT_SEEDS="0";
while getopts "ams:i:" opt
do
    case "${opt}" in
        a) ACTIVATIONS=1;;
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

if [ "$#" -lt 6 ]; then
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
MARKERS_NAME=$(basename ${MARKERS_DIR}); 
superpixels=(5000 4000 5000) # superpixel per split

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
	fi
	
	MODEL_DIR="../Results/schisto/split${s}/${MARKERS_NAME}/gflim/${SPX_ALG}/${tmp}/${GRAPH_FEAT}";	# model directory

	ARGS="${IMG_PATH} ${MARKERS_PATH} ${ARCH} ${NLAYERS} ${MODEL_DIR} ${spx} ${GRAPH_FEAT} ${SPX_ALG} ";	

	[ "$SPX_ALG" == "DISF" ] && ARGS+="-init_seeds ${INIT_SEEDS} ";	# init seeds (DISF)
	[ ${ACTIVATIONS} -eq 1 ] && ARGS+="-act "; # write activations
	
	python3 ../demo/GFLIM/GFLIM/GFLIM-train.py ${ARGS};
}

if [ ${SPLIT} == "0" ]; then
	for s in 1 2 3; do
		spx=${superpixels[${s}-1]};
		execGFLIM;
	done;
else
	s=$((${SPLIT} + 0));
	spx=${superpixels[${s}-1]};
	execGFLIM;
fi


