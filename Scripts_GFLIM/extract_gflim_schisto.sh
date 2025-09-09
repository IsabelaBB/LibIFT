#bin/bash

function usage {
	echo "Usage: $0 [-s split] [-i init_seeds] IMGS_DIR MARKERS_NAME GRAPH_FEAT SPX_ALG ARCH MARKERS_NAME NLAYERS BATCH_SIZE"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Example: ../Datasets/schisto/image-splits2d "
	echo "  MARKERS_NAME          Name of the image markers folder. Example: stroke_markers"
	echo "  GRAPH_FEAT:           Graph features option (See iftGFLIMGraphFeature at iftGFLIMGraph.h). Examples: MEAN, SUM"
	echo "  SPX_ALG:              Superpixel segmentation option (see iftGFLIMGraphType at iftGFLIMGraph.h). Options: DISF, ERGC"
	echo "  ARCH:                 Architecture (.json) file. Example: arch1_noStride.json"
	echo "  NLAYERS               Number of layers"
	echo "  BATCH_SIZE            Batch size (> 0). Example: 64"
	echo "Optional arguments:"
	echo "  -s:                   split (1, 2, or 3). Default: 3 splits"
	echo "  -i:                   initial seeds (needed if SPX_ALG is DISF"
	exit 1
}

# Optional arguments
KMEANS=1;
SPLIT=0;
INIT_SEEDS="0";
while getopts "ms:i:" opt
do
    case "${opt}" in 
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

if [ "$#" -lt 7 ]; then
	echo "num args: $#" 
	usage
fi

# Mandatory arguments
IMGS_DIR=$1; 
MARKERS_NAME=$2; 
GRAPH_FEAT=$3; 
SPX_ALG=$4; 
ARCH=$5; 
NLAYERS=$6;
BATCH_SIZE=$7;
superpixels=(5000 4000 5000) # superpixel per split

function execGFLIM {
	IMG_PATH="${IMGS_DIR}/split${s}/orig/val/";
	tmp=""
	if [ "$SPX_ALG" == "DISF" ]; then
		tmp+="${INIT_SEEDS}-"
	fi
	tmp+="${spx}/"
	if [ ${KMEANS} -eq 0 ]; then # not used
		tmp+="noKmeans";
	else 
		tmp+="kmeans"
	fi
	
	MODEL_DIR="../Results/schisto/split${s}/${MARKERS_NAME}/gflim/${SPX_ALG}/${tmp}/${GRAPH_FEAT}/";	# model directory

	ARGS="${IMG_PATH} ${ARCH} ${NLAYERS} ${MODEL_DIR} none ${spx} ${GRAPH_FEAT} ${SPX_ALG} tristate NONE ${BATCH_SIZE} CONCAT ";
	
	if [ "$SPX_ALG" == "DISF" ]; then
		ARGS+="-init_seeds ${INIT_SEEDS} ";
	fi	

	python3 ../demo/GFLIM/GFLIM-extract.py ${ARGS};
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
