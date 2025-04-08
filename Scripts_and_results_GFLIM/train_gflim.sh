#bin/bash

function usage {
	echo "Usage: $0 [-a] [-s split] [-i init_seeds] IMGS_DIR MARKERS_DIR MARKERS_NAME GRAPH_FEAT SPX_ALG ARCH NLAYERS"
	echo "Mandatory arguments:"
	echo "  IMGS_DIR:             Images directory. Ex: ../../datasets/baseline_flim/schisto/image-splits2d"
	echo "  MARKERS_DIR:          Image markers directory. Ex: ../../datasets/baseline_flim/schisto"
	echo "  MARKERS_NAME          Name of the image markers folder. Ex: stroke_markers"
	echo "  GRAPH_FEAT:           How to compute graph nodes features. Options: MEAN, SUM, SUM_MULT_RELSIZE, MEAN_MULT_RELSIZE, MEAN_CONCAT_RELSIZE, MEAN_CONCAT_RELSIZE_NEIGHB, MEAN_CONCAT_RELSIZE_HARALICK, SUM_MULT_RELSIZE_CONCAT_HARALICK, SUM_CONCAT_HARALICK (See iftFLIMGraph.c)"
	echo "  SPX_ALG:              DISF, ERGC"
	echo "  ARCH:                 architecture file (.json)"
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

if [ "$#" -lt 7 ]; then
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
	fi
	
	MODEL_DIR="./schisto/split${s}/${MARKERS_NAME}/gflim/${SPX_ALG}/${tmp}/${GRAPH_FEAT}";	# model directory
}

function execGFLIM {
	ARGS="${IMG_PATH} ${MARKERS_PATH} ${ARCH} ";
	ARGS+="${NLAYERS} "; # nlayers
	ARGS+="${MODEL_DIR} ${spx} ${GRAPH_FEAT} ${SPX_ALG} ";	

	ARGS+="-init_seeds ${INIT_SEEDS} ";	# init seeds (DISF)
	
	[ ${ACTIVATIONS} -eq 1 ] && ARGS+="-act "; # write activations
	
	python3 ../demo/GFLIM/GFLIM/GFLIM-train.py ${ARGS};
}

superpixels=(5000 4000 5000)

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


