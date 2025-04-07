#bin/bash

function usage {
	echo "Usage: $0 [-s split] [-i init_seeds] GRAPH_FEAT SPX_ALG ARCH MARKERS_NAME NLAYERS"
	echo "Mandatory arguments:"
	echo "  GRAPH_FEAT:           How to compute graph nodes features. Options: MEAN, SUM, SUM_MULT_RELSIZE, MEAN_MULT_RELSIZE, MEAN_CONCAT_RELSIZE, MEAN_CONCAT_RELSIZE_NEIGHB, MEAN_CONCAT_RELSIZE_HARALICK, SUM_MULT_RELSIZE_CONCAT_HARALICK, SUM_CONCAT_HARALICK (See iftFLIMGraph.c)"
	echo "  SPX_ALG:              DISF, ERGC"
	echo "  ARCH:                 architecture file (.json)"
	echo "  MARKERS_NAME:         image markers path"
	echo "	NLAYERS				  Number of layers"
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

if [ "$#" -lt 5 ]; then
	echo "num args: $#" 
	usage
fi

# Mandatory arguments
GRAPH_FEAT=$1; # MEAN, SUM, SUM_MULT_RESIZE, MEAN_CONCAT_RESIZE
SPX_ALG=$2; # DISF, ERGC
ARCH=$3; # .json file
MARKERS_NAME=$4;
NLAYERS=$5;

function execGFLIM {
	# model directory
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
	
	MODEL_DIR="./schisto/split${s}/${MARKERS_NAME}/gflim/${SPX_ALG}/${tmp}/${GRAPH_FEAT}/";	# model directory

	ARGS="../../datasets/baseline_flim/schisto/image-splits2d/split${s}/orig/val/ "; 	# img_dir
	ARGS+="${ARCH} "; # arch file path
	ARGS+="${NLAYERS} "; # nlayers
	ARGS+="${MODEL_DIR} none ${spx} ${GRAPH_FEAT} ${SPX_ALG} tristate NONE ";
	ARGS+="64 "; # bach size
	ARGS+="CONCAT "; 
	
	if [ "$SPX_ALG" == "DISF" ]; then
		ARGS+="-init_seeds ${INIT_SEEDS} ";
	fi	

	echo "split ${s}, spx ${spx}, graph feature ${GRAPH_FEAT}, sup. algorithm ${SPX_ALG}, arch ${ARCH}";
	python3 ../demo/GFLIM/GFLIM-extract.py ${ARGS};
}

superpixels=(6000 4000 5000)


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
