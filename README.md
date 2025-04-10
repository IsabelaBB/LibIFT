# LibIFT: A Library for the Image Forest Transform and its tools

## Requirements

- CUDA toolkit
- GCC 4.9+
- BLAS + LAPACK
- OpenMP

## Build, compile, and install

```shell
mkdir build; cd build; cmake ..; make; make install;
```

## GFLIM

- Programs: <https://github.com/IsabelaBB/GFLIM_BoFP>

The following files are required to create a model:

- A .json file with the architecture's description. Examples: [arch1.json](Scripts_GFLIM/arch1.json) and [arch1_noStride.json](Scripts_GFLIM/arch1_noStride.json)
- A path with images for train, validation, and test. Example: [Datasets/schisto/image-splits2d](Datasets/schisto/image-splits2d)
- A path with .txt files, one for each training image, with image coordinates and obj/bkg label to marked pixels. Example: [stroke_markers](Datasets/schisto/stroke_markers)

### Scripts

[Scripts_GFLIM](Scripts_GFLIM) provide bash scripts to train a model using GFLIM and GFLIM-BoFP and extract features from a trained model. These scripts require the GFLIM and GFLIM-BoFP files.

#### Train GFLIM

- File: [train_gflim.sh](Scripts_GFLIM/train_gflim.sh)
- Train example:

```bash
bash train_gflim.sh -a -s 1 -i 20000 ../Datasets/schisto/image-splits2d ../Datasets/schisto/stroke_markers SUM DISF arch1_noStride.json 3
```

#### Extract features from GFLIM model

- File: [extract_gflim.sh](Scripts_GFLIM/extract_gflim.sh)
- Extract features example:

```bash
bash extract_gflim.sh -s 1 -i 20000 ../Datasets/schisto/image-splits2d stroke_markers SUM DISF arch1_noStride.json 3 64
```

#### Train GFLIM-BoFP

- File: [train_gflim-BoFP.sh](Scripts_GFLIM/train_gflim-BoFP.sh)
- Train example:

```bash
bash train_gflim-BoFP.sh -a -s 1 -i 20000 ../Datasets/schisto/image-splits2d ../Datasets/schisto/stroke_markers SUM DISF arch1.json 3 graph-based hybrid CLASS_DOMINANCE CONCAT 
```

#### Extract features from GFLIM-BoFP model

- File: [extract_gflim-BoFP.sh](Scripts_GFLIM/extract_gflim-BoFP.sh)
- Extract features example:

```bash
bash extract_gflim-BoFP.sh -s 1 -i 20000 ../Datasets/schisto/image-splits2d stroke_markers SUM DISF arch1.json 3 graph-based hybrid CLASS_DOMINANCE CONCAT 64
```
