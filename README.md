# LibIFT: A Library for the Image Forest Transform and its tools

## Requirements

- CUDA toolkit
- GCC 4.9+
- BLAS + LAPACK
- OpenMP

## Build, compile, and install the IFT Library

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

[pyGFLIM](pyGFLIM) provide python scripts to train a model, extract features from a trained model, and more. These scripts require the GFLIM files.
