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
- Python scripts: <https://github.com/IsabelaBB/pyGFLIM>

The following files are required to create a model:

- A .json file with the architecture's description. Example: [arch1.json](Datasets/archs/arch1.json).
- A path with images for train, validation, and test. Example: all files in [Datasets/helminth-eggs/splits_incremental/split1](Datasets/helminth-eggs/splits_incremental/split1), [Datasets/helminth-eggs/splits_incremental/split2](Datasets/helminth-eggs/splits_incremental/split2), and [Datasets/helminth-eggs/splits_incremental/split3](Datasets/helminth-eggs/splits_incremental/split3).
- A path with .txt files, one for each training image, with image coordinates and label to marked pixels. Example: [img_seeds](Datasets/helminth-eggs/img_seeds).
