/**
 * @file iftGFLIM.h
 * @brief GFLIM: Graph-based Feature Learning From Image Markers  
 * 
 * Learn the parameters of a sequence of convolutional layers from
 * labeled nodes in graphs and employ the resulting model for feature extraction.
 * 
 * @note <b>Programs:</b>
 * * @ref iftCreateGFLIMGraph.c = Convert images to graphs
 * * @ref iftGFLIMConvertSeeds.c = Convert image markers to graph markers
 * * @ref iftGFLIMTrainLayer.c = Train a layer using GFLIM
 * * @ref iftGFLIMDecodeLayer.c = Decode a layer using GFLIM
 * * @ref iftGFLIMExtract.c = Extract features from graphs using a trained model
 * 
 * @note Some of the aforementioned programs are the same for GFLIM-BoFP, requiring therefore GFLIMBoFP files.
 * 
 * @author Isabela Borlido Barcelos
 */

#ifndef IFT_GFLIM_H
#define IFT_GFLIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftFLIM.h"
#include "iftMatrix.h"
#include "iftDataSet.h"
#include "iftMImage.h"
#include "iftGFLIMGraph.h"



typedef struct iftGFLIMLayer {
  int  kernel_size;      /* kernel sizes along x, y, and z */
  int  dilation_rate;    /* dilation rates sx, sy, sz */
  int  nkernels_per_image;  /* number of filters (kernels, output channels) per image */
  int  nkernels_per_marker; /* number of filters (kernels) per marker */
  int  noutput_channels;    /* final number of filters (kernels, output channels) in a consensus layer from several training images */
  bool relu;                /* with (true) or without (false) relu */
  char *pool_type;          /* "no_pool, avg_pool, max_pool" */
  int  pool_size;        /* pooling sizes align x, y, and z */
  float  pool_stride;         /* pooling stride */
  int *skip_connection;    /* index of the layers to apply skip connection */
  char *skip_type;    /* type of skip connection: dotprod, dotprod_sumScale0, dotprod_sumScalei, sum, sum_sumScale0, sum_sumScalei */
} iftGFLIMLayer;

//! swig(destroyer = iftDestroyGFLIMArch)
typedef struct iftGFLIMArch {
  int nlayers;                  /* number of layers */
  iftGFLIMLayer *layer;          /* convolutional layer */
  float stdev_factor;           /* an adjustment parameter for marker-based normalization */
  bool apply_intrinsic_atrous;  /* intrinsic dilation meant to take the pooling stride into consideration during training*/
} iftGFLIMArch;

iftGFLIMArch *iftReadGFLIMArch(char *filename);
void iftWriteGFLIMArch(iftGFLIMArch *arch, char *filename);
void iftDestroyGFLIMArch(iftGFLIMArch **arch);

/* --------------------- GFLIM seeds (user drawn markers) ----------------------------- */

void iftGFLIMRelabelConnectedComponentsInPlace(iftGFLIMGraph *graph, 
    iftLabeledSet *S, 
    int *obj_label_map);

iftDataSet *iftGFLIMExtractLabelDataSet(iftGFLIMGraph *graph,
        iftLabeledSet *S,
        int kernel_size,
        iftGFLIMGraphAttention attention_opt,
        iftGFLIMAggregation aggregation_opt);

/* --------------------- GFLIM training ----------------------------- */

iftMImage *iftGFLIMSumMImgToFloatArray(iftMImage *mimg, float *bias, bool inplace);

void iftGFLIMTrainLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftGFLIMArch *arch, char *output_dir);

/* --------------------- GFLIM extract features ----------------------------- */

void iftGFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, 
                                        iftGFLIMArch *arch, char *param_dir, int layer_index,
                                        char *feat_dir, int batchsize);

/* --------------------- GFLIM pooling ----------------------------- */

void iftGFLIMPoolingLayer(iftGFLIMGraph *graph,
                        char *pool_type, 
                        int pool_size, 
                        float pool_stride);

/* --------------------- GFLIM decoder ----------------------------- */

float *iftGFLIMTriStateAdaptativeWeigth(iftMImage *mimg);

iftFImage *iftGFLIMTriStateAdaptativeDecoder(iftMImage *mimg);

#ifdef __cplusplus
}
#endif
#endif