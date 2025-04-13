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
#include "iftFLIMGraph.h"

/* --------------------- GFLIM seeds (user drawn markers) ----------------------------- */

void iftGFLIMRelabelConnectedComponentsInPlace(iftFLIMGraph *graph, 
    iftLabeledSet *S, 
    int *obj_label_map);

iftDataSet *iftGFLIMExtractLabelDataSet(iftFLIMGraph *graph,
        iftLabeledSet *S,
        int kernel_size,
        iftFLIMGraphAttention attention_opt,
        iftGFLIMAggregationType aggregation_opt);

/* --------------------- GFLIM training ----------------------------- */

iftMImage *iftGFLIMSumMImgToFloatArray(iftMImage *mimg, float *bias, bool inplace);

void iftGFLIMTrainLayer(char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir);

/* --------------------- GFLIM extract features ----------------------------- */

void iftFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, 
                                        iftFLIMArch *arch, char *param_dir, int layer_index,
                                        char *feat_dir, int batchsize);

/* --------------------- GFLIM pooling ----------------------------- */

void iftGFLIMPoolingLayer(iftFLIMGraph *graph,
                        char *pool_type, 
                        int pool_size, 
                        int pool_stride);

/* --------------------- GFLIM decoder ----------------------------- */

float *iftGFLIMTriStateAdaptativeWeigth(iftMImage *mimg);

iftFImage *iftGFLIMTriStateAdaptativeDecoder(iftMImage *mimg);

#ifdef __cplusplus
}
#endif
#endif