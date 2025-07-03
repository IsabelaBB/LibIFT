/**
 * @file iftGFLIM_BoFP.c
 * @brief GFLIM: Graph-based Feature Learning From Image Markers -- version with Bag of Feature Points (BoFP).
 * 
 * @note Programs:
 * * @ref iftGFLIMConvertSeeds.c = Convert image markers to graph markers and reduce the seed set to a maximum number of points per marker.
 * * @ref iftGFLIMCreateLayerModel.c = Create a layer model.
 * * @ref iftGFLIMMergeLayerModels.c = Merge layer models.
 * * @ref iftGFLIMEncodeLayer.c = Encode a layer.
 * * @ref iftGFLIMEncodeMergedLayer.c = Encode a merged layer.
 * * @ref iftGFLIMDecodeLayer.c = Decode a layer.
 * * @ref iftGFLIMExtract.c = Extract features from a graph using a trained model
 * 
 * @author Isabela Borlido Barcelos
 * @date dec 2024
 */

#ifndef IFT_GFLIM_BOFP_H
#define IFT_GFLIM_BOFP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ift.h"
#include "iftGFLIM.h"

/**
 * @enum iftGFLIMStrideType
 * @brief Enumerates the stride types for GFLIM.
 * @see iftGFLIMImageStride() and iftGFLIMGraphStride()
 */
typedef enum { 
  IFT_GFLIM_NOSTRIDE,     // No stride option.
  IFT_GFLIM_IMAGESTRIDE,  // Stride based on the image size. It reduces the image-graph mapping by a factor of the image size and reconstructs the graph from it. @see iftGFLIMImageStride()
  IFT_GFLIM_GRAPHSTRIDE   // Stride based on the graph size. It reduces the graph nodes by edge contraction based on a nodes set size faction. @see iftGFLIMGraphStride()
} iftGFLIMStrideType;

/**
 * @enum iftGFLIMDecoderType
 * @brief Enumerates the decoder types for GFLIM.
 */
typedef enum { 
  IFT_GFLIM_TRISTATE_DECODER,       // Tristate decoder. @see iftGFLIMTristateDecoder()
  IFT_GFLIM_HYBRID_DECODER,         // Hybrid decoder. @see iftGFLIMHybridDecoder()
  IFT_GFLIM_GRAPHADAPTIVE_DECODER,  // Graph adaptive decoder. @see iftGFLIMGraphAdaptiveDecoder()
  IFT_GFLIM_GAUSSIAN_DECODER,        // Gaussian decoder. @see iftGFLIMGaussianDecoder()
  IFT_GFLIM_TEST_DECODER,
  IFT_GFLIM_MEAN_DECODER,         // Mean decoder. @see iftGFLIMMeanDecoder()
} iftGFLIMDecoderType;

typedef enum { // not used
  IFT_GFLIM_DIMREDUCTION_NONE,
  IFT_GFLIM_DIMREDUCTION_TSNE,
  IFT_GFLIM_DIMREDUCTION_PCA
} iftGFLIMDimReductionType;

// ============ AUXILIAR FUNCTIONS ============ //
int iftGFLIMNumberOfTruelabels(int *truelabels, int n);

// ============ FUNCTIONS TO VALIDATE TYPES ============ //
iftGFLIMStrideType iftGFLIMGetStrideType(const char *name);
iftGFLIMDecoderType iftGLFLIMGetDecoderType(const char *name);
iftGFLIMDimReductionType iftGLFLIMGetDimReductionType(const char *name);

// ============ FUNCTIONS TO READ AND SAVE MODEL FILES ============ //
void iftGFLIMSaveBias(char *basepath, float *bias, int number_of_kernels);
void iftGFLIMSaveKernelLabels(char *basepath, int *truelabel, int number_of_kernels);
float *iftGFLIMLoadBias(char *basepath);
int *iftGFLIMLoadKernelLabels(char *filename, int *nkernels);
void iftGFLIMSaveMergedModel(char *basepath, iftMatrix *K, float *bias, int *truelabels);
void iftGFLIMLoadMergedModel(char *basepath, iftMatrix **K, float **bias);

// ============ CREATE, CONVERT, AND UPDATE GRAPH-BASED MARKERS ============ //
iftLabeledSet *iftGFLIMImageMarkersToGraphMarkers(iftImage *label_img, 
                                                  iftLabeledSet *S, 
                                                  iftGFLIMStrideType stride,
                                                  int img_xsize, int img_ysize);
iftLabeledSet *iftGFLIMUpdateGraphMarkers(iftLabeledSet *S, 
                                        iftImage *label_img, 
                                        iftImage *label_img0);

// ============ CONVERT FROM SEEDS/GRAPH TO DATASET ============ //
iftLabeledSet *iftGFLIMSelectClusterCenters(iftFLIMGraph *graph, 
                                            iftLabeledSet *S, 
                                            int kernel_size, 
                                            int nclusters_per_marker, 
                                            iftFLIMGraphAttention attention_opt,
                                            iftGFLIMDimReductionType dimReduction, 
                                            int ncomponents,
                                            iftGFLIMAggregationType aggregation_opt);
iftDataSet *iftGFLIMExtractMarkerDataSet(iftFLIMGraph *graph,
                                              iftLabeledSet *S,
                                              int kernel_size,
                                              iftFLIMGraphAttention attention_opt,
                                              iftGFLIMAggregationType aggregation_opt);
          
// ============ FUNCTIONS TO CREATE A LAYER ============ //
iftMatrix *iftGFLIMComputeKernelsAndTrueLabels(iftDataSet *Z, int *truelabel);
float *iftGFLIMComputeBias(iftMatrix *kernels, iftDataSet *Z);
iftMatrix *iftGFLIMCreateLayerModel(iftFLIMGraph *graph, 
                                    iftLabeledSet *S,
                                    iftImage *label_img0, iftImage *label_img, 
                                    iftFLIMLayer layer, float stdev_factor,
                                    iftGFLIMStrideType stride, 
                                    float **bias, int **truelabel,
                                    iftFLIMGraphAttention attention_opt,
                                    iftGFLIMAggregationType aggregation_opt);

// ============ FUNCTIONS FOR STRIDE ============ //
iftFLIMGraph *iftGFLIMImageStride(iftFLIMGraph *graph, 
                                  iftImage *input_labels, 
                                  iftImage *output_labels, 
                                  int stride);
iftFLIMGraph *iftGFLIMGraphStride(iftFLIMGraph *graph, int stride, int *nodes_map);

// ============ FUNCTIONS TO ENCODE LAYER ============ //
void iftGFLIMPoolingLayer(iftFLIMGraph *graph, 
                        char *pool_type, 
                        int pool_size, 
                        int pool_stride);
iftMImageArray *iftGFLIMSumMImgArrayToFloatArray(iftMImageArray *mimgArray, 
                                                float *bias, bool inplace);
iftMImage *iftGFLIMConvLayer(iftFLIMGraph *graph, iftMatrix *kernels,
                              int kernel_size, iftFLIMGraphAttention attention_opt,
                              int *kernel_labels, iftGFLIMAggregationType aggregation_opt,
                              /*bool *maintain_kernels,*/ bool reduce);
iftMImageArray *iftGFLIMConvLayerBatch(iftFLIMGraphArray *graph, 
                                      iftMatrix *kernels, 
                                      int kernel_size, 
                                      iftFLIMGraphAttention attention_opt, 
                                      int *kernel_labels, 
                                      iftGFLIMAggregationType aggregation_opt, 
                                      /*bool **maintain_kernels,*/ bool reduce);
void iftGFLIMPoolingArray(iftMImageArray *activ_array, iftFLIMGraphArray *graph, 
                          char *pool_type, int pool_size, int pool_stride);
iftFLIMGraph *iftGFLIMEncodeLayer(iftFLIMGraph *graph, 
                            iftMatrix *Kernels, float *bias,
                            iftFLIMLayer layer, iftGFLIMStrideType stride,
                            iftImage *input_labels, iftImage **output_labels, 
                            iftFLIMGraphAttention attention_opt, int *kernel_labels,
                            iftGFLIMAggregationType aggregation_opt, bool reduce);
void iftGFLIMEncodeLayerBatch(iftFLIMGraphArray *input, 
                              iftMatrix *Kmerged, iftFLIMLayer layer,
                              char *in_labels_dir, char *out_labels_dir,
                              float *bias_merged,
                              char **basename, int nbasenames,
                              iftGFLIMStrideType stride, int layer_index,
                              iftFLIMGraphAttention attention_opt, int *kernel_labels,
                              iftGFLIMAggregationType aggregation_opt, bool reduce);
void iftGFLIMEncodeLayerBatch_v2(iftFLIMGraphArray *input,
                                iftMatrix *Kmerged, iftFLIMLayer layer,
                                iftImageArray *labels_img,
                                float *bias_merged,
                                iftGFLIMStrideType stride,
                                iftFLIMGraphAttention attention_opt, int *kernel_labels,
                                iftGFLIMAggregationType aggregation_opt, bool reduce);
// ============ FUNCTIONS TO DECODE FEATURES ============ //
iftImageArray *iftGFLIMDecodeLayer(iftFLIMGraph *graph, 
                                  iftImage *label_img, 
                                  iftImage *label_img0, 
                                  int *truelabels, int nlabels,
                                  iftGFLIMDecoderType decoder);

// =========== FUNCTION TO EXTRACT FEATURES ============ //
iftFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch(iftFileSet *fs, char *labels_dir, 
                                int first, int last, 
                                int initial_seeds, int nsuperpixels,
                                char **basename,
                                iftFLIMGraphFeature graph_feat,
                                iftFLIMGraphType graph_type);
iftFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch_v2(iftFileSet *fs,
                                  int first, int last,
                                  int initial_seeds, int nsuperpixels,
                                  iftFLIMGraphFeature graph_feat,
                                  iftFLIMGraphType graph_type,
                                  iftImageArray *label_imgs);

#ifdef __cplusplus
}
#endif


#endif
