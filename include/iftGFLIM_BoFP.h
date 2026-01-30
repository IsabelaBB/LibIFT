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
  IFT_GFLIM_GRAPHSTRIDE,   // Stride based on the graph size. It reduces the graph nodes by edge contraction based on a nodes set size faction. @see iftGFLIMGraphStride()
  IFT_GFLIM_PRECOMPUTED_GRAPH
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
void iftGFLIMUpdateGraphMarkers(iftLabeledSet *S, 
                                        iftImage *label_img, 
                                        iftImage *label_img0);
void iftGFLIMUpdateMarkersFromPrecomputedGraphs(iftLabeledSet **T,
                                        iftGFLIMLayer layer,
                                        iftImage *label_img,
                                        iftImage *label_img0);
// ============ CONVERT FROM SEEDS/GRAPH TO DATASET ============ //
iftLabeledSet *iftGFLIMSelectClusterCenters(iftGFLIMGraph *graph, 
                                            iftLabeledSet *S, 
                                            int kernel_size, 
                                            int nclusters_per_marker, 
                                            iftGFLIMGraphAttention attention_opt,
                                            iftGFLIMDimReductionType dimReduction, 
                                            int ncomponents,
                                            iftGFLIMAggregation aggregation_opt);
iftDataSet *iftGFLIMExtractMarkerDataSet(iftGFLIMGraph *graph,
                                              iftLabeledSet *S,
                                              int kernel_size,
                                              iftGFLIMGraphAttention attention_opt,
                                              iftGFLIMAggregation aggregation_opt);
iftLabeledSet *iftGFLIMSelectClusterCentersV3(iftGFLIMGraph *graph,
                                            iftLabeledSet *S,
                                            int kernel_size,
                                            int nclusters_per_marker,
                                            iftMatrix *neighbors,
                                            iftGFLIMAggregation aggregation_opt);
// ============ FUNCTIONS TO CREATE A LAYER ============ //
iftMatrix *iftGFLIMComputeKernelsAndTrueLabels(iftDataSet *Z, int *truelabel);
float *iftGFLIMComputeBias(iftMatrix *kernels, iftDataSet *Z);
iftMatrix *iftGFLIMCreateLayerModel(iftGFLIMGraph *graph, 
                                    iftLabeledSet *S,
                                    iftImage *label_img0, iftImage *label_img, 
                                    iftGFLIMLayer layer, float stdev_factor,
                                    iftGFLIMStrideType stride, 
                                    float **bias, int **truelabel,
                                    iftGFLIMGraphAttention attention_opt,
                                    iftGFLIMAggregation aggregation_opt);
iftMatrix *iftGFLIMCreateLayerModelV3(iftGFLIMGraph *graph, iftLabeledSet *S,
                                    iftGFLIMLayer layer, float stdev_factor,
                                    float **bias, int **truelabel,
                                    iftMatrix *neighbors,
                                    char *kernel_neighbors,
                                    iftGFLIMAggregation aggregation_opt);
// ============ FUNCTIONS FOR STRIDE ============ //
iftGFLIMGraph *iftGFLIMImageStride(iftGFLIMGraph *graph, 
                                  iftImage *input_labels, 
                                  iftImage *output_labels, 
                                  int stride);
iftGFLIMGraph *iftGFLIMGraphStride(iftGFLIMGraph *graph, 
                                  float stride, 
                                  int *nodes_map);

// ============ FUNCTIONS TO ENCODE LAYER ============ //
iftMImageArray *iftGFLIMSumMImgArrayToFloatArray(iftMImageArray *mimgArray, 
                                                float *bias, bool inplace);
iftMImage *iftGFLIMConvLayer(iftGFLIMGraph *graph, iftMatrix *kernels,
                              int kernel_size, iftGFLIMGraphAttention attention_opt,
                              int *kernel_labels, iftGFLIMAggregation aggregation_opt,
                              bool reduce);
iftMImage *iftGFLIMConvLayerV2(iftGFLIMGraph *graph, iftMatrix *kernels,
                              iftGFLIMGraphAttention attention_opt,
                              int *kernel_labels, int kernel_size,
                              iftMatrix *neighbors,
                              iftGFLIMAggregation aggregation_opt);
iftMImageArray *iftGFLIMConvLayerBatch(iftGFLIMGraphArray *graph, 
                                      iftMatrix *kernels, 
                                      int kernel_size, 
                                      iftGFLIMGraphAttention attention_opt, 
                                      int *kernel_labels, 
                                      iftGFLIMAggregation aggregation_opt, 
                                      bool reduce);
iftMImageArray *iftGFLIMConvLayerBatchV2(iftGFLIMGraphArray *graph, 
                                      iftMatrix *kernels, 
                                      iftGFLIMGraphAttention attention_opt, 
                                      int *kernel_labels, int kernel_size,
                                      iftMatrix **neighbors,
                                      iftGFLIMAggregation aggregation_opt);
void iftGFLIMPoolingArray(iftMImageArray *activ_array, iftGFLIMGraphArray *graph, 
                          char *pool_type, int pool_size);
void iftGFLIMPoolingArrayV2(iftMImageArray *activ_array, iftGFLIMGraphArray *graph,
                          char *pool_type, int pool_size, iftMatrix **neighbors);
iftGFLIMGraph *iftGFLIMEncodeLayer(iftGFLIMGraph *graph, 
                            iftMatrix *Kernels, float *bias,
                            iftGFLIMLayer layer, iftGFLIMStrideType stride,
                            iftImage *input_labels, iftImage **output_labels, 
                            iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                            iftGFLIMAggregation aggregation_opt, bool reduce);
iftGFLIMGraph *iftGFLIMEncodeLayerV2(iftGFLIMGraph *graph,
                                  iftMatrix *Kernels, 
                                  float *bias,
                                  iftGFLIMLayer layer, 
                                  iftGFLIMGraphAttention attention_opt, 
                                  int *kernel_labels,
                                  iftMatrix *neighbors,
                                  char *kernel_neighbors,
                                  char *pool_neighbors,
                                  iftGFLIMAggregation aggregation_opt);
void iftGFLIMEncodeLayerBatch(iftGFLIMGraphArray *input,
                                 iftMatrix *Kmerged, iftGFLIMLayer layer,
                                 iftImageArray *labels_img,
                                 float *bias_merged,
                                 iftGFLIMStrideType stride,
                                 iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                                 iftGFLIMAggregation aggregation_opt, bool reduce);
void iftGFLIMEncodeLayerBatchV2(iftGFLIMGraphArray *input,
                                 iftMatrix *Kmerged, 
                                 iftGFLIMLayer layer,
                                 float *bias_merged,
                                 iftGFLIMGraphAttention attention_opt, 
                                 int *kernel_labels,
                                 iftMatrix **neighbors,
                                 char *kernel_neighbors,
                                 char *pool_neighbors,
                                  iftGFLIMAggregation aggregation_opt);
void iftGFLIMStrideBatch(iftGFLIMGraphArray *graph, iftGFLIMLayer layer,
                            iftImageArray *labels_img, iftGFLIMStrideType stride);
void iftGFLIMStridePrecomputed(iftGFLIMGraph *graph,
                            iftGFLIMGraph *graph_resized,
                            iftImage *label_img,
                            iftImage *label_img_resized,
                            float *scale);
// ============ FUNCTIONS TO DECODE FEATURES ============ //
iftImageArray *iftGFLIMDecodeLayer(iftGFLIMGraph *graph, 
                                  iftImage *label_img, 
                                  iftImage *label_img0, 
                                  int *truelabels, int nlabels,
                                  iftGFLIMDecoderType decoder);

// =========== FUNCTION TO EXTRACT FEATURES ============ //
iftGFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch_bkp(iftFileSet *fs, char *labels_dir, 
                                int first, int last, 
                                int initial_seeds, int nsuperpixels,
                                char **basename,
                                iftGFLIMGraphNodeFeature graph_feat,
                                iftGFLIMGraphSpxMethod graph_type,
                                bool use_mask);
iftGFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch(iftFileSet *fs,
                                  int first, int last,
                                  int initial_seeds, int nsuperpixels,
                                  iftGFLIMGraphNodeFeature graph_feat,
                                  iftGFLIMGraphSpxMethod graph_type,
                                  iftImageArray *label_imgs,
                                  bool use_mask);

void iftGFLIMSkipConnectionBatch(iftGFLIMGraphArray *input, iftImageArray *label_imgs,
                            iftFileSet *fs, int first, int last, int layer, 
                            iftGFLIMArch *arch, char *kernels_file, 
                            char *model_dir);

void iftGFLIMSkipConnection(iftGFLIMArch *arch, int layer, char *kernels_file, 
                            char *model_dir, iftFileSet *fs);
                            
int* iftGFLIMGetFinerResidualLayer(iftGFLIMArch *arch);
iftIntArray **iftGFLIMCompute_kernel_maps(char *filename, int *nscales);
iftMatrix *iftGFLIMMergeMatrices_dotProd(iftMatrix **X, int n, 
                                        int finer_scale, bool use_otsu, bool use_std);
iftMatrix *iftGFLIMMergeMatrices_dotProd_sumScale0(iftMatrix **X, int n, 
                                        int finer_scale, bool use_otsu, bool use_std);
iftMatrix *iftGFLIMMergeMatrices_dotProd_sumScalei(iftMatrix **X, int n, 
                                        int finer_scale, bool use_otsu, bool use_std);
iftMatrix *iftGFLIMMergeMatrices_sum(iftMatrix **X, int n, 
                                        int finer_scale, bool use_otsu, bool use_std,
                                        int reinforce_scale);
iftMatrix *iftGFLIMMergeMatrices_sum_att(iftMatrix **X, int n, 
                                        int *kernel_labels,
                                        int finer_scale,
                                    iftGFLIMGraph *graph);
int *iftGFLIMComputeNodeMap(iftImage *spx_imgi, iftImage *spx_img0);
void iftGFLIMMaxNormGraph(iftGFLIMGraph *graph);
iftMatrix *iftGFLIMUpsampleFeatureMatrix(iftMatrix *X, 
                                    int *nodes_map, iftIntArray *kernels_map, 
                                    int nnodes, int nfeats);

int getLayersToConcat(iftGFLIMArch *arch, int layer, int skip_layer, iftIntArray *layers_to_concat, int index);

#ifdef __cplusplus
}
#endif


#endif
