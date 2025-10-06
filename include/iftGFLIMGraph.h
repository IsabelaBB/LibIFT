/**
 * @file iftGFLIMGraph.h
 * @brief Provide functions to operate on Graphs for GFLIM or GFLIM-BoFP.
 * @author Isabela Borlido Barcelos
 */

#ifndef IFT_FLIMGRAPH_H
#define IFT_FLIMGRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ift/core/dtypes/Set.h"
#include "iftAdjacency.h"
#include "iftMImage.h"
#include "iftImage.h"
#include "ift/core/dtypes/DHeap.h"
#include "ift/core/dtypes/BasicDataTypes.h"
#include "ift/core/io/Json.h"
#include "ift/core/io/Dir.h"
#include "iftSegmentation.h"
#include<locale.h>

#define	LC_ALL		  __LC_ALL

/* --------------------- GFLIMGraph structures definition ----------------------------- */

typedef struct iftGFLIMGraphNode iftGFLIMGraphNode;

/**
 * @struct iftGFLIMGraphNode
 * @brief Node of a graph for GFLIM.
 * @see iftGFLIMGraph
 */
typedef struct iftGFLIMGraphNode{
    double  *feats;             // Features of the node
    unsigned long numNeighbors; // Number of neighbors
    unsigned long numPixels;    // Number of pixels in the node (considering its a superpixel)
    iftSet *adj;                // Adjacency list of the node, in which each element contains the index of a neighbor node
}iftGFLIMGraphNode;

/**
 * @struct iftGFLIMGraph
 * @brief Graph for GFLIM.
 */
typedef struct _iftGFLIMGraph {
    iftGFLIMGraphNode *nodes;    // Array of nodes. @see iftGFLIMGraphNode
    unsigned long num_nodes;    // Number of nodes
    unsigned long num_feats;    // Number of features in each node
    unsigned long num_edges;    // Number of edges
} iftGFLIMGraph;

/* --------------------- GFLIMGraphArray structure definition ----------------------------- */

/**
 * @struct iftGFLIMGraphArray
 * @brief Array of graphs iftGFLIMGraph.
 * @see iftGFLIMGraphArrayAlloc()
 * @see iftGFLIMGraphArrayFree()
 */
typedef struct iftGFLIMGraphArray{
    iftGFLIMGraph **val; // Array of graphs. @see iftGFLIMGraph 
    int n;              // Number of graphs.
} iftGFLIMGraphArray;

/* --------------------- Enums for types in GFLIMGraph ----------------------------- */

/**
 * @enum iftGFLIMGraphAttention
 * @brief Enumerates the attention types for GFLIM.
 */
typedef enum {
    IFT_FLIMGRAPH_ATTENTION_NONE,           // No attention
    IFT_FLIMGRAPH_ATTENTION_NODE_RELSIZE,   // number of pixels by the total number of pixels in the graph. No softmax
    IFT_FLIMGRAPH_ATTENTION_NODE_SIZE,      // number of pixels in the neighborhood
    //IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL,   // neighbors similarity --> removed
    IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIZE,    // not implemented anymore // number of pixels and neighbors similarity
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_TEST, 
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE,                // best attention so far
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_POWER2,
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_SOFTMAX_NEIGHB,
    IFT_FLIMGRAPH_ATTENTION_COSINE_SIMILARITY
} iftGFLIMGraphAttention;

/**
 * @enum iftGFLIMGraphNodeFeature
 * @brief Enumerates the feature types for the nodes in FLIMGraph.
 */
typedef enum {
    IFT_FLIMGRAPH_FEATURE_MEAN,                             // mean pixel color
    IFT_FLIMGRAPH_FEATURE_SUM,                              // sum of pixel color
    IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE,                 // sum of color multiplied by 1 + (superpixel size / image size)
    IFT_FLIMGRAPH_FEATURE_MEAN_MULT_RELSIZE,                // mean of color multiplied by 1 + (superpixel size / image size)
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE,              // concatenate the mean color with (superpixel size / image size)
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB,       // concatenate the mean color with (superpixel size / neighborhood size)
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK,     // concatenate the mean color with (superpixel size / neighborhood size) and with haralick features
    IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK, // sum of color multiplied by 1 + (superpixel size / image size) and concatenated with haralick features
    IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK,              // concatenated sum of color with haralick features
    IFT_FLIMGRAPH_FEATURE_RBD                               // RBD descriptor 
} iftGFLIMGraphNodeFeature;

/**
 * @enum iftGFLIMGraphSpxMethod
 * @brief Enumerates the superpixel methods to apply in an input image.
 * A FLIMGraph is created from the superpixels.
 */
typedef enum {
    IFT_FLIMGRAPH_TYPE_DISF, // DISF. @see iftDISF() at iftSegmentation.c
    IFT_FLIMGRAPH_TYPE_ERGC, // ERGC. @see iftERGC() at iftGFLIMGraph.c
} iftGFLIMGraphSpxMethod;

/**
 * @enum iftGFLIMAggregation
 * @brief Enumerates ways to aggregate feature in a FLIMGraph.
 * In GFLIM-BoFP, the aggregation is done among the neighbor features of the node.
 * In GFLIM, the aggregation is CONCATENATE.
 */
typedef enum {
    IFT_GFLIM_AGGREGATION_CONCAT,   // Concatenate the features of the nodes
    IFT_GFLIM_AGGREGATION_MEAN,     // Average the features of the nodes
    IFT_GFLIM_AGGREGATION_WMEAN     // Weighted mean of the features of the nodes
  } iftGFLIMAggregation;

/* --------------------- GraphArray functions ----------------------------- */
iftGFLIMGraphArray *iftCreateGFLIMGraphArray(int n);
void iftDestroyGFLIMGraphArray(iftGFLIMGraphArray **arr);

/* --------------------- FLIMGraph construction ----------------------------- */
iftGFLIMGraph *iftCreateGFLIMGraph(int num_feats, int num_nodes);
void iftDestroyGFLIMGraph(iftGFLIMGraph **graph);
void iftGFLIMGraphInsertNeighborNode(iftGFLIMGraphNode *node, int neighbor);
iftGFLIMGraph *iftCopyGFLIMGraph(int num_feats, iftGFLIMGraph *graph_orig, bool copy_features);
bool iftIsValidGFLIMGraph(iftGFLIMGraph *graph);

/* --------------------- GFLIMGraph transform ----------------------------- */
void iftGFLIMGraphZNormalization(iftGFLIMGraph *graph, float *mean, float *stdev);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesMean(
                    unsigned long num_superpixels, 
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesSum(
                    unsigned long num_superpixels,
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesSumMultRelsize(
                    unsigned long num_superpixels,
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesMeanMultRelsize(
    unsigned long num_superpixels,
    iftAdjRel *A, iftMImage *oiginal_img, 
    iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesMeanConcatRelsize(
                unsigned long num_superpixels,
                iftAdjRel *A,
                iftMImage *oiginal_img, iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesMeanConcatRelsizeNeighb(
        unsigned long num_superpixels,
        iftAdjRel *A, iftMImage *oiginal_img, 
        iftImage *labeled_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesMeanConcatRelsizeHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A, iftMImage *oiginal_img, 
    iftImage *labeled_img,
    iftImage *gray_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesSumMultRelsizeConcatHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img,
    iftImage *gray_img);
iftGFLIMGraph *iftComputeGFLIMGraphFeaturesSumConcatHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img,
    iftGFLIMGraphNodeFeature feat_type,
    iftImage *gray_img);

iftGFLIMGraph *iftImageToGFLIMGraph(iftImage *img, 
    int num_superpixels, int param,
    iftImage **label_img,
    iftGFLIMGraphNodeFeature graph_feat,
    iftGFLIMGraphSpxMethod graph_type);
iftMImage *iftGFLIMGraphToMImage(iftGFLIMGraph *graph, iftImage *labels);
iftGFLIMGraph *iftMImageToGFLIMGraph(iftMImage *mimg,
    iftGFLIMGraph *graph_ref);
void iftMImageToGFLIMGraphInplace(iftMImage *mimg, iftGFLIMGraph *graph_ref);
iftImage **iftGFLIMGraphToActivations(iftGFLIMGraph *graph, iftImage *labels);

iftMatrix *iftGFLIMGraphToNodeFeaturesMatrix(iftGFLIMGraph *graph);
iftGFLIMGraph *iftFeatureMatrixToGFLIMGraph(iftMatrix *matrix,
                                   iftGFLIMGraph *graph_ref);

/* --------------------- GFLIMGraph attention ----------------------------- */
double *iftGFLIMGraphAttentionRelNodeSize(iftGFLIMGraph *graph, iftIntArray *neighbors);
double *iftGFLIMGraphAttentionNodeSize(iftGFLIMGraph *graph, iftIntArray *neighbors);
iftMatrix *iftGFLIMGraphAttentionNeighborSimilarity(iftMatrix *features, iftIntArray *neighbors, int *kernel_labels, int label);
double *iftGFLIMGraphAttentionCosineSimilarity(iftGFLIMGraph *graph, iftIntArray *neighbors);
iftMatrix * iftGFLIMGraphAttentionClassDominance_test(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftGFLIMGraphAttentionClassDominance_softmaxNeigh(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftGFLIMGraphAttentionClassDominance_v2(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftGFLIMGraphAttentionClassDominance(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix *iftGFLIMGraphAttentionClassDominancePower2(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);

double *iftGFLIMGraphGetAttentionValues(iftIntArray *neighbors, iftGFLIMGraph *graph, iftGFLIMGraphAttention *attention_opt);

/* --------------------- FLIMGraph neighbor features ----------------------------- */
iftIntArray *iftFLIMGraphGetKNeighborsAggregation(int node_index, iftGFLIMGraph *graph, int k,
                           int dilation, bool most_similar, iftGFLIMGraphAttention attention_opt);
double *iftGFLIMGraphAggregateFeatures(int p, 
    iftIntArray *neighbors, 
    iftGFLIMGraph *graph,
    int num_adj,
    iftGFLIMAggregation aggregation_opt);
iftMatrix *iftGFLIMGraphToNeighborhoodFeatureMatrix(iftGFLIMGraph *graph,
                                   int num_adj,
                                   iftGFLIMGraphAttention attention_opt,
                                   iftGFLIMAggregation aggregation_opt/*,
                                   iftIntArray **neighbors_all*/);

#ifdef __cplusplus
}
#endif
#endif


