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
 * @enum iftGFLIMGraphFeature
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
} iftGFLIMGraphFeature;

/**
 * @enum iftGFLIMGraphType
 * @brief Enumerates the superpixel methods to apply in an input image.
 * A FLIMGraph is created from the superpixels.
 */
typedef enum {
    IFT_FLIMGRAPH_TYPE_DISF, // DISF. @see iftDISF() at iftSegmentation.c
    IFT_FLIMGRAPH_TYPE_ERGC, // ERGC. @see iftERGC() at iftGFLIMGraph.c
} iftGFLIMGraphType;

/**
 * @enum iftGFLIMAggregationType
 * @brief Enumerates ways to aggregate feature in a FLIMGraph.
 * In GFLIM-BoFP, the aggregation is done among the neighbor features of the node.
 * In GFLIM, the aggregation is CONCATENATE.
 */
typedef enum {
    IFT_GFLIM_AGGREGATION_CONCAT,   // Concatenate the features of the nodes
    IFT_GFLIM_AGGREGATION_MEAN,     // Average the features of the nodes
    IFT_GFLIM_AGGREGATION_WMEAN     // Weighted mean of the features of the nodes
  } iftGFLIMAggregationType;


/* --------------------- GraphArray functions ----------------------------- */
iftGFLIMGraphArray *iftCreateEmptyFLIMGraphArray(int n);
void iftDestroyFLIMGraphArray(iftGFLIMGraphArray **arr);

/* --------------------- FLIMGraph construction ----------------------------- */
iftGFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes);
void iftDestroyFLIMGraph(iftGFLIMGraph **graph);
void iftInsertFLIMGraphNeighborNode(iftGFLIMGraphNode *node, int neighbor);
bool iftIsValidFLIMGraph(iftGFLIMGraph *graph);
iftGFLIMGraph *iftCopyFLIMGraph(int num_feats, iftGFLIMGraph *graph_orig, bool copy_features);
iftGFLIMGraph *ComputeGraphFeatureMean(
                    unsigned long num_superpixels, 
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureSum(
                    unsigned long num_superpixels,
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureSumMultRelsize(
                    unsigned long num_superpixels,
                    iftAdjRel *A, iftMImage *oiginal_img, 
                    iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureMeanMultRelsize(
    unsigned long num_superpixels,
    iftAdjRel *A, iftMImage *oiginal_img, 
    iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureMeanConcatRelsize(
                unsigned long num_superpixels,
                iftAdjRel *A,
                iftMImage *oiginal_img, iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureMeanConcatRelsizeNeighb(
        unsigned long num_superpixels,
        iftAdjRel *A, iftMImage *oiginal_img, 
        iftImage *labeled_img);
iftGFLIMGraph *ComputeGraphFeatureMeanConcatRelsizeHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A, iftMImage *oiginal_img, 
    iftImage *labeled_img,
    iftImage *gray_img);
iftGFLIMGraph *ComputeGraphFeatureSumMultRelsizeConcatHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img,
    iftImage *gray_img);
iftGFLIMGraph *ComputeGraphFeatureSumConcatHaralick(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img,
    iftGFLIMGraphFeature feat_type,
    iftImage *gray_img);
/* --------------------- FLIMGraph convert from string to a type ----------------------------- */
iftGFLIMGraphFeature iftGFLIMGetGraphFeature(const char *name);
iftGFLIMGraphType iftGFLIMGetGraphType(const char *name);
iftGFLIMGraphAttention iftGLIMGraphGetAttentionType(char *arg);
iftGFLIMGraphAttention iftGLIMGraphGetAggregationType(char *arg);

/* --------------------- FLIMGraph read and write ----------------------------- */
iftGFLIMGraph *iftReadFLIMGraph(char *filename);
void iftWriteFLIMGraph(iftGFLIMGraph *graph, char *filename);

/* --------------------- FLIMGraph transform ----------------------------- */
void iftNormalizeFLIMGraphByZScore(iftGFLIMGraph *graph, float *mean, float *stdev);
iftGFLIMGraph *iftImageToFLIMGraph(iftImage *img, 
    int num_superpixels, int param,
    iftImage **label_img,
    iftGFLIMGraphFeature graph_feat,
    iftGFLIMGraphType graph_type);
iftMImage *iftGFLIMGraphToMImage(iftGFLIMGraph *graph, iftImage *labels);
iftGFLIMGraph *iftMImageToFLIMGraph(iftMImage *mimg,
    iftGFLIMGraph *graph_ref);
void iftMImageToFLIMGraphInplace(iftMImage *mimg, iftGFLIMGraph *graph_ref);
iftImage **iftGFLIMGraphToActivations(iftGFLIMGraph *graph, iftImage *labels);

/* --------------------- FLIMGraph attention ----------------------------- */
double *iftGFLIMGraphGetAttentionValues(iftIntArray *neighbors, iftGFLIMGraph *graph, iftGFLIMGraphAttention *attention_opt);

iftMatrix * iftGFLIMGraphAttentionClassDominance(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix *iftGFLIMGraphAttentionClassDominancePower2(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);

double *iftGFLIMGraphAttentionRelNodeSize(iftGFLIMGraph *graph, iftIntArray *neighbors);
double *iftGFLIMGraphAttentionNodeSize(iftGFLIMGraph *graph, iftIntArray *neighbors);
iftMatrix *iftGFLIMGraphAttentionNeighborSimilarity(iftMatrix *features, iftIntArray *neighbors, int *kernel_labels, int label);
iftMatrix * iftGFLIMGraphAttentionClassDominance_test(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftGFLIMGraphAttentionClassDominance_v2(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftGFLIMGraphAttentionClassDominance_softmaxNeigh(iftMatrix *H, iftGFLIMGraph *graph, int *kernel_labels);
double *iftGFLIMGraphAttentionCosineSimilarity(iftGFLIMGraph *graph, iftIntArray *neighbors);

/* --------------------- FLIMGraph neighbor features ----------------------------- */
double *iftGFLIMGraphAggregateFeatures(int p, 
    iftIntArray *neighbors, 
    iftGFLIMGraph *graph,
    int num_adj,
    iftGFLIMAggregationType aggregation_opt);

iftMatrix *iftGFLIMGraphToFeatureMatrix(iftGFLIMGraph *graph,
                                   int num_adj,
                                   iftGFLIMGraphAttention attention_opt,
                                   iftGFLIMAggregationType aggregation_opt/*,
                                   iftIntArray **neighbors_all*/);

iftGFLIMGraph *iftFeatureMatrixToFLIMGraph(iftMatrix *matrix,
                                   iftGFLIMGraph *graph_ref);

iftIntArray *iftGetKNeighborsFLIMGraph(int node_index, iftGFLIMGraph *graph, int k,
                           int dilation, bool most_similar, iftGFLIMGraphAttention attention_opt);


#ifdef __cplusplus
}
#endif
#endif


