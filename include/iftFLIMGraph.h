/**
 * @file iftFLIMGraph.h
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

typedef struct iftFLIMGraphNode iftFLIMGraphNode;

/**
 * @struct iftFLIMGraphNode
 * @brief Node of a graph for GFLIM.
 * @see iftFLIMGraph
 */
typedef struct iftFLIMGraphNode{
    double  *feats;             // Features of the node
    unsigned long numNeighbors; // Number of neighbors
    unsigned long numPixels;    // Number of pixels in the node (considering its a superpixel)
    iftSet *adj;                // Adjacency list of the node, in which each element contains the index of a neighbor node 
}iftFLIMGraphNode;

/**
 * @struct iftFLIMGraph
 * @brief Graph for GFLIM.
 */
typedef struct _iftFLIMGraph {
    iftFLIMGraphNode *nodes;    // Array of nodes. @see iftFLIMGraphNode 
    unsigned long num_nodes;    // Number of nodes
    unsigned long num_feats;    // Number of features in each node
    unsigned long num_edges;    // Number of edges
} iftFLIMGraph;

/**
 * @struct iftFLIMGraphArray
 * @brief Array of graphs iftFLIMGraph.
 * @see iftGFLIMGraphArrayAlloc()
 * @see iftGFLIMGraphArrayFree()
 */
typedef struct iftFLIMGraphArray{
    iftFLIMGraph **val; // Array of graphs. @see iftFLIMGraph 
    int n;              // Number of graphs.
} iftFLIMGraphArray;

/**
 * @enum iftFLIMGraphAttention
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
} iftFLIMGraphAttention;

/**
 * @enum iftFLIMGraphFeature
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
    IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK               // concatenated sum of color with haralick features
} iftFLIMGraphFeature;

/**
 * @enum iftFLIMGraphType
 * @brief Enumerates the superpixel methods to apply in an input image.
 * A FLIMGraph is created from the superpixels.
 */
typedef enum {
    IFT_FLIMGRAPH_TYPE_DISF, // DISF. @see iftDISF() at iftSegmentation.c
    IFT_FLIMGRAPH_TYPE_ERGC, // ERGC. @see iftERGC() at iftFLIMGraph.c
} iftFLIMGraphType;

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
iftFLIMGraphArray *iftCreateEmptyFLIMGraphArray(int n);
void iftDestroyFLIMGraphArray(iftFLIMGraphArray **arr);

/* --------------------- FLIMGraph construction ----------------------------- */
iftFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes);
void iftDestroyFLIMGraph(iftFLIMGraph **graph);
void iftInsertFLIMGraphNeighborNode(iftFLIMGraphNode *node, int neighbor);
bool iftIsValidFLIMGraph(iftFLIMGraph *graph);

/* --------------------- FLIMGraph convert from string to a type ----------------------------- */
iftFLIMGraphFeature iftGFLIMGetGraphFeature(const char *name);
iftFLIMGraphType iftGFLIMGetGraphType(const char *name);
iftFLIMGraphAttention iftGLIMGraphGetAttentionType(char *arg);
iftFLIMGraphAttention iftGLIMGraphGetAggregationType(char *arg);

/* --------------------- FLIMGraph read and write ----------------------------- */
iftFLIMGraph *iftReadFLIMGraph(char *filename);
void iftWriteFLIMGraph(iftFLIMGraph *graph, char *filename);

/* --------------------- FLIMGraph transform ----------------------------- */
void iftNormalizeFLIMGraphByZScore(iftFLIMGraph *graph, float *mean, float *stdev);
iftFLIMGraph *iftImageToFLIMGraph(iftImage *img, 
    int num_superpixels, int param,
    iftImage **label_img,
    iftFLIMGraphFeature graph_feat,
    iftFLIMGraphType graph_type);
iftMImage *iftFLIMGraphToMImage(iftFLIMGraph *graph, iftImage *labels);
iftFLIMGraph *iftMImageToFLIMGraph(iftMImage *mimg,
    iftFLIMGraph *graph_ref);
void iftMImageToFLIMGraphInplace(iftMImage *mimg, iftFLIMGraph *graph_ref);
iftImage **iftFLIMGraphToActivations(iftFLIMGraph *graph, iftImage *labels);

/* --------------------- FLIMGraph attention ----------------------------- */
double *iftFLIMGraphAttentionRelNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors);
double *iftFLIMGraphAttentionNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors);
iftMatrix *iftFLIMGraphAttentionNeighborSimilarity(iftMatrix *features, iftIntArray *neighbors, int *kernel_labels, int label);
double *iftFLIMGraphGetAttentionValues(iftIntArray *neighbors, iftFLIMGraph *graph, iftFLIMGraphAttention *attention_opt);
iftMatrix * iftFLIMGraphAttentionClassDominance_test(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance_v2(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominancePower2(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance_softmaxNeigh(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
double *iftFLIMGraphAttentionCosineSimilarity(iftFLIMGraph *graph, iftIntArray *neighbors);

/* --------------------- FLIMGraph neighbor features ----------------------------- */
double *iftFLIMgraphAggregateFeatures(int p, 
    iftIntArray *neighbors, 
    iftFLIMGraph *graph,
    int num_adj,
    iftGFLIMAggregationType aggregation_opt);

iftMatrix *iftFLIMGraphToFeatureMatrix(iftFLIMGraph *graph,
                                   int num_adj,
                                   iftFLIMGraphAttention attention_opt,
                                   iftGFLIMAggregationType aggregation_opt/*,
                                   iftIntArray **neighbors_all*/);

iftFLIMGraph *iftFeatureMatrixToFLIMGraph(iftMatrix *matrix,
                                   iftFLIMGraph *graph_ref);

iftIntArray *iftGetKNeighborsFLIMGraph(int node_index, iftFLIMGraph *graph, int k,
                           int dilation, bool most_similar, iftFLIMGraphAttention attention_opt);


#ifdef __cplusplus
}
#endif
#endif


