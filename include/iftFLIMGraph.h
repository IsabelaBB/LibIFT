/**
 * @file 
 * @brief ...
 */

#ifndef IFT_FLIMGRAPH_H
#define IFT_FLIMGRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftSegmentation.h"
#include "ift/core/tools/Numerical.h"
#include "ift/core/io/Json.h"
#include "ift/core/io/Dir.h"

typedef struct iftFLIMGraphNode iftFLIMGraphNode;

typedef struct iftFLIMGraphNode{
    double *feats;
    unsigned long numNeighbors;
    unsigned long numPixels;
    iftSet *adj;
}iftFLIMGraphNode;

typedef struct _iftFLIMGraph {
    iftFLIMGraphNode *nodes;
    unsigned long num_nodes;
    unsigned long num_feats;
    unsigned long num_edges;
} iftFLIMGraph;

/**
 * @struct iftFLIMGraphArray
 * @brief Array of graphs for GFLIM.
 * @see iftGFLIMGraphArrayAlloc()
 * @see iftGFLIMGraphArrayFree()
 */
typedef struct iftFLIMGraphArray{
    iftFLIMGraph **val; // Array of graphs. @see iftFLIMGraph
    int n;              // Number of graphs.
} iftFLIMGraphArray;

typedef enum {
    IFT_FLIMGRAPH_ATTENTION_NONE,
    IFT_FLIMGRAPH_ATTENTION_NODE_RELSIZE, // number of pixels by the total number of pixels in the graph. No softmax
    IFT_FLIMGRAPH_ATTENTION_NODE_SIZE,    // number of pixels in the neighborhood
    IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL, // neighbors similarity
    IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIZE, // number of pixels and neighbors similarity
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE,
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_WINNER,
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_WINNER_POWER2,
    IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_SOFTMAX_NEIGHB,
    IFT_FLIMGRAPH_ATTENTION_COSINE_SIMILARITY
} iftFLIMGraphAttention;

typedef enum {
    IFT_FLIMGRAPH_FEATURE_MEAN, // color mean only
    IFT_FLIMGRAPH_FEATURE_SUM, // sum of color
    IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE, // sum of color multiplied by 1 + the relative size of the node
    IFT_FLIMGRAPH_FEATURE_MEAN_MULT_RELSIZE, // mean color multiplied by 1 + the relative size of the node
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE, // concatenate the relative spx. size with the mean color
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB, // concatenate the relative spx. size (normalized by the neighborhood) with the mean color
    IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK, // concatenate the relative spx. size (normalized by the neighborhood) with the mean color and haralick features
    IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK, // sum of color multiplied by 1 + the relative size of the node and concatenate with haralick features
    IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK // sum of color and concatenate with haralick features
} iftFLIMGraphFeature;

typedef enum {
    IFT_FLIMGRAPH_TYPE_DISF,
    IFT_FLIMGRAPH_TYPE_ERGC,
} iftFLIMGraphType;

typedef enum {
    IFT_GFLIM_AGGREGATION_CONCAT,
    IFT_GFLIM_AGGREGATION_MEAN,
    IFT_GFLIM_AGGREGATION_WMEAN
  } iftGFLIMAggregationType;

// ============ GraphArray CONTRUCTOR AND DESTRUCT ============ //
iftFLIMGraphArray *iftFLIMGraphCreateEmptyFLIMGraphArray(int n);
void iftDestroyFLIMGraphArray(iftFLIMGraphArray **arr);

iftFLIMGraphFeature iftGFLIMGetGraphFeature(const char *name);
iftFLIMGraphType iftGFLIMGetGraphType(const char *name);
iftFLIMGraphAttention iftGLIMGraphGetAttentionType(char *arg);

iftFLIMGraphAttention iftGLIMGraphGetAggregationType(char *arg);

double *iftFLIMGraphAttentionRelNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors);
double *iftFLIMGraphAttentionNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors);
iftMatrix *iftFLIMGraphAttentionNeighborSimilarity(iftMatrix *features, iftIntArray *neighbors, int *kernel_labels, int label);

double *iftFLIMGraphGetAttentionValues(iftIntArray *neighbors, iftFLIMGraph *graph, iftFLIMGraphAttention *attention_opt);
iftMatrix * iftFLIMGraphAttentionClassDominance(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance_winner(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance_winnerPower2(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);
iftMatrix * iftFLIMGraphAttentionClassDominance_softmaxNeigh(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels);

double *iftFLIMGraphAttentionCosineSimilarity(iftFLIMGraph *graph, iftIntArray *neighbors);

/* --------------------- FLIMGraph construction ----------------------------- */
// create / insert
iftFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes);
void iftInsertFLIMGraphNeighborNode(iftFLIMGraphNode *node, int neighbor);

// copy
iftFLIMGraph *iftCopyFLIMGraph(int num_feats, iftFLIMGraph *graph_orig, bool copy_features);

// read from / write to disk
iftFLIMGraph *iftReadFLIMGraph(char *filename);

void iftWriteFLIMGraph(iftFLIMGraph *graph, char *filename);

// validate
bool iftIsValidFLIMGraph(iftFLIMGraph *graph);

// destroy
void iftDestroyFLIMGraph(iftFLIMGraph **graph);

/* --------------------- FLIMGraph transform ----------------------------- */

void iftNormalizeFLIMGraphByZScore(iftFLIMGraph *graph, float *mean, float *stdev);

/* --------------------- FLIMGraph feature ----------------------------- */

double *iftFLIMgraphAggregateFeatures(int p, 
    iftIntArray *neighbors, 
    iftFLIMGraph *graph,
    int num_adj,
    iftGFLIMAggregationType aggregation_opt);

/* --------------------- FLIMGraph conversion ----------------------------- */

// image to graph
iftFLIMGraph *iftSuperpixelImageToFLIMGraph(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img, iftFLIMGraphFeature feat_type,
    iftImage *gray_img);

iftFLIMGraph* iftGetFLIMGraphFromGrid(iftImage *img,
                             iftImage **label_img);

int iftImageToFLIMGraphDISFDisk(char *image_dir,
                         char *labels_dir,
                         char *graph_dir,
                         int num_init_seeds,
                         int num_superpixels,
                         iftFLIMGraphFeature graph_feat);
int iftImageToFLIMGraphDISFDir(char *image_dir,
                          char *labels_dir,
                          char *graph_dir,
                          int num_init_seeds,
                          int num_superpixels,
                          iftFLIMGraphFeature graph_feat);

iftFLIMGraph *iftImageToFLIMGraphDISF(iftImage *img, 
                                    int init_seeds, int final_spx,
                                    iftImage **label_img,
                                    iftFLIMGraphFeature graph_feat);
                                    
int iftImageToFLIMGraphERGCDisk(char *image_dir,
                         char *labels_dir,
                         char *graph_dir,
                         int num_superpixels,
                         int m,
                         iftFLIMGraphFeature graph_feat);
int iftImageToFLIMGraphERGCDir(char *image_dir,
                          char *labels_dir,
                          char *graph_dir,
                          int num_superpixels,
                          int m,
                          iftFLIMGraphFeature graph_feat);
iftFLIMGraph *iftImageToFLIMGraphERGC(iftImage *img, 
                                    int final_spx, int m,
                                    iftImage **label_img,
                                    iftFLIMGraphFeature graph_feat);

// graph to iftImage / iftMImage
iftImage *iftFLIMGraphToImage(iftFLIMGraph *graph, iftImage *labels, int Imax, int band);

// graph from / to MIMG
iftMImage *iftFLIMGraphToMImage(iftFLIMGraph *graph, iftImage *labels);

iftFLIMGraph *iftMImageToFLIMGraph(iftMImage *mimg,
                                   iftFLIMGraph *graph_ref);

void iftMImageToFLIMGraphInplace(iftMImage *mimg, iftFLIMGraph *graph_ref);

//(write in disk)
void iftFLIMGraphToMIMG(char *graph_dir, char *labels_dir, char *output_dir);

// graph from/to iftMatrix
iftMatrix *iftFLIMGraphToFeatureMatrix(iftFLIMGraph *graph,
                                   int num_adj,
                                   iftFLIMGraphAttention attention_opt,
                                   iftGFLIMAggregationType aggregation_opt,
                                   iftIntArray **neighbors_all);

iftFLIMGraph *iftFeatureMatrixToFLIMGraph(iftMatrix *matrix,
                                   iftFLIMGraph *graph_ref);

// graph to image activations (one grayscale image per node feature)
iftImage **iftFlimGraphToActivations(iftFLIMGraph *graph, iftImage *labels);

// get neigbors
void iftGetNeighborsFLIMGraph(iftFLIMGraph *graph, int *roots, int num_roots, 
                           int k, double *dissimilarity,
                           double *feature_reference,
                           iftDHeap *heap,
                           bool most_similar);

iftIntArray *iftGetKNeighborsFLIMGraph(int node_index, iftFLIMGraph *graph, int k,
                           int dilation, bool most_similar, iftFLIMGraphAttention attention_opt);


#ifdef __cplusplus
}
#endif
#endif


