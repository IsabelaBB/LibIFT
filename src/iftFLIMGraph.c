#include "iftFLIMGraph.h"

/* --------------------- GraphArray functions ----------------------------- */
/** @defgroup GraphArrayFunc GraphArray functions
 *  Functions to create and destroy iftFLIMGraphArray.
 *  @{
 */

/**
 * @fn iftCreateEmptyFLIMGraphArray
 * @brief Create an array with n empty graphs.
 * @param n The number of graphs.
 * @return An array with n empty graphs.
 * @note The graphs are allocated but not initialized.
 */
iftFLIMGraphArray *iftCreateEmptyFLIMGraphArray(int n)
{
    iftFLIMGraphArray *array = (iftFLIMGraphArray *)calloc(1, sizeof(iftFLIMGraphArray));
    array->val = (iftFLIMGraph **)calloc(n, sizeof(iftFLIMGraph *));
    array->n = n;
    return array;
}

/**
 * @fn iftDestroyFLIMGraphArray
 * @brief Destroy an array of graphs.
 * @param arr The array of graphs to be destroyed.
 */
void iftDestroyFLIMGraphArray(iftFLIMGraphArray **arr)
{
    iftFLIMGraphArray *aux = *arr;

    if (aux != NULL)
    {
        for (int i = 0; i < aux->n; i++)
        {
            if (aux->val[i] != NULL)
                iftDestroyFLIMGraph(&aux->val[i]);
        }
        iftFree(aux->val);
        iftFree(aux);
        *arr = NULL;
    }
}

/** @} */

/* --------------------- FLIMGraph construction ----------------------------- */
/** @defgroup FLIMGraphConstr FLIMGraph construction
 *  Functions to create and destroy iftFLIMGraph.
 *  @{
 */

/**
 * @fn iftCreateEmptyFLIMGraph
 * @brief Create a new graph with `num_nodes` nodes and `num_feats` features, but no edges.
 *
 * @note The features, number of neighbors, and number of pixels of all nodes are initialized with 0.
 * @warning `num_feats` and `num_nodes` must be greater than 0
 * @note The graph has no edges. To add edges, use `iftInsertFLIMGraphNeighborNode()`
 *
 * @param num_feats Number of features
 * @param num_nodes Number of nodes
 * @return The empty graph
 *
 * @author Isabela B. Barcelos
 * @date fev, 2024
 */
iftFLIMGraph *iftCreateEmptyFLIMGraph(int num_feats, int num_nodes)
{
    if (num_feats < 0)
        iftError("Number of features in graph must be positive", "iftCreateEmptyFLIMGraph");
    if (num_nodes < 0)
        iftError("Number of nodes in graph must be positive", "iftCreateEmptyFLIMGraph");

    iftFLIMGraph *graph = (iftFLIMGraph *)calloc(1, sizeof(iftFLIMGraph));
    graph->num_nodes = num_nodes;
    graph->num_feats = num_feats;
    graph->num_edges = 0;
    graph->nodes = (iftFLIMGraphNode *)calloc(graph->num_nodes, sizeof(iftFLIMGraphNode));

#pragma omp parallel for
    for (unsigned long spx_index = 0; spx_index < graph->num_nodes; spx_index++)
    {
        graph->nodes[spx_index].numNeighbors = 0;
        // graph->nodes[spx_index].neighbors = iftCreateBMap(graph->num_nodes);
        graph->nodes[spx_index].adj = NULL;
        graph->nodes[spx_index].numPixels = 0;
    }

    for (unsigned long spx_index = 0; spx_index < graph->num_nodes; spx_index++)
        graph->nodes[spx_index].feats = (double *)calloc(num_feats, sizeof(double));

    return graph;
}

/**
 * @fn iftDestroyFLIMGraph
 * @brief Destroy an iftFLIMGraph
 */
void iftDestroyFLIMGraph(iftFLIMGraph **graph)
{
    iftFLIMGraph *aux = *graph;

    if (aux != NULL)
    {
        for (int i = 0; i < aux->num_nodes; i++)
        {
            // iftDestroyBMap(aux->nodes[i].neighbors);
            iftDestroySet(&(aux->nodes[i].adj));
            iftFree(aux->nodes[i].feats);
        }
        iftFree(aux->nodes);
        iftFree(aux);
        *graph = NULL;
    }
}

/**
 * @fn iftInsertFLIMGraphNeighborNode
 * @brief Add `neighbor` to the adjacent list of `node`
 *
 * @warning This function allows parallel edges
 * @warning This function does not check if `neighbor` exists in the graph
 * @warning This function does not increment the number of edges in the graph
 * @note For undirected edge, also call `iftInsertFLIMGraphNeighborNode()` switching `node` and `neighbor`
 *
 * @param node A pointer to a graph node
 * @param neighbor The neighbor index in the graph (> 0)
 *
 * @author Isabela B. Barcelos
 * @date fev, 2024
 */
void iftInsertFLIMGraphNeighborNode(iftFLIMGraphNode *node, int neighbor)
{
    if (node == NULL)
        iftError("Node is NULL", __func__);
    if (neighbor < 0)
        iftError("Neighbor index value is invalid %d", __func__, neighbor);

    iftInsertSet(&(node->adj), neighbor);
    (*node).numNeighbors++;
}

/**
 * @fn iftCopyFLIMGraph
 * @brief Create a copy of `graph_orig`
 *
 * Make a copy of `graph_orig`.
 * If `copy_features` is true, the features in `graph_orig` are copied. Otherwise, set them to 0.
 * When copy_features is true and:
 * - num_feats is greater than the number of features in `graph_orig`, the remaining features are set to 0.
 * - num_feats is less than the number of features in `graph_orig`, the remaining features are not copied.
 *
 * @param num_feats Number of features (> 0)
 * @param graph_orig Original graph
 * @param copy_features If true, copy the features from the original graph. Otherwise, set all features to 0
 * @return The copy of graph_orig
 *
 * @author Isabela B. Barcelos
 * @date abr, 2023
 */
iftFLIMGraph *iftCopyFLIMGraph(int num_feats, iftFLIMGraph *graph_orig, bool copy_features)
{
    iftFLIMGraph *graph;

    if (num_feats <= 0)
        iftError("Number of features must be positive", __func__);

    iftIsValidFLIMGraph(graph_orig);
    graph = iftCreateEmptyFLIMGraph(num_feats, graph_orig->num_nodes);

    int min_feats = iftMin(graph->num_feats, graph_orig->num_feats);

    iftFLIMGraphNode *node = graph->nodes;
    iftFLIMGraphNode *node_orig = graph_orig->nodes;
    for (unsigned long node_index = 0; node_index < graph->num_nodes; node_index++)
    {
        if (copy_features)
        {
            memcpy(node->feats, node_orig->feats, min_feats * sizeof(double));
            for (int i = min_feats; i < graph->num_feats; i++)
            {
                node->feats[i] = 0;
            }
        }

        node->numPixels = node_orig->numPixels;
        node->adj = iftSetCopy(node_orig->adj);
        node->numNeighbors = node_orig->numNeighbors;

        node++;
        node_orig++;
    }
    graph->num_edges = graph_orig->num_edges;
    node = NULL;
    node_orig = NULL;
    return graph;
}

/**
 * @fn iftIsValidFLIMGraph
 * @brief Check whether a graph is valid, i.e., it has at least one node.
 * @param graph Graph
 * @return True if the graph is valid, false otherwise
 */
bool iftIsValidFLIMGraph(iftFLIMGraph *graph)
{
    if (graph == NULL)
        iftError("Graph is NULL", __func__);
    if (graph->num_nodes == 0)
        iftError("Graph has no nodes", __func__);
    if (graph->nodes == NULL)
        iftError("Graph has no nodes", __func__);
    return true;
}

/** @} */

/* --------------------- FLIMGraph convert from string to a type ----------------------------- */
/** @defgroup getFLIMGraphTypes Get types
 * Functions to get types from a string.
 *  @{
 */

/**
 * @fn iftGLIMGraphGetAggregationType
 * @brief Get the graph aggregation type from a string
 * @param arg aggregation type as string
 * @return iftFLIMGraphAggregation
 */
iftFLIMGraphAttention iftGLIMGraphGetAggregationType(char *arg)
{
    if (strcmp(arg, "CONCAT") == 0)
        return IFT_GFLIM_AGGREGATION_CONCAT;
    else if (strcmp(arg, "MEAN") == 0)
        return IFT_GFLIM_AGGREGATION_MEAN;
    else if (strcmp(arg, "WMEAN") == 0)
        return IFT_GFLIM_AGGREGATION_WMEAN;
    else
        iftError("Invalid aggregation type %s", __func__, arg);
    return IFT_GFLIM_AGGREGATION_CONCAT;
}

/**
 * @fn iftGLIMGraphGetAttentionType
 * @brief Get the graph attention from a string
 * @param arg attention type as string
 * @return iftFLIMGraphAttention
 */
iftFLIMGraphAttention iftGLIMGraphGetAttentionType(char *arg)
{
    if (strcmp(arg, "NONE") == 0)
        return IFT_FLIMGRAPH_ATTENTION_NONE;
    else if (strcmp(arg, "NODE_RELSIZE") == 0)
        return IFT_FLIMGRAPH_ATTENTION_NODE_RELSIZE;
    else if (strcmp(arg, "NODE_SIZE") == 0)
        return IFT_FLIMGRAPH_ATTENTION_NODE_SIZE;
    /*else if (strcmp(arg, "NEIGHB_SIZE") == 0)
        return IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIZE;*/
    /*else if (strcmp(arg, "NEIGHB_SIMIL") == 0)
        return IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL;*/
    else if (strcmp(arg, "CLASS_DOMINANCE") == 0)
        return IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE;
    else if (strcmp(arg, "CLASS_DOMINANCE_TEST") == 0)
        return IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_TEST;
    else if (strcmp(arg, "CLASS_DOMINANCE_POWER2") == 0)
        return IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_POWER2;
    else if (strcmp(arg, "CLASS_DOMINANCE_SOFTMAX_NEIGHB") == 0)
        return IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_SOFTMAX_NEIGHB;
    else if (strcmp(arg, "COSINE_SIMILARITY") == 0)
        return IFT_FLIMGRAPH_ATTENTION_COSINE_SIMILARITY;
    else
        iftError("Invalid attention type %s", __func__, arg);
    return IFT_FLIMGRAPH_ATTENTION_NONE;
}

/**
 * @fn iftGFLIMGetGraphFeature
 * @brief Get the graph feature from a string
 * @param name Name of the graph feature
 * @return iftFLIMGraphFeature Graph feature
 */
iftFLIMGraphFeature iftGFLIMGetGraphFeature(const char *name)
{
    if (strcmp(name, "MEAN") == 0)
        return IFT_FLIMGRAPH_FEATURE_MEAN;
    if (strcmp(name, "SUM") == 0)
        return IFT_FLIMGRAPH_FEATURE_SUM;
    if (strcmp(name, "SUM_MULT_RELSIZE") == 0)
        return IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE;
    if (strcmp(name, "MEAN_MULT_RELSIZE") == 0)
        return IFT_FLIMGRAPH_FEATURE_MEAN_MULT_RELSIZE;
    if (strcmp(name, "MEAN_CONCAT_RELSIZE") == 0)
        return IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE;
    if (strcmp(name, "MEAN_CONCAT_RELSIZE_NEIGHB") == 0)
        return IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB;
    if (strcmp(name, "MEAN_CONCAT_RELSIZE_HARALICK") == 0)
        return IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK;
    if (strcmp(name, "SUM_MULT_RELSIZE_CONCAT_HARALICK") == 0)
        return IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK;
    if (strcmp(name, "SUM_CONCAT_HARALICK") == 0)
        return IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK;
    iftError("Invalid iftFLIMGraphFeature name %s", __func__, name);
    return IFT_FLIMGRAPH_FEATURE_MEAN;
}

/**
 * @fn iftGFLIMGetGraphType
 * @brief Get the graph type from a string
 * @param name Name of the graph type
 * @return iftFLIMGraphType Graph type
 */
iftFLIMGraphType iftGFLIMGetGraphType(const char *name)
{
    if (strcmp(name, "DISF") == 0)
        return IFT_FLIMGRAPH_TYPE_DISF;
    if (strcmp(name, "ERGC") == 0)
        return IFT_FLIMGRAPH_TYPE_ERGC;
    iftError("Invalid iftFLIMGraphType name %s", __func__, name);
    return IFT_FLIMGRAPH_TYPE_DISF;
}

/** @} */

/* --------------------- FLIMGraph read and write ----------------------------- */
/** @defgroup FLIMGraphReadWrite Read and write FLIMGraph
 *  @{
 */

/**
 * @fn iftReadFLIMGraph
 * @brief Read a graph from a json file
 *
 * @param filename File name
 * @return The graph
 *
 * @author Isabela B. Barcelos
 * @date abr, 2023
 */
iftFLIMGraph *iftReadFLIMGraph(char *filename)
{

    if (!iftFileExists(filename))
        iftError("Graph file does not exist", __func__);
    iftDict *graph_dict = iftReadJson(filename);

    unsigned long num_nodes = (unsigned long)iftGetLongValFromDict("num_nodes", graph_dict); 
    unsigned long num_feats = (unsigned long)iftGetLongValFromDict("num_feats", graph_dict); 

    bool directed = false;
    if (iftDictContainKey("directed", graph_dict, NULL))
        directed = (bool)iftGetLongValFromDict("directed", graph_dict);

    if (num_nodes < 0)
        iftError("Number of nodes must be greater than 0", __func__);
    if (num_feats < 0)
        iftError("Number of features must be greater than 0", __func__);

    iftFLIMGraph *graph = iftCreateEmptyFLIMGraph(num_feats, num_nodes);
    graph->num_edges = (unsigned long)iftGetLongValFromDict("num_edges", graph_dict); 

    char name[50];
    iftFLIMGraphNode *node = graph->nodes;
    for (unsigned long l = 0; l < graph->num_nodes; l++)
    {
        sprintf(name, "node%lu", l);
        iftDict *node_info = iftGetDictFromDict(name, graph_dict);

        node->numPixels = (unsigned long)iftGetLongValFromDict("num_pixels", node_info);

        iftDblArray *feats = iftGetDblArrayFromDict("feats", node_info);
        memcpy(node->feats, feats->val, graph->num_feats * sizeof(double));
        iftDestroyDblArray(&feats);

        if (iftDictContainKey("neighbors", node_info, NULL))
        {
            iftIntArray *input = iftGetIntArrayFromDict("neighbors", node_info);
            for (int i = 0; i < input->n; i++)
            {
                iftInsertFLIMGraphNeighborNode(node, input->val[i]);
                if (!directed)
                    iftInsertFLIMGraphNeighborNode(&(graph->nodes[input->val[i]]), l);
            }
            iftDestroyIntArray(&input);
        }

        iftDestroyDict(&node_info);
        node++;
    }
    iftDestroyDict(&graph_dict);
    node = NULL;

    return (graph);
}

/**
 * @fn iftWriteFLIMGraph
 * @brief Write a graph to a json file
 * @param graph Graph
 * @param filename File path
 */
void iftWriteFLIMGraph(iftFLIMGraph *graph, char *filename)
{
    iftIsValidFLIMGraph(graph);

    // create json data
    iftDict *info = iftCreateDict();

    iftInsertIntoDict("num_nodes", (long)graph->num_nodes, info);
    iftInsertIntoDict("num_feats", (long)graph->num_feats, info);
    iftInsertIntoDict("num_edges", (long)graph->num_edges, info);
    iftInsertIntoDict("directed", false, info);

    // save nodes information
    iftDict *node_info[graph->num_nodes];
    iftDblArray *feats[graph->num_nodes];
    iftIntArray *neighbors[graph->num_nodes];

    iftFLIMGraphNode *node = graph->nodes;
    for (unsigned long i = 0; i < graph->num_nodes; i++)
    {
        char node_name[100];
        sprintf(node_name, "node%lu", i);
        node_info[i] = iftCreateDict();
        iftInsertIntoDict("num_pixels", (long)node->numPixels, node_info[i]);
        feats[i] = iftCopyDblArray(node->feats, graph->num_feats);
        iftInsertIntoDict("feats", feats[i], node_info[i]);

        long j = 0;
        iftSet *neigh = node->adj;
        while (neigh != NULL)
        {
            if (neigh->elem < i)
                j++;
            neigh = neigh->next;
        }
        neighbors[i] = NULL;
        if (j > 0)
        {
            neighbors[i] = iftCreateIntArray(j);
            neigh = node->adj;
            j = 0;
            while (neigh != NULL)
            {
                if (neigh->elem < i)
                {
                    neighbors[i]->val[j] = neigh->elem;
                    j++;
                }
                neigh = neigh->next;
            }
            iftInsertIntoDict("neighbors", neighbors[i], node_info[i]);
        }
        iftInsertIntoDict(node_name, node_info[i], info);
        node++;
    }
    node = NULL;
    char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");
    iftWriteJson(info, filename);
    setlocale(LC_ALL, old_locale);
	free(old_locale);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        iftDestroyDict(&node_info[i]);
        iftDestroyDblArray(&feats[i]);
        if (neighbors[i] != NULL)
            iftDestroyIntArray(&neighbors[i]);
    }
    iftDestroyDict(&info);
}

/** @} */

/* --------------------- Functions for Haralick features ----------------------------- */
/** @defgroup FLIMGraphHaralickFeatures Functions for Haralick features
 *  @{
 */

/**
 * @fn ComputeNormalizedGLCM
 * @brief Compute the normalized GLCM for a given superpixel in an image.
 *
 * @param original_img The original image
 * @param label_img The superpixel image
 * @param spx_label The superpixel label
 * @param spx_size The size of the superpixel
 * @param angles The angles for the GLCM
 * @param nangles The number of angles
 * @param num_levels The number of gray levels
 * @return The normalized GLCM for the superpixel
 */
iftMatrix **ComputeNormalizedGLCM(iftImage *original_img, iftImage *label_img,
                                  int spx_label, int spx_size,
                                  double *angles, int nangles, int num_levels)
{
    iftMatrix **glcm = (iftMatrix **)malloc(nangles * sizeof(iftMatrix *));
    for (int a = 0; a < nangles; a++)
        glcm[a] = iftCreateMatrix(num_levels, num_levels);

    int rows = original_img->xsize;
    int cols = original_img->ysize;
    for (int p = 0; p < original_img->n; p++)
    {
        if (label_img->val[p] == spx_label + 1)
        {
            for (int a = 0; a < nangles; a++)
            {
                double angle = angles[a];
                int offset_row = round(sin(angle));
                int offset_col = round(cos(angle));
                int i = original_img->val[p];

                int j_index = p + offset_row * cols + offset_col;
                if (j_index >= rows * cols)
                    continue;
                int j = original_img->val[j_index];

                if (i >= 0 && j >= 0 && i < num_levels && j < num_levels)
                    iftMatrixElem(glcm[a], i, j) += 1;
            }
        }
    }

    for (int a = 0; a < nangles; a++)
    {
        // make it symmetric
        iftMatrix *T = iftTransposeMatrix(glcm[a]);
        iftSumMatricesInPlace(glcm[a], T, glcm[a]);
        iftDestroyMatrix(&T);

        // normalize
        double norm = iftMatrixSum(glcm[a]);
        if (norm != 0)
            iftComputeDivisionBetweenMatrixScalar(glcm[a], norm, 'f');
    }

    return glcm;
}

/**
 * @fn getContrastHaralick
 * @brief Compute contrast from a GLCM.
 * Let P(i,j) the value in the i-th row and j-th column of a GLCM, 0 < i,j <= N,
 * contrast = sum_{i=1,j=1}^{N} P(i,j) * (i-j)^2
 * @param glcm The GLCM
 * @return The contrast
 */
float getContrastHaralick(iftMatrix *glcm)
{
    float contrast = 0.0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
            contrast += iftMatrixElem(glcm, j, i) * (i - j) * (i - j);
    }
    return contrast;
}

/**
 * @fn getCorrelationHaralick
 * @brief Compute correlation from a GLCM.
 * Let:
 * - P(i,j) the value in the i-th row and j-th column of a GLCM, 0 < i,j <= N,
 * - Px(i) = sum_{j=1}^{N} P(i,j) , mx(i) = sum_{i=1}^{N} i * Px(i) ,
 * - Py(j) = sum_{j=1}^{N} P(i,j) , my(j) = sum_{j=1}^{N} j * Py(j) ,
 * - stdx(i) = sqrt(sum_{i=1}^{N} (i - mx(i))^2 * Px(i)) ,
 * - stdy(j) = sqrt(sum_{j=1}^{N} (j - my(j))^2 * Py(j)) ,
 * Then, correlation = (1 / (stdx(i) * stdy(j))) * sum_{i=1,j=1}^{N} (i - mx(i)) * (j - my(j)) * P(i,j)
 * @param glcm The GLCM
 * @return The correlation
 */
float getCorrelationHaralick(iftMatrix *glcm)
{
    // Compute Pi e Pj (sum of the rows and of the cols, respectively)
    // and the average values of Pi e Pj
    float *Pi = iftAllocFloatArray(glcm->nrows); // Pi = sum_{j=1}^{N} P(i,j)
    float *Pj = iftAllocFloatArray(glcm->ncols); // Pj = sum_{i=1}^{N} P(i,j)

    float mean_i = 0, mean_j = 0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
        {
            Pi[i] += iftMatrixElem(glcm, j, i);
            Pj[i] += iftMatrixElem(glcm, i, j);
        }

        mean_i += i * Pi[i];
        mean_j += i * Pj[i];
    }

    // Compute the standar deviation of Pi and Pj
    // and pre-compute (i - mean_i) e (j - mean_j)
    float std_i = 0, std_j = 0;
    float i_diff[glcm->nrows], j_diff[glcm->ncols];
    for (int i = 0; i < glcm->nrows; i++)
    {
        std_i += pow(i - mean_i, 2) * Pi[i];
        std_j += pow(i - mean_j, 2) * Pj[i];

        i_diff[i] = i - mean_i;
        j_diff[i] = i - mean_j;
    }

    if (std_i == 0)
        std_i = 1;
    if (std_j == 0)
        std_j = 1;

    std_i = sqrt(std_i);
    std_j = sqrt(std_j);

    // Compute the correlation
    float correlation = 0.0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
        {
            correlation += i_diff[i] * j_diff[j] * iftMatrixElem(glcm, j, i);
        }
    }
    correlation /= (std_i * std_j);

    iftFree(Pi);
    iftFree(Pj);

    return correlation;
}

/**
 * @fn getEnergyHaralick
 * @brief Compute energy (angular second moment) from a GLCM.
 * Let P(i,j) the value in the i-th row and j-th column of a GLCM, 0 < i,j <= N,
 * energy = sum_{i=1,j=1}^{N} P(i,j)^2
 * @param glcm The GLCM
 * @return The energy
 */
float getEnergyHaralick(iftMatrix *glcm)
{
    float energy = 0.0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
            energy += pow(iftMatrixElem(glcm, j, i), 2);
    }
    return energy;
}

/**
 * @fn getHomogeneityHaralick
 * @brief Compute homogeneity from a GLCM.
 * Let P(i,j) the value in the i-th row and j-th column of a GLCM, 0 < i,j <= N,
 * homogeneity = sum_{i=1,j=1}^{N} P(i,j) / (1 + (i-j)^2)
 * @param glcm The GLCM
 * @return The homogeneity
 */
float getHomogeneityHaralick(iftMatrix *glcm)
{
    float homogeneity = 0.0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
            homogeneity += iftMatrixElem(glcm, j, i) / (1 + pow(i - j, 2));
    }
    return homogeneity;
}

/**
 * @fn getEntropyHaralick
 * @brief Compute entropy from a GLCM.
 * Let P(i,j) the value in the i-th row and j-th column of a GLCM, 0 < i,j <= N,
 * entropy = - sum_{i=1,j=1}^{N} P(i,j) * log_2(P(i,j))
 * @param glcm The GLCM
 * @return The entropy
 */
float getEntropyHaralick(iftMatrix *glcm)
{
    float entropy = 0.0;
    for (int i = 0; i < glcm->nrows; i++)
    {
        for (int j = 0; j < glcm->ncols; j++)
            if (iftMatrixElem(glcm, j, i) > 0)
                entropy += iftMatrixElem(glcm, j, i) * log(iftMatrixElem(glcm, j, i));
    }
    return -entropy;
}

/**
 * @fn iftFLIMGraphHaralickFeatures
 * @brief Compute Haralick features for a superpixel image.
 * @param graph The FLIMGraph
 * @param original_img The original image
 * @param label_img The superpixel image
 * @param feat_start The starting index for the features in the graph
 * @note The features are stored in the graph nodes.
 * @note The features are computed for each superpixel in the image.
 * @note It averages the features for the angles 0, 45, 90, and 135 degrees.
 * @note The features are contrast, correlation, energy, homogeneity, and entropy.
 */
void iftFLIMGraphHaralickFeatures(
    iftFLIMGraph *graph,
    iftImage *original_img,
    iftImage *label_img,
    int feat_start)
{
    if (original_img == NULL)
        iftError("Original image is NULL", __func__);
    if (label_img == NULL)
        iftError("Labeled image is NULL", __func__);
    if (graph == NULL)
        iftError("Graph is NULL", __func__);

    int num_levels = iftMaximumValue(original_img) + 1;
    double angles[] = {0, IFT_PI / 4, IFT_PI / 2, 3 * IFT_PI / 4}; // 0, 45, 90, 135
    int nangles = 4;

    iftFLIMGraphNode *node = graph->nodes;
    for (int i = 0; i < graph->num_nodes; i++)
    {
        // https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
        // https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/texture.py#L15-L167
        // https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_texture.pyx
        // https://medium.com/@girishajmera/feature-extraction-of-images-using-glcm-gray-level-cooccurrence-matrix-e4bda8729498

        // 1. Create k GLCM matrices and a superpixel graph
        iftMatrix **glcm = ComputeNormalizedGLCM(original_img, label_img, i, node->numPixels,
                                                 angles, nangles, num_levels);

        // 2. Compute Haralick features for each GLCM matrix
        for (int p = 0; p < nangles; p++)
        {
            node->feats[feat_start] += getContrastHaralick(glcm[p]);
            node->feats[feat_start + 1] += getCorrelationHaralick(glcm[p]);
            node->feats[feat_start + 2] += getEnergyHaralick(glcm[p]);
            node->feats[feat_start + 3] += getHomogeneityHaralick(glcm[p]);
            node->feats[feat_start + 4] += getEntropyHaralick(glcm[p]);
            iftDestroyMatrix(&glcm[p]);
        }

        for (int p = 0; p < 5; p++)
            node->feats[feat_start + p] /= nangles;

        free(glcm);
        node++;
    }
}

/** @} */

/* --------------------- FLIMGraph segmentation ----------------------------- */
/** @defgroup FLIMGraphSegmentation FLIMGraph segmentation
 *  @{
 */

/**
 * @fn iftERGC
 * @brief Create a segmentation map using ERGC algorithm.
 *
 * @param mimg Multiband image
 * @param A Adjacency relation
 * @param num_superpixels Number of superpixels
 * @param m Compacity parameter.
 * @return The segmentation map.
 */
iftImage *iftERGC(iftMImage *mimg, iftAdjRel *A, int *num_superpixels, int m)
{
    iftImage *seeds_img;
    double *distances;
    int *superpixel_area;
    float **superpixel_feats;
    iftDHeap *heap;

    // GRID SAMPLING ==============================================================
    iftAdjRel *A_grad = iftCircular(sqrtf(2.0));
    iftImage *grad = iftMImageGradient(mimg, A_grad, iftMMaximumValue(mimg, -1));
    int s = (float)(mimg->xsize * mimg->ysize) / (float)(*num_superpixels);
    int dy = 0.5f + sqrt(s * mimg->ysize / (float)mimg->xsize);
    int dx = 0.5f + s / (float)(0.5f + sqrt(s * mimg->ysize / (float)mimg->xsize));
    int num_seeds = 1;
    seeds_img = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    for (int x = dx / 2; x < mimg->xsize; x += dx)
    {
        for (int y = dy / 2; y < mimg->ysize; y += dy)
        {
            iftVoxel u;
            u.x = x;
            u.y = y;
            u.z = 0;
            u.t = 0;
            for (int j = 1; j < A_grad->n; j++)
            {
                iftVoxel v = iftGetAdjacentVoxel(A_grad, u, j);
                if (iftMValidVoxel(mimg, v))
                {
                    int v_index = iftMGetVoxelIndex(mimg, v);
                    if (grad->val[v_index] < grad->val[iftMGetVoxelIndex(mimg, u)] && seeds_img->val[v_index] == 0)
                    {
                        u.x = v.x;
                        u.y = v.y;
                    }
                }
            }
            seeds_img->val[iftMGetVoxelIndex(mimg, u)] = num_seeds;
            num_seeds++;
        }
    }
    iftDestroyImage(&grad);
    iftDestroyAdjRel(&A_grad);
    num_seeds--;
    *num_superpixels = num_seeds;

    // Init --------------------------------------------------------------------
    distances = (double *)calloc(mimg->n, sizeof(double));
    heap = iftCreateDHeap(mimg->n, distances);
    superpixel_area = (int *)calloc(num_seeds, sizeof(int));
    superpixel_feats = (float **)calloc(num_seeds, sizeof(float *));
    float Sz = (float)mimg->n / (float)num_seeds;

    int *superpixel_root = (int *)calloc(num_seeds, sizeof(int));
    // initialize_images
#pragma omp parallel for
    for (int p_index = 0; p_index < mimg->n; p_index++)
    {
        if (seeds_img->val[p_index] == 0)
        {
            distances[p_index] = INFINITY;
        }
        else
        {
            distances[p_index] = 0;
            superpixel_root[seeds_img->val[p_index] - 1] = p_index;
        }
    }

    // initialize_superpixels
    for (int seed_index = 0; seed_index < num_seeds; seed_index++)
    {
        int p_index = superpixel_root[seed_index];
        superpixel_area[seed_index] = 1;
        superpixel_feats[seed_index] = (float *)calloc(mimg->m, sizeof(float));
        for (int j = 0; j < mimg->m; ++j)
            superpixel_feats[seed_index][j] = mimg->val[p_index][j];

        iftVoxel p_voxel = iftMGetVoxelCoord(mimg, p_index);
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
            if (iftMValidVoxel(mimg, q_voxel))
            {
                int q_index = iftMGetVoxelIndex(mimg, q_voxel);
                if (seeds_img->val[q_index] != 0)
                    continue;

                superpixel_area[seed_index]++;
                for (int j = 0; j < mimg->m; ++j)
                    superpixel_feats[seed_index][j] += mimg->val[q_index][j];
            }
        }
        iftInsertDHeap(heap, p_index);
    }

    while (!iftEmptyDHeap(heap))
    {
        int p_index, p_label;
        iftVoxel p_voxel;

        p_index = iftRemoveDHeap(heap);

        p_voxel = iftMGetVoxelCoord(mimg, p_index);
        p_label = seeds_img->val[p_index];

        // Add spel p to its tree
        superpixel_area[p_label - 1]++;
        for (int j = 0; j < mimg->m; j++)
            superpixel_feats[p_label - 1][j] += mimg->val[p_index][j];

        // For each adjacent
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
            if (iftMValidVoxel(mimg, q_voxel))
            {
                int q_index = iftMGetVoxelIndex(mimg, q_voxel);

                if (heap->color[q_index] == IFT_BLACK)
                    continue;

                double P = 0;
                for (int j = 0; j < mimg->m; j++)
                {
                    double tmp = mimg->val[q_index][j] - (superpixel_feats[p_label - 1][j] / (float)superpixel_area[p_label - 1]);
                    P += tmp * tmp;
                }
                if (m > 0)
                {
                    iftVoxel root = iftMGetVoxelCoord(mimg, superpixel_root[p_label - 1]);
                    float dxy = (root.x - q_voxel.x) * (root.x - q_voxel.x) + (root.y - q_voxel.y) * (root.y - q_voxel.y);
                    P = sqrt(P * P + dxy * dxy * m * m / (Sz * Sz));
                }

                double a1 = INFINITY;
                if (q_voxel.x < mimg->xsize - 1)
                    a1 = distances[q_index + 1];
                if (q_voxel.x > 0)
                    a1 = a1 < distances[q_index - 1] ? a1 : distances[q_index - 1];

                double a2 = INFINITY;
                if (q_voxel.y < mimg->ysize - 1)
                    a2 = distances[q_index + mimg->xsize];
                if (q_voxel.y > 0)
                    a2 = a2 < distances[q_index - mimg->xsize] ? a2 : distances[q_index - mimg->xsize];

                if (a1 > a2)
                {
                    double tmp = a1;
                    a1 = a2;
                    a2 = tmp;
                }

                float A1 = 0;
                if (P * P > (a2 - a1) * (a2 - a1))
                {
                    float delta = 2 * P * P - (a2 - a1) * (a2 - a1);
                    A1 = (a1 + a2 + sqrt(delta)) / 2.0;
                }
                else
                    A1 = a1 + P;

                if (heap->color[q_index] == IFT_GRAY)
                {
                    if (A1 < distances[q_index])
                    {
                        // update distance
                        distances[q_index] = A1;
                        seeds_img->val[q_index] = p_label;
                        iftGoUpDHeap(heap, heap->pos[q_index]);
                    }
                }
                else
                {
                    if (heap->color[q_index] == IFT_WHITE)
                    {
                        // add new point2d
                        distances[q_index] = A1;
                        seeds_img->val[q_index] = p_label;
                        iftInsertDHeap(heap, q_index);
                    }
                }
            }
        }
    }

    // Clear -------------------------------------------------------------------
    for (int i = 0; i < num_seeds; i++)
        free(superpixel_feats[i]);
    iftFree(superpixel_root);
    free(superpixel_feats);
    free(superpixel_area);
    iftDestroyDHeap(&heap);
    free(distances);

    return seeds_img;
}

/**
 * @fn iftGetFLIMGraphFromGrid
 * @brief Create a grid graph from an image
 * @param img input image
 * @param label_img a pointer to a superpixel segmentation
 * @return The FLIMGraph
 * @note `label_img` is allocated inside the function and must be destroyed by the caller.
 */
iftFLIMGraph *iftGetFLIMGraphFromGrid(iftImage *img, iftImage **label_img)
{
    iftMImage *mimg = NULL;
    iftAdjRel *A = NULL;
    iftFLIMGraph *graph = NULL;

    if (img == NULL)
        iftError("Image is null.", __func__);

    // Init other inputs
    if (iftIsColorImage(img))
        mimg = iftImageToMImage(img, LABNorm_CSPACE);
    else
        mimg = iftImageToMImage(img, GRAYNorm_CSPACE);

    if (iftIs3DMImage(mimg))
        A = iftSpheric(sqrtf(3.0));
    else
        A = iftCircular(sqrtf(2.0));

    (*label_img) = iftCreateImage(mimg->xsize, mimg->ysize, 1);
    graph = iftCreateEmptyFLIMGraph(mimg->m, mimg->n);
    graph->num_edges = (mimg->xsize - 2) * (mimg->ysize - 2) * 4 + (2 * (mimg->xsize - 2) + 2 * (mimg->ysize - 2)) * 3 + 8;

#pragma omp parallel for
    for (int i = 0; i < mimg->n; i++)
    {
        iftVoxel p_voxel;
        iftFLIMGraphNode *node;

        p_voxel = iftMGetVoxelCoord(mimg, i);
        (*label_img)->val[i] = i + 1;
        node = &(graph->nodes[i]);
        node->numPixels = 1;

        for (int j = 0; j < graph->num_feats; j++)
            node->feats[j] = mimg->val[i][j];

        for (int j = 0; j < A->n; j++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, j);
            if (iftMValidVoxel(mimg, q_voxel))
            {
                iftInsertFLIMGraphNeighborNode(node, iftMGetVoxelIndex(mimg, q_voxel));
            }
        }
        node = NULL;
    }
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    return graph;
}

/**
 * @fn iftImageToGridFLIMGraphs
 * @brief Create grid graphs from images in a folder.
 *
 * Create grid graphs from images in a folder.
 * The superpixel maps and graphs are saved in the respective directories.
 *
 * @note The graph and superpixel directory are created if they do not exist
 *
 * @param image_dir image directory
 * @param labels_dir superpixel map directory
 * @param graph_dir graphs directory
 */
void iftImageToGridFLIMGraphs(char *image_dir,
                              char *labels_dir,
                              char *graph_dir)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t image_dir: %s\n", image_dir == NULL ? "NULL" : image_dir);
    printf("\t labels_dir: %s\n", labels_dir == NULL ? "NULL" : labels_dir);
    printf("\t graph_dir: %s\n", graph_dir == NULL ? "NULL" : graph_dir);
#endif

    if (!iftDirExists(image_dir))
        iftError("Image directory does not exist.", __func__);
    iftFileSet *fs_imgs = iftLoadFileSetFromDir(image_dir, 1);
    if (fs_imgs->n < 1)
        iftError("No images found.", __func__);

    iftMakeDir(labels_dir);
    iftMakeDir(graph_dir);

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[200], ext[10];

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);

        iftImage *img = iftReadImageByExt(fs_imgs->files[i]->path);
        iftImage *label_img;
        iftFLIMGraph *graph = iftGetFLIMGraphFromGrid(img, &label_img);
        iftDestroyImage(&img);

        sprintf(filename, "%s/%s.pgm", labels_dir, basename);
        iftWriteImageByExt(label_img, filename);
        iftDestroyImage(&label_img);

        sprintf(filename, "%s/%s.json", graph_dir, basename);
        iftWriteFLIMGraph(graph, filename);
        iftDestroyFLIMGraph(&graph);
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);
}

/* --------------------- FLIMGraph transform ----------------------------- */
/** @defgroup FLIMGraphTransform FLIMGraph transform
 * Functions to transform the FLIMGraph to and from other data structures
 *  @{
 */

/**
 * @fn iftNormalizeFLIMGraphByZScore
 * @brief Normalize the node features based on mean and standard deviation values
 *
 * @warning `mean` and `stdev` must have the same number of values as the number of features in the graph
 *
 * @param graph Graph
 * @param mean Mean values
 * @param stdev Standard deviation values
 */
void iftNormalizeFLIMGraphByZScore(iftFLIMGraph *graph, float *mean, float *stdev)
{
#pragma omp parallel for
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
            graph->nodes[p].feats[b] = stdev[b] != 0 ? ((float)graph->nodes[p].feats[b] - mean[b]) / stdev[b] : ((float)graph->nodes[p].feats[b] - mean[b]);
    }
}

/**
 * @fn iftSuperpixelImageToFLIMGraph
 * @brief Create a graph from the original image and its segmentation map.
 *
 * @param num_superpixels   The number of superpixels in `labeled_img`.
 * @param A                 An adjacency relation between pixels.
 * @param oiginal_img       The original image.
 * @param labeled_img       A map of superpixel labels.
 * @param feat_type         The type of feature to be computed.
 * @param gray_img         The gray image. It is used to compute the Haralick features.
 * @return The FLIMGraph.
 *
 * @note The values in the segmentation map must be in the range [1, num_superpixels]
 * @author Isabela Borlido.
 */
iftFLIMGraph *iftSuperpixelImageToFLIMGraph(
    unsigned long num_superpixels,
    iftAdjRel *A,
    iftMImage *oiginal_img, iftImage *labeled_img,
    iftFLIMGraphFeature feat_type,
    iftImage *gray_img)
{
    if (num_superpixels < 1)
        iftError("Number of superpixels must be positive", __func__);
    if ((feat_type == IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK ||
         feat_type == IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK ||
         feat_type == IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK) &&
        gray_img == NULL)
        iftError("Gray image is NULL", __func__);

    iftFLIMGraph *graph;
    iftBMap *neighbors[num_superpixels];

    int feat_num = 0;
    switch (feat_type)
    {
    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE:
    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB:
        feat_num = oiginal_img->m + 1;
        break;

    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK:
        feat_num = oiginal_img->m + 6;
        break;

    case IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK:
    case IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK:
        feat_num = oiginal_img->m + 5;
        break;

    default:
        feat_num = oiginal_img->m;
        break;
    }

    graph = iftCreateEmptyFLIMGraph(feat_num, num_superpixels);

    for (int i = 0; i < num_superpixels; i++)
        neighbors[i] = iftCreateBMap(num_superpixels);

    // for each pixel
    for (int p_index = 0; p_index < labeled_img->n; p_index++)
    {
        unsigned long label = (unsigned long)(labeled_img->val[p_index] - 1); // get its label
        iftFLIMGraphNode *node = &(graph->nodes[label]);                      // get the node in the graph
        node->numPixels++;

        // increase the features (all feature types involve the mean or sum of the original image color values)
        for (unsigned long band_index = 0; band_index < oiginal_img->m; band_index++)
            node->feats[band_index] += oiginal_img->val[p_index][band_index];

        // look at neighbor pixels
        iftVoxel p_voxel = iftMGetVoxelCoord(oiginal_img, p_index);
        for (int i = 1; i < A->n; i++)
        {
            iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
            if (iftMValidVoxel(oiginal_img, q_voxel))
            {
                // get the label of the neighbor pixel
                unsigned long q_label = (unsigned long)(labeled_img->val[iftMGetVoxelIndex(oiginal_img, q_voxel)] - 1);

                // check if neighbor relationship (the edge) is already in the graph
                // since the graph is undirected, we check only one side and add both
                if (label > q_label && !iftBMapValue(neighbors[label], q_label))
                {
                    iftInsertFLIMGraphNeighborNode(node, q_label);
                    iftInsertFLIMGraphNeighborNode(&(graph->nodes[q_label]), label);
                    iftBMapSet1(neighbors[label], q_label);
                    graph->num_edges++;
                }
            }
        }
    }

    iftFLIMGraphNode *node = graph->nodes;
    switch (feat_type)
    {
    case IFT_FLIMGRAPH_FEATURE_MEAN:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            for (unsigned long band_index = 0; band_index < graph->num_feats; band_index++)
                node->feats[band_index] = (node->feats[band_index] / (double)node->numPixels);
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_SUM:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            double rel_size = (double)node->numPixels / (double)oiginal_img->n;
            for (unsigned long band_index = 0; band_index < graph->num_feats; band_index++)
                node->feats[band_index] *= (1.0 + rel_size);
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_MEAN_MULT_RELSIZE:
        /*int max_size = node->numPixels;
        node++;
        for (unsigned long spx_index = 1; spx_index < num_superpixels; spx_index++)
        {
            max_size = iftMax(max_size, node->numPixels);
            node++;
        }*/
        node = graph->nodes;
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            double rel_size = (double)node->numPixels / (double)oiginal_img->n;
            for (unsigned long band_index = 0; band_index < graph->num_feats; band_index++)
            {
                node->feats[band_index] /= (double)node->numPixels;
                node->feats[band_index] *= (1.0 + rel_size);
            }
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            for (unsigned long band_index = 0; band_index < graph->num_feats - 1; band_index++)
                node->feats[band_index] = (node->feats[band_index] / (double)node->numPixels);
            node->feats[graph->num_feats - 1] = (double)node->numPixels / (double)oiginal_img->n;
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_NEIGHB:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            for (unsigned long band_index = 0; band_index < graph->num_feats - 1; band_index++)
                node->feats[band_index] = (node->feats[band_index] / (double)node->numPixels);

            int sum_size = 0;
            iftSet *neigh = node->adj;
            while (neigh != NULL)
            {
                sum_size += graph->nodes[neigh->elem].numPixels;
                neigh = neigh->next;
            }
            sum_size += node->numPixels;
            node->feats[graph->num_feats - 1] = (double)node->numPixels / (double)sum_size;
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        break;

    case IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            for (unsigned long band_index = 0; band_index < oiginal_img->m; band_index++)
                node->feats[band_index] = (node->feats[band_index] / (double)node->numPixels);
            node->feats[oiginal_img->m] = (double)node->numPixels / (double)oiginal_img->n;
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        iftFLIMGraphHaralickFeatures(graph, gray_img, labeled_img, oiginal_img->m + 1);
        break;

    case IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            double rel_size = (double)node->numPixels / (double)oiginal_img->n;
            for (unsigned long band_index = 0; band_index < oiginal_img->m; band_index++)
                node->feats[band_index] *= (1.0 + rel_size);
            node++;
            iftDestroyBMap(&neighbors[spx_index]);
        }
        iftFLIMGraphHaralickFeatures(graph, gray_img, labeled_img, oiginal_img->m);
        break;
    case IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK:
        for (unsigned long spx_index = 0; spx_index < num_superpixels; spx_index++)
        {
            iftDestroyBMap(&neighbors[spx_index]);
        }
        iftFLIMGraphHaralickFeatures(graph, gray_img, labeled_img, oiginal_img->m);
        break;
    default:
        iftError("Invalid feature type %d ", __func__, feat_type);
        break;
    }
    node = NULL;

    return graph;
}

/**
 * @fn iftImageToFLIMGraph
 * @brief Create a graph from a image based on a superpixel segmentation.
 *
 * @note label_img is allocated inside the function and must be destroyed by the caller.
 *
 * @param img Input image
 * @param num_superpixels Number of superpixels
 * @param param superpixel segmentation parameter (initial seeds for DISF and compactness for ERGC)
 * @param label_img A pointer to a superpixel segmentation
 * @param graph_feat Feature type to be used in the graph
 */
iftFLIMGraph *iftImageToFLIMGraph(iftImage *img,
                                  int num_superpixels, int param,
                                  iftImage **label_img,
                                  iftFLIMGraphFeature graph_feat,
                                  iftFLIMGraphType graph_type)
{
    iftImage *label_img_tmp = NULL;
    iftMImage *mimg;
    iftAdjRel *A = NULL;
    iftFLIMGraph *graph;

    if (iftIsColorImage(img))
        mimg = iftImageToMImage(img, LABNorm_CSPACE);
    else
        mimg = iftImageToMImage(img, GRAYNorm_CSPACE);

    switch (graph_type)
    {
    case IFT_FLIMGRAPH_TYPE_DISF:
        A = iftCircular(sqrtf(2.0));
        label_img_tmp = iftDISF(mimg, A, param, num_superpixels, NULL);
        break;

    case IFT_FLIMGRAPH_TYPE_ERGC:
        A = iftCircular(1.0);
        label_img_tmp = iftERGC(mimg, A, &num_superpixels, param);
        break;

    default:
        iftError("Invalid graph type %d", __func__, graph_type);
    }

    iftImage *gray_img = NULL;
    if (graph_feat == IFT_FLIMGRAPH_FEATURE_SUM_CONCAT_HARALICK ||
        graph_feat == IFT_FLIMGRAPH_FEATURE_MEAN_CONCAT_RELSIZE_HARALICK ||
        graph_feat == IFT_FLIMGRAPH_FEATURE_SUM_MULT_RELSIZE_CONCAT_HARALICK)
    {
        iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
#pragma omp parallel for
        for (int i = 0; i < mimg->n; i++)
            gray_img->val[i] = mimg->val[i][0]; //*255;
    }

    graph = iftSuperpixelImageToFLIMGraph(num_superpixels, A, mimg, label_img_tmp, graph_feat, gray_img);
    if (gray_img != NULL)
        iftDestroyImage(&gray_img);

    (*label_img) = label_img_tmp;
    label_img_tmp = NULL;

    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
    return graph;
}

/**
 * @fn iftFLIMGraphToImage
 * @brief Convert a band of the node features in a graph to an `iftImage`
 * @note It performs a min-max normalization on the features and scale it to `Imax`
 * @param graph Graph
 * @param labels Superpixel image
 * @param Imax Maximum intensity value
 * @param band node feature to be converted to image
 */
iftImage *iftFLIMGraphToImage(iftFLIMGraph *graph, iftImage *labels, int Imax, int band)
{
    int p, b = band;

    iftIsValidFLIMGraph(graph);
    if ((band < 0) || (band >= (int)graph->num_feats))
        iftError("Invalid band", __func__);

    iftImage *img = iftCreateImage(labels->xsize, labels->ysize, labels->zsize);
    double min = IFT_INFINITY_FLT, max = IFT_INFINITY_FLT_NEG;

    iftFLIMGraphNode *node = graph->nodes;
    for (p = 0; p < graph->num_nodes; p++)
    {
        if (node->feats[b] < min)
            min = node->feats[b];

        if (node->feats[b] > max)
            max = node->feats[b];
        node++;
    }
    node = NULL;

    if (max != 0)
    {
        for (p = 0; p < img->n; p++)
        {
            img->val[p] = (int)((double)(Imax) * ((graph->nodes[labels->val[p] - 1].feats[b] - min) / (max - min)));
            //img->val[p] = (int)((double)(Imax) * (graph->nodes[labels->val[p] - 1].feats[b] / max));
        }
    }

    img->dx = 1;
    img->dy = 1;
    img->dz = 1;

    return img;
}

/**
 * @fn iftFLIMGraphToMImage
 * @brief Convert a graph to a multi-band image
 * @param graph Graph
 * @param labels Superpixel image
 * @return iftMImage*
 * @note The number of bands in the `*iftImage` is equal to the number of features in the graph
 */
iftMImage *iftFLIMGraphToMImage(iftFLIMGraph *graph, iftImage *labels)
{
    int p, b;

    iftIsValidFLIMGraph(graph);
    if (labels == NULL)
        iftError("Labels image is NULL", __func__);

    iftMImage *img = iftCreateMImage(labels->xsize, labels->ysize, labels->zsize, graph->num_feats);
    for (b = 0; b < img->m; b++)
    {
        for (p = 0; p < img->n; p++)
        {
            if (labels->val[p] - 1 < 0 || labels->val[p] - 1 >= graph->num_nodes)
                iftError("Invalid label %d %d ", __func__, labels->val[p], graph->num_nodes);
            img->val[p][b] = graph->nodes[labels->val[p] - 1].feats[b];
        }
    }

    img->dx = 1;
    img->dy = 1;
    img->dz = 1;

    return img;
}

/**
 * @fn iftMImageToFLIMGraph
 * @brief Convert an iftMImage to an iftFLIMGraph
 * @param mimg MImage
 * @param graph_ref Graph reference
 * @return iftFLIMGraph*
 * @note The number of nodes in `graph_ref` must be equal to the number of pixels in `mimg`
 * @note The `graph_ref` is used to copy the adjacency relations
 * @note The features in `mimg` are copied to the output graph
 */
iftFLIMGraph *iftMImageToFLIMGraph(iftMImage *mimg,
                                   iftFLIMGraph *graph_ref)
{
    if (mimg->n != graph_ref->num_nodes)
        iftError("Number of nodes in the graph is different from the number of pixels in the image", __func__);
    iftIsValidFLIMGraph(graph_ref);

    iftFLIMGraph *graph = iftCreateEmptyFLIMGraph(mimg->m, graph_ref->num_nodes);
    graph->num_edges = graph_ref->num_edges;
    
#pragma omp parallel for 
    for (int i = 0; i < mimg->n; i++)
    {
        iftFLIMGraphNode *node = &(graph->nodes[i]);
        iftFLIMGraphNode *node_ref = &(graph_ref->nodes[i]);
        float *mimg_feat = mimg->val[i];
        for (int j = 0; j < mimg->m; j++)
            node->feats[j] = mimg_feat[j];
        node->numPixels = node_ref->numPixels;
        node->numNeighbors = node_ref->numNeighbors;
    }

    iftFLIMGraphNode *node = graph->nodes;
    iftFLIMGraphNode *node_ref = graph_ref->nodes;
    for (int i = 0; i < mimg->n; i++){
        node->adj = iftSetCopy(node_ref->adj);
        node++;
        node_ref++;
    }

    node = NULL;
    node_ref = NULL;

    return graph;
}

/**
 * @fn iftMImageToFLIMGraphInplace
 * @brief Copy the features from the `mimg` to `graph_ref`
 * @param mimg Multi-band image
 * @param graph_ref Graph reference
 * @note The number of nodes in `graph_ref` must be equal to the number of pixels in `mimg`
 */
void iftMImageToFLIMGraphInplace(iftMImage *mimg,
                                 iftFLIMGraph *graph_ref)
{
    if (mimg->n != graph_ref->num_nodes)
        iftError("Number of nodes in the graph is different from the number of pixels in the image", __func__);
    iftIsValidFLIMGraph(graph_ref);

    iftFLIMGraphNode *node = graph_ref->nodes;
    for (int i = 0; i < mimg->n; i++)
    {
        iftFree(node->feats);
        node->feats = iftAllocDoubleArray(mimg->m);
        node++;
    }
    
#pragma omp parallel for 
    for (int i = 0; i < mimg->n; i++)
    {
        iftFLIMGraphNode *node = &(graph_ref->nodes[i]);
        for (int j = 0; j < mimg->m; j++)
            node->feats[j] = mimg->val[i][j];
    }
    graph_ref->num_feats = mimg->m;
    node = NULL;
}

/**
 * @fn iftFLIMGraphToActivations
 * @brief Convert a graph to a vector of iftImage
 * @param graph Graph
 * @param labels Superpixel image
 * @return iftImage**
 * @note The number of images is equal to the number of features in the graph
 */
iftImage **iftFLIMGraphToActivations(iftFLIMGraph *graph, iftImage *labels)
{
    iftImage **activations = (iftImage **)malloc(graph->num_feats * sizeof(iftImage *));

    for (int i = 0; i < (int)(graph->num_feats); i++)
    {
        iftImage *band = iftFLIMGraphToImage(graph, labels, 255, i);
        if (iftIs3DImage(band))
            activations[i] = iftGetXYSlice(band, band->zsize / 2);
        else
            activations[i] = iftCopyImage(band);
        iftDestroyImage(&band);
    }

    return activations;
}

/** @} */

/* --------------------- FLIMGraph attention ----------------------------- */
/** @defgroup FLIMGraphAttention FLIMGraph attention
 * Functions to compute attention values for the nodes in a graph
 *  @{
 */

/**
 * @fn iftFLIMGraphAttentionRelNodeSize
 * @brief Compute the attention values for the nodes in `neighbors` based on the
 * ration between the number of pixels in each node and the total number of
 * pixels in the neighbors
 * @param graph graph with node relations and features
 * @param neighbors indexes of the neighbors
 * @return double* attention values for the neighbors
 */
double *iftFLIMGraphAttentionRelNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors)
{
    double *attention_vals = iftAllocDoubleArray(neighbors->n);

    int num_pixels = 0;
    for (int j = 0; j < neighbors->n; j++)
        num_pixels += graph->nodes[neighbors->val[j]].numPixels;

    for (int j = 0; j < neighbors->n; j++)
        attention_vals[j] = graph->nodes[neighbors->val[j]].numPixels / (double)num_pixels;

    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionNodeSize
 * @brief Compute the attention values for the nodes in `neighbors` with a softmax function
 * based on the number of pixels in each node and the total number of pixels in the neighborhood.
 * @param graph graph with node relations and features
 * @param neighbors indexes of the neighbors
 * @return double* attention values for the neighbors
 */
double *iftFLIMGraphAttentionNodeSize(iftFLIMGraph *graph, iftIntArray *neighbors)
{
    double *attention_vals = iftAllocDoubleArray(neighbors->n);
    double softmax_norm = IFT_EPSILON;

    int num_pixels = 0;
    for (int j = 0; j < neighbors->n; j++)
        num_pixels += graph->nodes[neighbors->val[j]].numPixels;

    for (int j = 0; j < neighbors->n; j++)
    {
        attention_vals[j] = exp((double)graph->nodes[neighbors->val[j]].numPixels / (double)num_pixels);
        softmax_norm += attention_vals[j];
    }
    for (int j = 0; j < neighbors->n; j++)
        attention_vals[j] /= softmax_norm;

    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionNeighborSimilarity
 * @brief Compute the attention values for the nodes in `neighbors` based on the similarity of the features
 * @param graph graph with node relations and features
 * @param neighbors indexes of the neighbors
 * @return double* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionNeighborSimilarity(iftMatrix *features, iftIntArray *neighbors,
                                                   int *kernel_labels, int label)
{
    int nkernels = 0;
    for (int j = 0; j < features->ncols; j++)
        if (kernel_labels[j] == label)
            nkernels++;

    if (nkernels < 1)
        iftError("No kernels found for label %d", __func__, label);

    iftMatrix *A = iftCreateMatrix(nkernels, neighbors->n);
    iftMatrix *AT = iftCreateMatrix(neighbors->n, nkernels);

    // matrix for feature similarity
    for (int j = 0; j < neighbors->n; j++)
    {
        int k = 0;
        for (int b = 0; b < features->ncols; b++)
        {
            if (kernel_labels[b] == label)
            {
                iftMatrixElem(A, k, j) = iftMatrixElem(features, b, neighbors->val[j]);
                iftMatrixElem(AT, j, k) = iftMatrixElem(A, k, j);
                k++;
            }
        }
    }
    iftMatrix *X = iftMultMatrices(AT, A); // ncols: nkernels, nrows: nkernels

    iftDestroyMatrix(&AT);
    iftDestroyMatrix(&A);

    // compute a softmax for each row
    for (int b = 0; b < nkernels; b++)
    {
        double max = iftMatrixElem(X, 0, b);
        for (int j = 0; j < nkernels; j++)
            max = iftMax(iftMatrixElem(X, j, b), max);

        double norm = 0; // normalization factor for the softmax
        for (int j = 0; j < nkernels; j++)
        {
            iftMatrixElem(X, j, b) = exp(iftMatrixElem(X, j, b) - max /*/sum*/);
            norm += iftMatrixElem(X, j, b);
        }
        if (norm == 0)
            norm = 1;

        // compute the attention values
        for (int j = 0; j < nkernels; j++)
            iftMatrixElem(X, j, b) /= norm;
    }

    return X;
}

/**
 * @fn iftFLIMGraphAttentionCosineSimilarity
 * @brief Compute the attention values for the nodes in `neighbors` based on the cosine similarity of the features
 * @param graph graph with node relations and features
 * @param neighbors indexes of the neighbors
 * @return double* attention values for the neighbors
 */
double *iftFLIMGraphAttentionCosineSimilarity(iftFLIMGraph *graph, iftIntArray *neighbors)
{
    double *attention_vals = iftAllocDoubleArray(neighbors->n);

    iftMatrix *feat_matrix_v0 = iftCreateMatrix(graph->num_feats, 1);           // size: [1 x m]
    iftMatrix *feat_matrix_T = iftCreateMatrix(neighbors->n, graph->num_feats); // size: [m x k]

    // matrix for feature similarity
    for (int j = 0; j < neighbors->n; j++)
    {
        double viSize = 0; // ||vi|| = sqrt(sum({vil | l in [1, num_feats]}))
        for (int b = 0; b < graph->num_feats; b++)
        {
            iftMatrixElem(feat_matrix_T, j, b) = graph->nodes[neighbors->val[j]].feats[b];
            viSize += pow(iftMatrixElem(feat_matrix_T, j, b), 2);
        }
        viSize = sqrt(viSize);
        if (viSize == 0)
            viSize = 1;

        for (int b = 0; b < graph->num_feats; b++)
        {
            iftMatrixElem(feat_matrix_T, j, b) /= viSize; // normalize the feature matrix
            if (j == 0)
                iftMatrixElem(feat_matrix_v0, b, 0) = iftMatrixElem(feat_matrix_T, 0, b); // transpose the feature matrix
        }
    }

    iftMatrix *X = iftMultMatrices(feat_matrix_v0, feat_matrix_T); // ncols: 1, nrows: neighbors->n
    iftDestroyMatrix(&feat_matrix_T);
    iftDestroyMatrix(&feat_matrix_v0);

    // compute a softmax for each row --- per neighbor
    for (int b = 0; b < neighbors->n; b++)
    {
        // compute the attention values
        attention_vals[b] = iftMatrixElem(X, b, 0);
    }

    iftDestroyMatrix(&X);

    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionCosineSimilarity
 * @brief Compute the attention values for the nodes in `neighbors` based on the cosine similarity of the features
 * @param graph graph with node relations and features
 * @param neighbors indexes of the neighbors
 * @return double* attention values for the neighbors
 */
double *iftFLIMGraphAttentionCosineSimilarity_v1(iftFLIMGraph *graph, iftIntArray *neighbors)
{
    double *attention_vals = iftAllocDoubleArray(neighbors->n);

    iftMatrix *feat_matrix = iftCreateMatrix(graph->num_feats, neighbors->n);
    iftMatrix *feat_matrix_T = iftCreateMatrix(neighbors->n, graph->num_feats);

    // matrix for feature similarity
    for (int j = 0; j < neighbors->n; j++)
    {
        double viSize = 0; // ||vi|| = sqrt(sum({vil | l in [1, num_feats]}))
        for (int b = 0; b < graph->num_feats; b++)
        {
            iftMatrixElem(feat_matrix, b, j) = graph->nodes[neighbors->val[j]].feats[b]; // compute feature matrix F in R^{num_nodes x num_feats}
            viSize += pow(iftMatrixElem(feat_matrix, b, j), 2);
        }
        viSize = sqrt(viSize);
        if (viSize == 0)
            viSize = 1;

        for (int b = 0; b < graph->num_feats; b++)
        {
            iftMatrixElem(feat_matrix, b, j) /= viSize;                            // normalize the feature matrix
            iftMatrixElem(feat_matrix_T, j, b) = iftMatrixElem(feat_matrix, b, j); // transpose the feature matrix
        }
    }
    iftMatrix *X = iftMultMatrices(feat_matrix, feat_matrix_T); // ncols: neighbors->n, nrows: neighbors->n

    iftDestroyMatrix(&feat_matrix_T);
    iftDestroyMatrix(&feat_matrix);

    // compute a softmax for each row --- per neighbor
    for (int b = 0; b < neighbors->n; b++)
    {
        // compute the attention values
        for (int j = 0; j < neighbors->n; j++)
            if (b != j)
                attention_vals[j] += (iftMatrixElem(X, j, b));
    }

    iftDestroyMatrix(&X);

    return attention_vals;
}

// function in construction
iftMatrix *iftFLIMGraphAttentionClassDominance_test(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    int *nkernels = iftAllocIntArray(nlabels); // number of kernels per label
    for (int j = 0; j < H->ncols; j++)
        nkernels[kernel_labels[j] - minLabel]++;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
        }

        double sL[nlabels];    // label strength
        double sK[nlabels];    // mean strength of the other labels
        double alpha[nlabels]; // reduction factor per label based on the mean activation differences

        for (int j = 0; j < nlabels; j++)
        {
            sL[j] = 0;
            sK[j] = 0;
        }

        for (int j = 0; j < H->ncols; j++)
            sL[kernel_labels[j] - minLabel] += iftMatrixElem(H_norm, j, i);

        for (int j = 0; j < nlabels; j++)
            sL[j] /= nkernels[j];

        for (int j = 0; j < nlabels; j++)
        {
            for (int l = 0; l < nlabels; l++)
            {
                if (l != j)
                    sK[l] += sL[j];
            }
        }

        for (int j = 0; j < nlabels; j++)
        {
            sK[j] /= (nlabels - 1);
            alpha[j] = (iftMax(sK[j], sL[j]) != 0) ? iftMax((sK[j] - sL[j]) / iftMax(sK[j], sL[j]), 0) : 0;
        }

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;

            double beta = iftMax(iftMatrixElem(H_norm, j, i), sK[label]) != 0 ? (iftMatrixElem(H_norm, j, i) - sK[label]) / iftMax(iftMatrixElem(H_norm, j, i), sK[label]) : 0;
            beta = iftMax(beta, 0);
            beta = 1;

            double label_dominance = iftMax(sL[label], iftMatrixElem(H_norm, j, i)) != 0 ? (sL[label] - iftMatrixElem(H_norm, j, i)) / iftMax(sL[label], iftMatrixElem(H_norm, j, i)) : 0; // [-1,1]

            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            /*if(i == 0){
                printf("A[%d][%d] = %f * %f + 1 - %f * %f = %f + %f = %f \n", j, i,
                    label_dominance, iftMatrixElem(H_norm, j, i), alpha[label], beta,
                    label_dominance*iftMatrixElem(H_norm, j, i), 1 - alpha[label] * beta,
                    label_dominance * iftMatrixElem(H_norm, j, i) + 1 - alpha[label] * beta
                );
                printf("A[%d][%d] = (%f * %f) + 1 = %f \n", j, i,
                    label_dominance, iftMatrixElem(H_norm, j, i),
                    (label_dominance*iftMatrixElem(H_norm, j, i)) + 1
                );
            }*/
            iftMatrixElem(H_norm, j, i) = iftMax(label_dominance * iftMatrixElem(H_norm, j, i) + 1 - alpha[label] * beta, 0);
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        double size = (double)graph->nodes[i].numPixels;

        iftSet *node = graph->nodes[i].adj;
        double max_diff = (double)fabs((double)graph->nodes[node->elem].numPixels - size);
        double w[graph->nodes[i].numNeighbors];
        int k = 0;
        while (node != NULL)
        {
            w[k] = fabs((double)graph->nodes[node->elem].numPixels - size);
            max_diff = iftMax(max_diff, w[k]);
            node = node->next;
            k++;
        }
        if (max_diff == 0)
            max_diff = 1;

        double sum = 0;
        k = 0;
        while (k < graph->nodes[i].numNeighbors)
        {
            w[k] = exp((max_diff - w[k]) / max_diff);
            sum += w[k];
            k++;
        }
        if (sum == 0)
            sum = 1;

        for (int j = 0; j < H->ncols; j++)
        {
            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            int k = 0;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += /*(w[k]/sum)**/ iftMatrixElem(H_norm, j, node->elem);
                k++;
                node = node->next;
            }

            iftMatrixElem(attention_vals, j, i) /= graph->nodes[i].numNeighbors;
            /*if(i == 0)
            {
                printf("H'[%d][%d] = %f * (%f + %f) = %f * %f = %f \n", j, i,
                    iftMatrixElem(H, j, i), iftMatrixElem(H_norm, j, i), iftMatrixElem(attention_vals, j, i),
                    iftMatrixElem(H, j, i), (iftMatrixElem(H_norm, j, i) + iftMatrixElem(attention_vals, j, i))/2,
                    ((iftMatrixElem(H_norm, j, i) + iftMatrixElem(attention_vals, j, i))/2)*iftMatrixElem(H, j, i)
                );
            }*/

            iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, i);
            // iftMatrixElem(attention_vals, j, i) /= 2;
        }
    }
    iftDestroyMatrix(&H_norm);

    // iftMatrix *out = iftMatricesMultiplicationPointWise(H, attention_vals);
    // iftDestroyMatrix(&attention_vals);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominanceCosine
 * @brief Compute attention values using cosine similarity and class dominance,
 * normalizing the input data and applying softmax to the cosine similarity among graph neighbors.
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominanceCosine(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    // compute vector norms (size) for cosine similarity
    iftMatrix *vector_norm = iftCreateMatrix(H->nrows, 1);
    for (int i = 0; i < H->nrows; i++)
    {
        double norm = 0;
        for (int j = 0; j < graph->num_feats; j++)
            norm += pow(graph->nodes[i].feats[j], 2);
        iftMatrixElem(vector_norm, i, 0) = sqrt(norm);
    }

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
        }

        double label_strength[nlabels];
        int nkernels[nlabels];
        for (int j = 0; j < nlabels; j++)
        {
            label_strength[j] = 0;
            nkernels[j] = 0;
        }

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            label_strength[label] += iftMatrixElem(H_norm, j, i);
            nkernels[label]++;
        }

        for (int j = 0; j < nlabels; j++)
            label_strength[j] /= nkernels[j];

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            double max_val = iftMax(label_strength[label], iftMatrixElem(H_norm, j, i));

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (label_strength[label] - iftMatrixElem(H_norm, j, i)) / max_val; // [-1,1]

            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        int num_neighbors = graph->nodes[i].numNeighbors;

        // compute softmax of cosine similarity among neighbors
        iftMatrix *cosine_similarity = iftCreateMatrix(num_neighbors, 1);
        iftSet *node = graph->nodes[i].adj;
        int k = 0;
        double sum = 0, max = -1;
        while (node != NULL)
        {
            double dotProduct = 0;
            for (int b = 0; b < graph->num_feats; b++)
                dotProduct += graph->nodes[i].feats[b] * graph->nodes[node->elem].feats[b];

            iftMatrixElem(cosine_similarity, k, 0) = dotProduct / (iftMatrixElem(vector_norm, i, 0) * iftMatrixElem(vector_norm, node->elem, 0));
            max = iftMax(max, iftMatrixElem(cosine_similarity, k, 0));
            node = node->next;
            k++;
        }

        k = 0;
        node = graph->nodes[i].adj;
        while (node != NULL)
        {
            iftMatrixElem(cosine_similarity, k, 0) = exp(iftMatrixElem(cosine_similarity, k, 0) - max);
            sum += iftMatrixElem(cosine_similarity, k, 0);
            node = node->next;
            k++;
        }

        if (max == 0)
            max = 1;

        for (int j = 0; j < H->ncols; j++)
        {
            iftMatrixElem(attention_vals, j, i) = 0;
            node = graph->nodes[i].adj;
            k = 0;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += (iftMatrixElem(cosine_similarity, k, 0) / sum) * iftMatrixElem(H_norm, j, node->elem);
                node = node->next;
                k++;
            }
            // iftMatrixElem(attention_vals, j, i) /= nsimilarNeigh;
            iftMatrixElem(attention_vals, j, i) += pow(iftMatrixElem(attention_vals, j, i), 2);
            iftMatrixElem(attention_vals, j, i) += pow(iftMatrixElem(H_norm, j, i), 2);
        }
        iftDestroyMatrix(&cosine_similarity);
    }
    iftDestroyMatrix(&H_norm);
    iftDestroyMatrix(&vector_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominance_winnerPower2
 * @brief Compute attention values using class dominance
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominancePower2(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
        }

        double label_strength[nlabels];
        int nkernels[nlabels];
        for (int j = 0; j < nlabels; j++)
        {
            label_strength[j] = 0;
            nkernels[j] = 0;
        }

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            label_strength[label] += iftMatrixElem(H_norm, j, i);
            nkernels[label]++;
        }

        for (int j = 0; j < nlabels; j++)
            label_strength[j] /= nkernels[j];

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            double max_val = iftMax(label_strength[label], iftMatrixElem(H_norm, j, i));

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (label_strength[label] - iftMatrixElem(H_norm, j, i)) / iftMax(label_strength[label], iftMatrixElem(H_norm, j, i)); // [-1,1]

            /*if((i == 61 && j == 24) || (i == 12 && j == 21))
            {
                printf("A[%d][%d] = %f*((%f-%f)/%f) + 1 = (%f * %f) + 1 = %f, \n", i, j,
                    iftMatrixElem(H_norm, j, i), label_strength[label], iftMatrixElem(H_norm, j, i), max_val,
                    iftMatrixElem(H_norm, j, i), label_dominance, (iftMatrixElem(H_norm, j, i)*label_dominance)+1);
            }*/

            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        for (int j = 0; j < H->ncols; j++)
        {
            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, node->elem);
                node = node->next;
            }
            iftMatrixElem(attention_vals, j, i) /= graph->nodes[i].numNeighbors;
            iftMatrixElem(attention_vals, j, i) = pow(iftMatrixElem(attention_vals, j, i), 2);
            iftMatrixElem(attention_vals, j, i) += pow(iftMatrixElem(H_norm, j, i), 2);
        }
    }
    iftDestroyMatrix(&H_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominance_size
 * @brief Compute attention values using class dominance and weighting the neighbors contribution by their size
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominance_size(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        double label_strength[nlabels];
        int nkernels[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
            label_strength[i] = 0;
            nkernels[i] = 0;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
            nkernels[label]++;
        }

        for (int j = 0; j < nlabels; j++)
        {
            if (i == 0)
                printf("label %d: min=%f max=%f\n", j, min[j], max[j]);
        }

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            if (i == 0 && j == 0)
            {
                printf("[%d][%d]: H = %f, H_norm = %f, ", j, i, iftMatrixElem(H, j, i), iftMatrixElem(H_norm, j, i));
            }
            label_strength[label] += iftMatrixElem(H_norm, j, i);
        }

        for (int j = 0; j < nlabels; j++)
            label_strength[j] /= nkernels[j];

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            double max_val = iftMax(label_strength[label], iftMatrixElem(H_norm, j, i));

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (label_strength[label] - iftMatrixElem(H_norm, j, i)) / iftMax(label_strength[label], iftMatrixElem(H_norm, j, i)); // [-1,1]

            /*if((i == 61 && j == 24) || (i == 12 && j == 21))
            {
                printf("A[%d][%d] = %f*((%f-%f)/%f) + 1 = (%f * %f) + 1 = %f, \n", i, j,
                    iftMatrixElem(H_norm, j, i), label_strength[label], iftMatrixElem(H_norm, j, i), max_val,
                    iftMatrixElem(H_norm, j, i), label_dominance, (iftMatrixElem(H_norm, j, i)*label_dominance)+1);
            }*/

            if ((i == 1 && j == 0))
            {
                printf("A = %f*((%f-%f)/%f) + 1 = (%f * %f) + 1 = %f, ",
                       iftMatrixElem(H_norm, j, i), label_strength[label], iftMatrixElem(H_norm, j, i), max_val,
                       iftMatrixElem(H_norm, j, i), label_dominance, (iftMatrixElem(H_norm, j, i) * label_dominance) + 1);
            }
            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        int neighborhood_size = 0;
        iftSet *node;
        node = graph->nodes[i].adj;
        while (node != NULL)
        {
            neighborhood_size += graph->nodes[node->elem].numPixels;
            node = node->next;
        }

        int vi_size = graph->nodes[i].numPixels;

        for (int j = 0; j < H->ncols; j++)
        {
            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            double sum_importance = 0;
            while (node != NULL)
            {
                double size_importance = exp(-0.5 * pow((graph->nodes[node->elem].numPixels - vi_size) / neighborhood_size, 2));
                iftMatrixElem(attention_vals, j, i) += size_importance * iftMatrixElem(H_norm, j, node->elem);
                sum_importance += size_importance;
                node = node->next;
            }

            if (sum_importance != 0)
                iftMatrixElem(attention_vals, j, i) /= sum_importance;

            iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, i);
        }
    }
    iftDestroyMatrix(&H_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominance_softmaxNeigh
 * @brief Compute attention values using class dominance and softmax normalization
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominance_softmaxNeigh(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        int nkernels[nlabels];
        double modes[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
            nkernels[i] = 0;
            modes[i] = 0;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
            nkernels[label]++;
        }

        /*
        double **ordering_values = (double **)malloc(nlabels * sizeof(double *));
        int **indexes = (int **)malloc(nlabels * sizeof(int *));
        int label_sizes[nlabels];
        for(int j = 0; j < nlabels; j++){
            ordering_values[j] = (double *)malloc(nkernels[j] * sizeof(double));
            indexes[j] = (int *)malloc(nkernels[j] * sizeof(int));
            label_sizes[j] = 0;
            //if(i == 0) printf("label %d: min=%f max=%f\n", j, min[j], max[j]);
        }*/

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            /*if(i == 0 && j == 0)
            {
                printf("[%d][%d]: H = %f, H_norm = %f, ", j, i, iftMatrixElem(H, j, i), iftMatrixElem(H_norm, j, i));
            }*/

            /*ordering_values[label][label_sizes[label]] = iftMatrixElem(H_norm, j, i);
            indexes[label][label_sizes[label]] = label_sizes[label];
            label_sizes[label]++;*/
            modes[label] += iftMatrixElem(H_norm, j, i);
        }

        for (int j = 0; j < nlabels; j++)
        {
            /*iftDQuickSort(ordering_values[j], indexes[j], 0, label_sizes[j]-1, IFT_INCREASING);
            if(label_sizes[j] % 2 != 0)
                modes[j] = ordering_values[j][indexes[j][label_sizes[j]/2]];
            else
                modes[j] = (ordering_values[j][indexes[j][label_sizes[j]/2]] + ordering_values[j][indexes[j][(label_sizes[j]/2)-1]])/2;*/
            modes[j] /= nkernels[j];
        }
        /*for(int j = 0; j < nlabels; j++)
        {
            free(ordering_values[j]);
            free(indexes[j]);
        }
        free(ordering_values);
        free(indexes);*/

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            double max_val = modes[label] + iftMatrixElem(H_norm, j, i);

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (modes[label] - iftMatrixElem(H_norm, j, i)) / iftMax(modes[label], iftMatrixElem(H_norm, j, i)); // [-1,1]

            /*if((i == 61 && j == 24) || (i == 12 && j == 21))
            {
                printf("A[%d][%d] = %f*((%f-%f)/%f) + 1 = (%f * %f) + 1 = %f, \n", i, j,
                    iftMatrixElem(H_norm, j, i), modes[label], iftMatrixElem(H_norm, j, i), max_val,
                    iftMatrixElem(H_norm, j, i), label_dominance, (iftMatrixElem(H_norm, j, i)*label_dominance)+1);
            }*/

            /*if((i == 1 && j == 0))
            {
                printf("A = %f*((%f-%f)/%f) + 1 = (%f * %f) + 1 = %f, ",
                    iftMatrixElem(H_norm, j, i), modes[label], iftMatrixElem(H_norm, j, i), max_val,
                    iftMatrixElem(H_norm, j, i), label_dominance, (iftMatrixElem(H_norm, j, i)*label_dominance)+1);
            }*/

            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        int vi = graph->nodes[i].numPixels;
        int num_neighbors = graph->nodes[i].numNeighbors;

        double neighborhood_size = 0;
        double neigh_variance = 0, neighb_sum2 = 0;
        iftSet *node = graph->nodes[i].adj;
        while (node != NULL)
        {
            neighborhood_size += graph->nodes[node->elem].numPixels;
            neighb_sum2 += pow(graph->nodes[node->elem].numPixels, 2);
            node = node->next;
        }
        neigh_variance = neighb_sum2 / num_neighbors - pow(neighborhood_size / num_neighbors, 2);

        double sum_exp = 0;
        double vj[num_neighbors];
        node = graph->nodes[i].adj;
        int k = 0;
        while (node != NULL)
        {
            vj[k] = exp(-0.5 * pow((graph->nodes[node->elem].numPixels - vi), 2) / neigh_variance);
            sum_exp += vj[k];
            k++;
            node = node->next;
        }

        for (int j = 0; j < H->ncols; j++)
        {
            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            int k = 0;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, node->elem) * (vj[k] / sum_exp);
                k++;
                node = node->next;
            }
            iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, i);
        }
    }
    iftDestroyMatrix(&H_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominance_v2
 * @brief Compute attention values using class dominance
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominance_v2(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
        }

        double label_strength[nlabels];
        int nkernels[nlabels];
        for (int j = 0; j < nlabels; j++)
        {
            label_strength[j] = 0;
            nkernels[j] = 0;
        }

        // compute kernel strength and 
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            label_strength[label] += iftMatrixElem(H_norm, j, i);
            nkernels[label]++;
        }

        for (int j = 0; j < nlabels; j++)
            label_strength[j] /= nkernels[j];

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            
            double ls = label_strength[1];
            if(label == 0) ls = 1.0 - ls; 

            double max_val = iftMax(ls, iftMatrixElem(H_norm, j, i));

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (ls - iftMatrixElem(H_norm, j, i)) / max_val; // [-1,1]

            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        for (int j = 0; j < H->ncols; j++)
        {

            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, node->elem);
                node = node->next;
            }

            iftMatrixElem(attention_vals, j, i) /= graph->nodes[i].numNeighbors;
            iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, i);
        }
    }
    iftDestroyMatrix(&H_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphAttentionClassDominance
 * @brief Compute attention values using class dominance
 * @param H: matrix with the features
 * @param graph: graph with node relations and features
 * @param kernel_labels: labels of the kernels
 * @return iftMatrix* attention values for the neighbors
 */
iftMatrix *iftFLIMGraphAttentionClassDominance(iftMatrix *H, iftFLIMGraph *graph, int *kernel_labels)
{
    // get the number of labels and the minimum label
    int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
    for (int i = 1; i < H->ncols; i++)
    {
        if (kernel_labels[i] < minLabel)
            minLabel = kernel_labels[i];
        if (kernel_labels[i] > nlabels)
            nlabels = kernel_labels[i];
    }
    nlabels = nlabels - minLabel + 1;

    iftMatrix *H_norm = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        // min-max normalization
        double min[nlabels], max[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            min[i] = IFT_INFINITY_FLT;
            max[i] = IFT_INFINITY_FLT_NEG;
        }
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (iftMatrixElem(H, j, i) < min[label])
                min[label] = iftMatrixElem(H, j, i);
            if (iftMatrixElem(H, j, i) > max[label])
                max[label] = iftMatrixElem(H, j, i);
        }

        double label_strength[nlabels];
        int nkernels[nlabels];
        for (int j = 0; j < nlabels; j++)
        {
            label_strength[j] = 0;
            nkernels[j] = 0;
        }

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if (max[label] == min[label])
                iftMatrixElem(H_norm, j, i) = 0;
            else
                iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H, j, i) - min[label]) / (max[label] - min[label]);

            label_strength[label] += iftMatrixElem(H_norm, j, i);
            nkernels[label]++;
        }

        for (int j = 0; j < nlabels; j++)
            label_strength[j] /= nkernels[j];

        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            double max_val = iftMax(label_strength[label], iftMatrixElem(H_norm, j, i));

            double label_dominance = 0;
            if (max_val != 0)
                label_dominance = (label_strength[label] - iftMatrixElem(H_norm, j, i)) / max_val; // [-1,1]

            iftMatrixElem(H_norm, j, i) = (iftMatrixElem(H_norm, j, i) * label_dominance) + 1;
        }
    }

    iftMatrix *attention_vals = iftCreateMatrix(H->ncols, H->nrows);
    for (int i = 0; i < H->nrows; i++)
    {
        for (int j = 0; j < H->ncols; j++)
        {
            int label = kernel_labels[j] - minLabel;
            if(label == 0){
                iftMatrixElem(attention_vals, j, i) = 1;
                continue;
            }
            iftMatrixElem(attention_vals, j, i) = 0;
            iftSet *node = graph->nodes[i].adj;
            while (node != NULL)
            {
                iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, node->elem);
                node = node->next;
            }

            iftMatrixElem(attention_vals, j, i) /= graph->nodes[i].numNeighbors;
            iftMatrixElem(attention_vals, j, i) += iftMatrixElem(H_norm, j, i);
        }
    }
    iftDestroyMatrix(&H_norm);
    return attention_vals;
}

/**
 * @fn iftFLIMGraphGetAttentionValues
 * @brief Compute the attention values for the nodes in `neighbors`
 * @param neighbors: indexes of the neighbors
 * @param graph: graph with node relations and features
 * @param attention_opt: attention option
 * @return double*: attention values for the neighbors
 */
double *iftFLIMGraphGetAttentionValues(iftIntArray *neighbors, iftFLIMGraph *graph, iftFLIMGraphAttention *attention_opt)
{
    double *attention_vals = NULL;

    switch (*attention_opt)
    {
    case IFT_FLIMGRAPH_ATTENTION_NONE:
        break;

    case IFT_FLIMGRAPH_ATTENTION_NODE_RELSIZE:
        attention_vals = iftFLIMGraphAttentionRelNodeSize(graph, neighbors);
        break;

    case IFT_FLIMGRAPH_ATTENTION_NODE_SIZE:
        attention_vals = iftFLIMGraphAttentionNodeSize(graph, neighbors);
        break;

    /*case IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL:
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        break;*/

    case IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIZE:
        attention_vals = iftFLIMGraphAttentionNodeSize(graph, neighbors);
        // double *attention_vals_simil = iftFLIMGraphAttentionNeighborSimilarity(graph, neighbors);
        /*for(int j = 0; j < neighbors->n; j++)
            attention_vals[j] += attention_vals_simil[j];
        iftFree(attention_vals_simil);*/
        iftWarning("Attention partially implemented.", __func__);
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_TEST:
        /*attention_opt = IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL;
        attention_vals = iftFLIMGraphAttentionNeighborSimilarity(graph, neighbors);*/
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE:
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_POWER2:
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_SOFTMAX_NEIGHB:
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        break;

    case IFT_FLIMGRAPH_ATTENTION_COSINE_SIMILARITY:
        attention_vals = iftFLIMGraphAttentionCosineSimilarity(graph, neighbors);
        break;

    default:
        *attention_opt = IFT_FLIMGRAPH_ATTENTION_NONE;
        iftWarning("Attention option not implemented. Using IFT_FLIMGRAPH_ATTENTION_NONE", __func__);
        break;
    }

    return attention_vals;
}

/** @} */

/* --------------------- FLIMGraph neighbor features ----------------------------- */
/** @defgroup FLIMGraphNeighbFeat Neighbor features
 * Functions to compute the features of the neighbors of a node in a FLIMGraph.
 *  @{
 */

/**
 * @fn iftGetNeighborsFLIMGraph
 * @brief Insert in `heap` at most k neighbors of the nodes in `roots` whose features are
 * most similar with those in `feature_reference`.
 *
 * The function inserts up to `k` neighbors of the nodes in `roots` into the heap,
 * selecting those whose features are most (or least) similar to the `feature_reference`,
 * depending on the `most_similar` flag.
 *
 * @param graph: graph with node relations and features
 * @param roots: initial nodes indexes to get the neighbors
 * @param k: number of neighbors to get
 * @param dissimilarity: array to store the dissimilarity between the reference features and those from the neighbors.
 * @param feature_reference: reference node features
 * @param heap: heap of nodes to store the neighbors found
 * @warning Only neighbors with white state in heap may be inserted to it.
 *
 * @author Isabela
 * @date abr, 2023
 */
void iftGetNeighborsFLIMGraph(iftFLIMGraph *graph, int *roots, int num_roots,
                              int k, double *dissimilarity,
                              double *feature_reference,
                              iftDHeap *heap,
                              bool most_similar)
{
    if (roots == NULL)
        iftError("Roots is NULL", __func__);

    int *root_ptr = roots;
    for (int r = 0; r < num_roots; r++)
    {
        iftSet *neigh = graph->nodes[*root_ptr].adj;
        while (neigh != NULL)
        {
            if (heap->color[neigh->elem] == IFT_WHITE)
            {
                double dist = 0;
                double *feats = graph->nodes[neigh->elem].feats;
                for (int i = 0; i < graph->num_feats; i++)
                {
                    double feat_val = feature_reference[i] - feats[i];
                    dist += feat_val * feat_val;
                }

                // if te heap already has k elements
                // keep the most similar (or less similar) elements
                if (heap->last + 1 == k)
                {
                    int last = heap->node[0]; // look at the root of the heap
                    if ((most_similar && dist < dissimilarity[last]) ||
                        (!most_similar && dist > dissimilarity[last]))
                    {
                        iftRemoveDHeap(heap);
                        dissimilarity[neigh->elem] = dist;
                        iftInsertDHeap(heap, neigh->elem);
                    }
                }
                else
                {
                    dissimilarity[neigh->elem] = dist;
                    iftInsertDHeap(heap, neigh->elem);
                }
            }
            neigh = neigh->next;
        }
        root_ptr++;
    }
}

/**
 * @fn iftGetKNeighborsFLIMGraph
 * @brief Return `k` neighbors of node_index according to the GFLIM neighbor selection strategy.
 * @param node_index: initial node to get the neighbors
 * @param graph: graph with node relations and features
 * @param k: number of neighbors
 * @param dilation: number of skips to get neighbors (>= 1) (similar to dilation in convolutional network)
 * @param most_similar: if true, the neighbors are the most similar to the reference node
 * @return iftIntArray with the indexes of the k neighbors
 * @note The neighbors are selected according to the distance between the features of the nodes and the features of the initial node.
 * @author Isabela
 * @date abr, 2023
 */
iftIntArray *iftGetKNeighborsFLIMGraph(int node_index,
                                       iftFLIMGraph *graph, int k,
                                       int dilation, bool most_similar, iftFLIMGraphAttention attention_opt)
{
    if(graph == NULL)
        iftError("Graph is NULL", __func__);
    if(graph->num_nodes < 1)
        iftError("Graph has no nodes: %d", __func__, graph->num_nodes);

    double *dissimilarity = (double *)malloc(graph->num_nodes * sizeof(double));
    iftIntArray *neighbors = iftCreateIntArray(k);

    // MaxHeap para armazenar até k vizinhos mais similares.
    // Se a quantidade de vizinhos for maior que k, remove o menos similar
    // do heap (que está no topo) usando "pop" e verifica quem é o mais similar
    // para nova inserção.
    iftDHeap *auxiliaryHeap = iftCreateDHeap(graph->num_nodes, dissimilarity);
    iftDHeap *heap = iftCreateDHeap(graph->num_nodes, dissimilarity);
    if (most_similar)
        auxiliaryHeap->removal_policy = MAXVALUE;
    else
        heap->removal_policy = MAXVALUE;

    if(node_index < 0 || node_index >= graph->num_nodes)
        iftError("Node index out of bounds: %d", __func__, node_index);

    dissimilarity[node_index] = 0.0;
    iftInsertDHeap(auxiliaryHeap, node_index);
    iftRemoveDHeap(auxiliaryHeap);
    // neighbors->val[0] = node_index;
    neighbors->n = 1;
    iftInsertDHeap(heap, node_index);

    int skips_performed = 1;
    int roots[k];

    roots[0] = node_index;
    int i = 1; // number of values in root

    while (neighbors->n < k)
    {
        int miss_neighbors = k - neighbors->n;
        iftGetNeighborsFLIMGraph(graph, roots, i,
                                 miss_neighbors,
                                 dissimilarity,
                                 graph->nodes[node_index].feats,
                                 auxiliaryHeap,
                                 most_similar);

        if (iftEmptyDHeap(auxiliaryHeap))
            break; // there are no neighbors to include
        // bool new_iter = (auxiliaryHeap->last + 1 < miss_neighbors || skips_performed != dilation);

        i = 0;
        int found_neighbors = auxiliaryHeap->last + 1;
        // adiciona os k vizinhos selecionados (no auxiliaryHeap) em um minHeap
        while (!iftEmptyDHeap(auxiliaryHeap))
        {
            int node_index = iftRemoveDHeap(auxiliaryHeap);
            if (skips_performed == dilation)
            {
                // neighbors->val[neighbors->n + auxiliaryHeap->last + 1] = node_index;
                roots[i] = node_index;
                iftInsertDHeap(heap, node_index);
                i++;
            }
        }
        neighbors->n += found_neighbors;
        if (skips_performed != dilation)
            skips_performed++;
        else
            skips_performed = 1;
    }

    i = 0;
    while (i < neighbors->n)
    {
        neighbors->val[i] = iftRemoveDHeap(heap);
        i++;
    }

    free(dissimilarity);
    iftDestroyDHeap(&auxiliaryHeap);
    iftDestroyDHeap(&heap);

    double *attention_vals = iftFLIMGraphGetAttentionValues(neighbors, graph, &attention_opt);
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE)
    {
        for (int i = 0; i < neighbors->n; i++)
            neighbors->val[i] *= attention_vals[i];
    }

    if (attention_vals != NULL)
        iftFree(attention_vals);

    return neighbors;
}

/**
 * @fn iftFLIMgraphAggregateFeatures
 * @brief Aggregate features of the nodes in `neighbors`
 * @param p Node index
 * @param neighbors Array of neighbors
 * @param graph Graph
 * @param num_adj Number of neighbors to be included in the matrix
 * @param aggregation_opt Aggregation option
 * @return double* Aggregated features
 */
double *iftFLIMgraphAggregateFeatures(int p,
                                      iftIntArray *neighbors,
                                      iftFLIMGraph *graph,
                                      int num_adj,
                                      iftGFLIMAggregationType aggregation_opt)
{
    double *output_features = NULL;
    int i = 0;
    double w[neighbors->n - 1];

    if (neighbors->val[0] != p)
        iftError("Neighbor of %d is %d", __func__, p, neighbors->val[0]);

    switch (aggregation_opt)
    {
    case IFT_GFLIM_AGGREGATION_CONCAT:

        output_features = iftAllocDoubleArray(num_adj * graph->num_feats);

        while (i < neighbors->n)
        {
            memcpy(output_features + i * graph->num_feats, graph->nodes[neighbors->val[i]].feats, graph->num_feats * sizeof(double));
            i++;
        }

        break;

    case IFT_GFLIM_AGGREGATION_MEAN:

        output_features = iftAllocDoubleArray(2 * graph->num_feats);

        memcpy(output_features, graph->nodes[p].feats, graph->num_feats * sizeof(double));
        memcpy(output_features + graph->num_feats, graph->nodes[neighbors->val[1]].feats, graph->num_feats * sizeof(double));

        i = 2;
        while (i < neighbors->n)
        {
            for (int b = 0; b < graph->num_feats; b++)
                output_features[graph->num_feats + b] += graph->nodes[neighbors->val[i]].feats[b];
            i++;
        }

        for (int b = graph->num_feats; b < 2 * graph->num_feats; b++)
            output_features[b] /= (float)(neighbors->n - 1);

        break;

    case IFT_GFLIM_AGGREGATION_WMEAN:

        output_features = iftAllocDoubleArray(2 * graph->num_feats);

        double size = (double)graph->nodes[p].numPixels;
        double max_diff = fabs((double)graph->nodes[neighbors->val[1]].numPixels - size);

        memcpy(output_features, graph->nodes[p].feats, graph->num_feats * sizeof(double));

        i = 1;
        while (i < neighbors->n)
        {
            w[i - 1] = fabs((double)graph->nodes[neighbors->val[i]].numPixels - size);
            max_diff = iftMax(max_diff, w[i - 1]);
            i++;
        }
        if (max_diff == 0)
            max_diff = 1;

        i = 0;
        double sum = 0;
        while (i < neighbors->n - 1)
        {
            w[i] = exp((max_diff - w[i]) / max_diff);
            sum += w[i];
            i++;
        }

        i = 1;
        while (i < neighbors->n)
        {
            for (int b = 0; b < graph->num_feats; b++)
                output_features[graph->num_feats + b] += (float)((w[i - 1] / sum) * graph->nodes[neighbors->val[i]].feats[b]);
            i++;
        }
        break;

    default:
        iftError("Invalid aggregation option", __func__);
        break;
    }

    return output_features;
}

/**
 * @fn iftFLIMGraphToFeatureMatrix
 * @brief Create a matrix from a graph
 * @param graph Graph
 * @param num_adj Number of neighbors to be included in the matrix
 * @param attention_opt Attention option
 * @param aggregation_opt Aggregation option
 * @param neighbors_all Array of neighbors for each node.
 * @return iftMatrix*
 * @note The matrix size depends on the aggregation option:
 * - IFT_GFLIM_AGGREGATION_CONCAT: num_adj * graph->num_feats
 * - IFT_GFLIM_AGGREGATION_MEAN: 2 * graph->num_feats
 * - IFT_GFLIM_AGGREGATION_WMEAN: 2 * graph->num_feats
 * @note `neighbors_al` is used to store the neighbors of each node when the attention option is IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL.
 */
iftMatrix *iftFLIMGraphToFeatureMatrix(iftFLIMGraph *graph,
                                       int num_adj,
                                       iftFLIMGraphAttention attention_opt,
                                       iftGFLIMAggregationType aggregation_opt/*,
                                       iftIntArray **neighbors_all*/)
{
    iftIsValidFLIMGraph(graph);
    if (num_adj < 1)
        iftError("Number of neighbors must be positive", __func__);

    // cria a matriz de saída
    int out_num_feats = aggregation_opt == IFT_GFLIM_AGGREGATION_CONCAT ? num_adj * graph->num_feats : 2 * graph->num_feats;
    iftMatrix *matrix = iftCreateMatrix(out_num_feats, graph->num_nodes);

    for (int p = 0; p < graph->num_nodes; p++)
    {
        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(p, graph, num_adj, 1, true, attention_opt);

        /*if (attention_opt == IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL)
        {
            memcpy(neighbors_all[p]->val, neighbors->val, neighbors->n * sizeof(int));
            neighbors_all[p]->n = neighbors->n;
        }*/

        double *feature_vector = iftFLIMgraphAggregateFeatures(p, neighbors, graph, num_adj, aggregation_opt);

        for (int b = 0; b < out_num_feats; b++)
            iftMatrixElem(matrix, b, p) = (float)feature_vector[b];

        iftDestroyIntArray(&neighbors);
        free(feature_vector);
    }
    return matrix;
}

/**
 * @fn iftFeatureMatrixToFLIMGraph
 * @brief Create a graph from a matrix
 * @param matrix Matrix with the features
 * @param graph_ref Reference graph
 * @return iftFLIMGraph* Graph with the features
 */
iftFLIMGraph *iftFeatureMatrixToFLIMGraph(iftMatrix *matrix, iftFLIMGraph *graph_ref)
{
    if (matrix->nrows != graph_ref->num_nodes)
        iftError("Number of nodes in the graph is different from the number of rows in the matrix", __func__);
    iftIsValidFLIMGraph(graph_ref);

    iftFLIMGraph *graph = (iftFLIMGraph *)malloc(sizeof(iftFLIMGraph));
    graph->num_feats = matrix->ncols;
    graph->num_nodes = graph_ref->num_nodes;
    graph->nodes = (iftFLIMGraphNode *)calloc(graph->num_nodes, sizeof(iftFLIMGraphNode));

    iftFLIMGraphNode *node = graph->nodes;
    iftFLIMGraphNode *node_ref = graph_ref->nodes;
    for (int node_index = 0; node_index < graph->num_nodes; node_index++)
    {
        node->numNeighbors = node_ref->numNeighbors;
        node->numPixels = node_ref->numPixels;

        node->feats = (double *)malloc(matrix->ncols * sizeof(double));
        for (int b = 0; b < matrix->ncols; b++)
            node->feats[b] = iftMatrixElem(matrix, b, node_index);

        node->adj = iftSetCopy(node_ref->adj);
        node++;
        node_ref++;
    }
    iftIsValidFLIMGraph(graph);
    return graph;
}

/** @} */