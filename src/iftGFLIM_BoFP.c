
#include "iftGFLIM_BoFP.h"

// ============ AUXILIAR FUNCTIONS ============ //
/**
 * @fn GetVoxelCoord
 * @brief Convert an image index to a voxel coordinate.
 * @param xsize The xsize of the image.
 * @param ysize The ysize of the image.
 * @param zsize The zsize of the image.
 * @param p The image index.
 * @return The voxel coordinate.
 */
iftVoxel GetVoxelCoord(int xsize, int ysize, int zsize, int p)
{
    iftVoxel u;
    div_t res1 = div(p, xsize * ysize);
    div_t res2 = div(res1.rem, xsize);

    u.x = res2.rem;
    u.y = res2.quot;
    u.z = res1.quot;
    u.t = 0;
    return (u);
}

/**
 * @fn iftGFLIMNumberOfTruelabels
 * @brief Get the number of true labels in the truelabels array.
 * @param truelabels An array with the kernel labels.
 * @param n The number of elements in truelabels.
 * @return The maximum value in truelabels.
 */
int iftGFLIMNumberOfTruelabels(int *truelabels, int n)
{
    int nlabels = 0;
    for (int i = 0; i < n; i++)
        if (truelabels[i] > nlabels)
            nlabels = truelabels[i];
    return (nlabels);
}

// ============ FUNCTIONS TO VALIDATE TYPES ============ //

/**
 * @fn iftGFLIMGetStrideType
 * @brief Get the stride type from a string.
 * @param name The name of the stride type.
 * @return a iftGFLIMStrideType. @see iftGFLIMStrideType.
 */
iftGFLIMStrideType iftGFLIMGetStrideType(const char *name)
{
    if (strcmp(name, "none") == 0)
        return IFT_GFLIM_NOSTRIDE;
    if (strcmp(name, "image-based") == 0)
        return IFT_GFLIM_IMAGESTRIDE;
    if (strcmp(name, "graph-based") == 0)
        return IFT_GFLIM_GRAPHSTRIDE;
    iftError("Invalid stride type: %s. The options are none, image-based, and graph-based", __func__, name);
    return IFT_GFLIM_NOSTRIDE;
}

/**
 * @fn iftGLFLIMGetDecoderType
 * @brief Get the decoder type from a string.
 * @param name The name of the decoder type.
 * @return a iftGFLIMDecoderType. @see iftGFLIMDecoderType.
 */
iftGFLIMDecoderType iftGLFLIMGetDecoderType(const char *name)
{
    if (strcmp(name, "tristate") == 0)
        return IFT_GFLIM_TRISTATE_DECODER;
    if (strcmp(name, "hybrid") == 0)
        return IFT_GFLIM_HYBRID_DECODER;
    if (strcmp(name, "graph-adaptive") == 0)
        return IFT_GFLIM_GRAPHADAPTIVE_DECODER;
    if (strcmp(name, "gaussian") == 0)
        return IFT_GFLIM_GAUSSIAN_DECODER;
    if (strcmp(name, "test") == 0)
        return IFT_GFLIM_TEST_DECODER;
    if (strcmp(name, "mean") == 0)
        return IFT_GFLIM_MEAN_DECODER;
    iftError("Invalid decoder type: %s. The options are tristate, hybrid, graph-adaptive, and gaussian", __func__, name);
    return IFT_GFLIM_TRISTATE_DECODER;
}

/**
 * @fn iftGLFLIMGetDimReductionType
 * @brief Get the dimensionality reduction type from a string.
 * @param name The name of the dimensionality reduction type.
 * @return a iftGFLIMDimReductionType. @see iftGFLIMDimReductionType.
 */
iftGFLIMDimReductionType iftGLFLIMGetDimReductionType(const char *name)
{
    if (strcmp(name, "none") == 0)
        return IFT_GFLIM_DIMREDUCTION_NONE;
    if (strcmp(name, "tsne") == 0)
        return IFT_GFLIM_DIMREDUCTION_TSNE;
    if (strcmp(name, "pca") == 0)
        return IFT_GFLIM_DIMREDUCTION_PCA;
    iftError("Invalid dimensionality reduction type: %s. The options are none, tsne, and pca", __func__, name);
    return IFT_GFLIM_DIMREDUCTION_NONE;
}

// ============ FUNCTIONS TO READ AND SAVE MODEL FILES ============ //

/**
 * @fn iftGFLIMSaveBias
 * @brief Save bias values in a txt file.
 * @param basepath The base path of the file.
 * @param bias An array with the bias values.
 * @param number_of_kernels The number of kernels. Size of the bias array.
 * @note The bias values are saved in a txt file with the name basepath-bias.txt.
 */
void iftGFLIMSaveBias(char *basepath, float *bias, int number_of_kernels)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t number_of_kernels: %d\n", number_of_kernels);
#endif

    char filename[512];
    FILE *fp;

    sprintf(filename, "%s-bias.txt", basepath);
    fp = fopen(filename, "w");
    if (fp == NULL)
        iftError("Error opening file %s", __func__, filename);
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
        fprintf(fp, "%f ", bias[k]);
    fclose(fp);
}

/**
 * @fn iftGFLIMSaveKernelLabels
 * @brief Save the kernel labels in a txt file.
 * @param basepath The base path of the file.
 * @param truelabel An array with the kernel labels.
 * @param number_of_kernels The number of kernels. Size of the truelabel array.
 * @note The kernel labels are saved in a txt file with the name BASEPATH-labels.txt.
 * @note The kernel labels come from the labels of the graph markers.
 */
void iftGFLIMSaveKernelLabels(char *basepath, int *truelabel, int number_of_kernels)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
    printf("\t truelabel: %s\n", truelabel == NULL ? "NULL" : "not NULL");
    printf("\t number_of_kernels: %d\n", number_of_kernels);
#endif

    char filename[512];
    FILE *fp;

    sprintf(filename, "%s-labels.txt", basepath);
    fp = fopen(filename, "w");
    if (fp == NULL)
        iftError("Error opening file %s", __func__, filename);
    fprintf(fp, "%d\n", number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
        fprintf(fp, "%d ", truelabel[k]);

    fclose(fp);
}

/**
 * @fn iftGFLIMLoadBias
 * @brief Load the bias values from a txt file.
 * @param basepath The base path of the file.
 * @return An array with the bias values.
 * @note The bias values are loaded from a txt file with the name basepath-bias.txt.
 * @note The number of kernels is stored in the first line of the file.
 */
float *iftGFLIMLoadBias(char *basepath)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
#endif

    int number_of_kernels;
    char filename[512];
    FILE *fp;
    float *bias;

    sprintf(filename, "%s-bias.txt", basepath);
    if (!iftFileExists(filename))
        return NULL;
    fp = fopen(filename, "r");
    if (fp == NULL)
        iftError("Error opening file %s",  __func__, filename);
    if (fscanf(fp, "%d", &number_of_kernels) != 1)
        iftError("Error reading number of kernels from file %s", __func__, filename);
    bias = iftAllocFloatArray(number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        if (fscanf(fp, "%f ", &bias[k]) != 1)
            iftError("Error reading bias values from file %s",  __func__, filename);
    }
    fclose(fp);
    return bias;
}

/**
 * @fn iftGFLIMLoadKernelLabels
 * @brief Load the kernel labels from a txt file.
 * @param[in] basepath The base path of the file.
 * @param[out] nkernels Integer pointer to the number of kernels.
 * @return An array with the kernel labels.
 * @note The kernel labels are loaded from a txt file with the name basepath-labels.txt.
 * @note The number of kernels is stored in the first line of the file.
 * @note The kernel labels come from the labels of the graph markers.
 */
int *iftGFLIMLoadKernelLabels(char *filename, int *nkernels)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t filename: %s\n", filename == NULL ? "NULL" : filename);
    printf("\t nkernels: %s\n", nkernels == NULL ? "NULL" : "not NULL");
#endif

    int number_of_kernels;
    FILE *fp;
    int *truelabels;

    fp = fopen(filename, "r");
    if (fp == NULL)
        iftError("Error opening file %s",  __func__, filename);
    if (fscanf(fp, "%d", &number_of_kernels) != 1)
        iftError("Error reading number of kernels from file %s", __func__, filename);
    truelabels = iftAllocIntArray(number_of_kernels);
    for (int k = 0; k < number_of_kernels; k++)
    {
        if (fscanf(fp, "%d ", &truelabels[k]) != 1)
            iftError("Error reading kernel labels from file %s", __func__, filename);
    }
    fclose(fp);
    (*nkernels) = number_of_kernels;
    return truelabels;
}

/**
 * @fn iftGFLIMSaveMergedModel
 * @brief Save the kernels, bias and labels of a layer from the merged model.
 * @param basepath The base path of the file.
 * @param K The matrix with the kernels.
 * @param bias An array with the bias values.
 * @param truelabels An array with the kernel labels.
 * @note The kernels are saved in a npy file with the name basepath-kernels.npy.
 * @note The bias values are saved in a txt file with the name basepath-bias.txt.
 * @note The kernel labels are saved in a txt file with the name basepath-labels.txt.
 * @note The number of kernels is stored in the first line of the bias and labels files.
 * @note The kernel labels come from the labels of the graph markers.
 */
void iftGFLIMSaveMergedModel(char *basepath, iftMatrix *K, float *bias, int *truelabels)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
    printf("\t K: %s\n", K == NULL ? "NULL" : "not NULL");
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t truelabels: %s\n", truelabels == NULL ? "NULL" : "not NULL");
#endif

    char filename[512];
    sprintf(filename, "%s-kernels.npy", basepath);
    iftWriteMatrix(K, filename);
    iftGFLIMSaveBias(basepath, bias, K->ncols);
    iftGFLIMSaveKernelLabels(basepath, truelabels, K->ncols);
}

/**
 * @fn iftGFLIMLoadMergedModel
 * @brief Load the kernels, bias and labels of a layer from the merged model.
 * @param basepath The base path of the file.
 * @param K The kernels matrix.
 * @param bias An array with the bias values.
 * @param truelabels An array with the kernel labels.
 * @note The kernels are loaded from a npy file with the name basepath-kernels.npy.
 * @note The bias values are loaded from a txt file with the name basepath-bias.txt.
 * @note The kernel labels are loaded from a txt file with the name basepath-labels.txt.
 * @note The number of kernels is stored in the first line of the bias and labels files.
 * @note The kernel labels come from the labels of the graph markers.
 */
void iftGFLIMLoadMergedModel(char *basepath, iftMatrix **K, float **bias)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t basepath: %s\n", basepath == NULL ? "NULL" : basepath);
    printf("\t K: %s\n", K == NULL ? "NULL" : "not NULL");
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
#endif

    char filename[512];
    sprintf(filename, "%s-kernels.npy", basepath);
    *K = iftReadMatrix(filename);
    *bias = iftGFLIMLoadBias(basepath);
    if (*bias == NULL)
        iftError("Bias is NULL", __func__);
}

// ============ CREATE, CONVERT, AND UPDATE GRAPH-BASED MARKERS ============ //

/**
 * @fn iftGFLIMImageMarkersToGraphMarkers
 * @brief Transform the image markers into graph markers.
 * @param label_img The image with the superpixels. It maps from image pixels to graph nodes.
 * @param S The set of image markers.
 * @param stride The stride to map the image markers to the graph markers.
 * @param img_xsize The xsize of the image.
 * @param img_ysize The ysize of the image.
 * @return The set of graph markers.
 */
iftLabeledSet *iftGFLIMImageMarkersToGraphMarkers(iftImage *label_img,
                                                  iftLabeledSet *S,
                                                  iftGFLIMStrideType stride,
                                                  int img_xsize, int img_ysize)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    if (S == NULL)
        printf("\t S: NULL \n");
    else
        printf("\t S size: %d \n", iftLabeledSetSize(S));
    switch (stride)
    {
    case IFT_GFLIM_NOSTRIDE:
        printf("\t stride: IFT_GFLIM_NOSTRIDE \n");
        break;

    case IFT_GFLIM_IMAGESTRIDE:
        printf("\t stride: IFT_GFLIM_IMAGESTRIDE \n");
        break;

    case IFT_GFLIM_GRAPHSTRIDE:
        printf("\t stride: IFT_GFLIM_GRAPHSTRIDE \n");
        break;
    }
    printf("\t img_xsize: %d\n", img_xsize);
    printf("\t img_ysize: %d\n", img_ysize);
#endif

    iftLabeledSet *graph_seeds = NULL;
    int nclasses = 0, minclass = IFT_INFINITY_INT;
    iftLabeledSet *tmp;
    float scale[] = {1.0, 1.0, 1.0};

    // compute number of nodes
    int num_nodes = iftMaximumValue(label_img);
    if (num_nodes < 1)
        iftError("Number of nodes must be positive: %d", __func__, num_nodes);

    // find number of classes and minimum class value
    tmp = S;
    while (tmp != NULL)
    {
        if (tmp->label > nclasses)
            nclasses = tmp->label;
        if (tmp->label < minclass)
            minclass = tmp->label;
        tmp = tmp->next;
    }
    if (minclass == 0)
        nclasses++;

    // compute scale if its necessary
    if (stride == IFT_GFLIM_IMAGESTRIDE)
    {
        scale[0] = (float)img_xsize / (float)label_img->xsize;
        scale[1] = (float)img_ysize / (float)label_img->ysize;
    }

    // count the number of pixels per graph node and per class (label)
    int **classes_count = iftAlloc(num_nodes, sizeof(int *));
    for (int i = 0; i < num_nodes; i++)
        classes_count[i] = iftAlloc(nclasses, sizeof(int *));
    tmp = S;

    while (tmp != NULL)
    {
        int p = tmp->elem;
        iftVoxel u = GetVoxelCoord((int)(label_img->xsize * scale[0]),
                                   (int)(label_img->ysize * scale[1]), 1, p);
        u.x = u.x / scale[0];
        u.y = u.y / scale[1];
        u.z = 1.0;

        p = u.y * label_img->xsize + u.x;

        classes_count[label_img->val[p] - 1][tmp->label - minclass]++;
        tmp = tmp->next;
    }

    // insert the graph seeds in the graph_seeds set based on the class with the highest count
    for (int p = 0; p < num_nodes; p++)
    {
        int max_class = 0;
        int max_count = classes_count[p][0];
        for (int c = 1; c < nclasses; c++)
        {
            if (classes_count[p][c] > classes_count[p][max_class])
            {
                max_class = c;
                max_count = classes_count[p][c];
            }
        }
        if (max_count > 0)
            iftInsertLabeledSetMarkerAndHandicap(&graph_seeds, p, max_class, max_class, 0);
    }

    for (int i = 0; i < num_nodes; i++)
        iftFree(classes_count[i]);
    iftFree(classes_count);

    return graph_seeds;
}

/**
 * @fn iftGFLIMUpdateGraphMarkers
 * @brief Update graph markers considering the superpixel image size of the network's input and the current layer's input.
 * @param S The set of graph markers.
 * @param label_img The image with the superpixels. It maps the image pixels to the graph nodes (from 1).
 * @param label_img0 The image with the superpixels in "layer 0" (network's input). It maps the image pixels to the graph nodes (from 1).
 * @return The set of updated graph markers.
 */
void iftGFLIMUpdateGraphMarkers(iftLabeledSet *S, iftImage *label_img, iftImage *label_img0)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (S == NULL)
        printf("\t S: NULL \n");
    else
        printf("\t S size: %d \n", iftLabeledSetSize(S));
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    if (label_img0 == NULL)
        printf("\t label_img0: NULL \n");
    else
        printf("\t label_img0 size: %d %d %d \n", label_img0->xsize, label_img0->ysize, label_img0->zsize);
#endif
    int num_nodes = iftMaximumValue(label_img);
    int num_nodes0 = iftMaximumValue(label_img0);

    // compute scale
    int scale[3];
    scale[0] = (float)label_img0->xsize / (float)label_img->xsize;
    scale[1] = (float)label_img0->ysize / (float)label_img->ysize;
    scale[2] = (float)label_img0->zsize / (float)label_img->zsize;

    // find number of classes and minimum class value
    int nclasses = 0, minclass = INT_MAX;
    iftLabeledSet *tmp = S;
    while (tmp != NULL)
    {
        if (tmp->label > nclasses)
            nclasses = tmp->label;
        if (tmp->label < minclass)
            minclass = tmp->label;
        tmp = tmp->next;
    }
    if (minclass == 0)
        nclasses++;

    // map nodes from label_img0 to label_img (nodes_map) and compute the number of pixels per node (spxSize)
    int *nodes_map = iftAllocIntArray(num_nodes0); // num_nodes0 is always greater than or equal to num_nodes
    for (int i = 0; i < label_img0->n; i++)
    {
        if (nodes_map[label_img0->val[i] - 1] != 0)
            continue;

        iftVoxel u;
        int p = i;
        if (iftIs3DImage(label_img))
        {
            u = GetVoxelCoord((int)(label_img->xsize * scale[0]),
                              (int)(label_img->ysize * scale[1]),
                              (int)(label_img->zsize * scale[2]), p);
        }
        else
        {
            u = GetVoxelCoord((int)(label_img->xsize * scale[0]),
                              (int)(label_img->ysize * scale[1]), 1, p);
        }
        u.x = u.x / scale[0];
        u.y = u.y / scale[1];
        u.z = u.z / scale[2];

        p = iftGetVoxelIndex(label_img, u);
        nodes_map[label_img0->val[i] - 1] = label_img->val[p];
    }

    int **class_count = iftAlloc(num_nodes, sizeof(int *)); // pixels per class at each node
    for (int i = 0; i < num_nodes; i++)
        class_count[i] = iftAlloc(nclasses, sizeof(int));

    // compute the number of nodes per class at each node
    tmp = S;
    while (tmp != NULL)
    {
        int node = nodes_map[tmp->elem] - 1;
        class_count[node][tmp->label - minclass]++;
        tmp = tmp->next;
    }
    
    tmp = S;
    iftBMap *visited = iftCreateBMap(num_nodes); // to avoid inserting the same node multiple times
    while (tmp != NULL)
    {
        int node0 = tmp->elem;
        int node = nodes_map[node0] - 1; // map node0 to node
        
        if(iftBMapValue(visited, node) == 0)
        {
            iftBMapSet1(visited, node); // mark node as visited

            int max_class = 0;
            int max_count = class_count[node][0];
            for (int c = 1; c < nclasses; c++)
            {
                if (class_count[node][c] > max_count)
                {
                    max_class = c;
                    max_count = class_count[node][c];
                }
            }
            if (max_count > 0){
                tmp->elem = node;
                tmp->marker = max_class + minclass;
                tmp->label = max_class + minclass;
            }
            else
                iftError("Node %d has no class assigned", __func__, node0);
        
            tmp = tmp->next;
        }else{
            tmp = tmp->next;
            iftRemoveLabeledSetElem(&S, node0); // remove the node if it was already visited
        }
    }
    free(nodes_map);
    iftDestroyBMap(&visited);
    
    /*
    iftLabeledSet *T = NULL;
    for (int p = 0; p < num_nodes; p++)
    {
        int max_class = 0;
        int max_count = class_count[p][0];
        for (int c = 1; c < nclasses; c++)
        {
            if (class_count[p][c] > max_count)
            {
                max_class = c;
                max_count = class_count[p][c];
            }
        }
        if (max_count > 0)
            iftInsertLabeledSetMarkerAndHandicap(&T, p, max_class + minclass, max_class + minclass, 0);
    }*/


    for (int i = 0; i < num_nodes; i++)
        iftFree(class_count[i]);
    iftFree(class_count);
    
    //    return T;
}

// ============ CONVERT FROM SEEDS/GRAPH TO DATASET ============ //

/**
 * @fn iftGFLIMSelectClusterCenters
 * @brief Selects the cluster centers from the seeds in the graph.
 * @param graph The input graph where the seeds are defined.
 * @param S The set of graph seeds.
 * @param kernel_size The kernel size for the feature extraction.
 * @param nclusters_per_marker The number of cluster centers per marker.
 * @param attention_opt The attention option. @see iftGFLIMGraphAttention.
 * @param dimReduction The dimensionality reduction option. @see iftGFLIMDimReductionType.
 * @param ncomponents The number of components/dimensions for the dimensionality reduction.
 * @param aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @return The set of cluster centers.
 * @note The function assign a value to each connected set of labeled nodes in the marker field of the iftLabeledSet S.
 */
iftLabeledSet *iftGFLIMSelectClusterCenters(iftGFLIMGraph *graph,
                                            iftLabeledSet *S,
                                            int kernel_size,
                                            int nclusters_per_marker,
                                            iftGFLIMGraphAttention attention_opt,
                                            iftGFLIMDimReductionType dimReduction, int ncomponents,
                                            iftGFLIMAggregation aggregation_opt)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (S == NULL)
        printf("\t S: NULL \n");
    else
        printf("\t S size: %d \n", iftLabeledSetSize(S));
    printf("\t kernel_size: %d\n", kernel_size);
    printf("\t nclusters_per_marker: %d\n", nclusters_per_marker);
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t dimReduction: %d\n", dimReduction);
    printf("\t ncomponents: %d\n", ncomponents);
    printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

    iftLabeledSet *T = NULL;
    int obj_label_map[graph->num_nodes];
    iftGFLIMRelabelConnectedComponentsInPlace(graph, S, obj_label_map);

    // extract neighbor information and compute the iftDataSet
    iftDataSet *Z = iftGFLIMExtractMarkerDataSet(graph, S, kernel_size, attention_opt, aggregation_opt);

    // not using it
    if (dimReduction == IFT_GFLIM_DIMREDUCTION_PCA)
    {
        if (ncomponents < 1)
            iftError("Invalid number of components for PCA: %d. The minimum is 1", __func__, ncomponents);
        else
            ncomponents = iftMin(ncomponents, Z->nfeats);
    }

    // not using it
    if (dimReduction == IFT_GFLIM_DIMREDUCTION_TSNE && (ncomponents < 2 || ncomponents > 3))
        iftError("Invalid number of components for tSNE: %d. The options are 2 or 3", __func__, ncomponents);

    /* Estimate the given number of cluster centers per marker */
    for (int c = 1; c <= Z->nclasses; c++)
    {
        iftDataSet *Z1, *Z2;

        switch (dimReduction)
        {
        case IFT_GFLIM_DIMREDUCTION_NONE:
            Z1 = iftExtractSamplesFromClass(Z, c);
            // iftNormalizeDataSetByZScoreInPlace(Z1, NULL, 0.01);
            break;

        case IFT_GFLIM_DIMREDUCTION_PCA:
            Z2 = iftExtractSamplesFromClass(Z, c);
            iftSetStatus(Z2, IFT_TRAIN);

            // Step 1. Standardize the data to ensure that all variables have a mean of 0 and a standard deviation of 1.
            iftCentralizeDataSetInPlace(Z2);
            // iftNormalizeDataSetByZScoreInPlace(Z2, NULL, 0.01);

            // Step 2. Calculate the covariance matrix of the standardized data. This matrix shows how each variable is related to every other variable in the dataset.
            // Step 3. Calculate the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors represent the directions in which the data varies the most, while the eigenvalues represent the amount of variation along each eigenvector.
            // Step 4. Choose the principal components which are the eigenvectors with the highest eigenvalues. These components represent the directions in which the data varies the most.
            // Step 5. Transform the original data into the lower-dimensional space defined by the principal components.
            Z1 = iftTransFeatSpaceByPCA(Z2, ncomponents);
            iftDestroyDataSet(&Z2);
            break;

        case IFT_GFLIM_DIMREDUCTION_TSNE:
            Z2 = iftExtractSamplesFromClass(Z, c);
            iftSetStatus(Z2, IFT_TRAIN);
            // iftNormalizeDataSetByZScoreInPlace(Z2, NULL, 0.01);
            iftCentralizeDataSetInPlace(Z2);
            Z1 = iftDimReductionByTSNE(Z2, ncomponents, iftMin(30, (Z2->nsamples - 1) / 3), 1000);
            iftDestroyDataSet(&Z2);
            break;

        default:
            iftError("Invalid dimensionality reduction %d \n", __func__, dimReduction);
            break;
        }

        int ngroups = iftMin(Z1->nsamples, nclusters_per_marker);
        iftKMeans(Z1, ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
        for (int s = 0; s < Z1->nsamples; s++)
        {
            if (iftHasSampleStatus(Z1->sample[s], IFT_PROTOTYPE))
            {
                int p = Z1->sample[s].id;
                iftInsertLabeledSetMarkerAndHandicap(&T, p,
                                                     obj_label_map[p],
                                                     c,
                                                     0);
            }
        }
        iftDestroyDataSet(&Z1);
    }
    iftDestroyDataSet(&Z);
    return (T);
}

/**
 * @fn iftGFLIMExtractMarkerDataSet
 * @brief Computes a feature matrix from the nodes in a seed set S.
 * @param graph The input graph.
 * @param S The set of graph seeds.
 * @param kernel_size The kernel size for the feature extraction.
 * @param most_similar If true, the most similar neighbors are selected. Otherwise, the most dissimilar neighbors are selected.
 * @param attention_opt The attention option to compute the features. @see iftGFLIMGraphAttention.
 * @param aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @return The dataset with the features extracted from the graph nodes.
 */
iftDataSet *iftGFLIMExtractMarkerDataSet(iftGFLIMGraph *graph,
                                         iftLabeledSet *S,
                                         int kernel_size,
                                         iftGFLIMGraphAttention attention_opt,
                                         iftGFLIMAggregation aggregation_opt)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (S == NULL)
        printf("\t S: NULL \n");
    else
        printf("\t S size: %d \n", iftLabeledSetSize(S));
    printf("\t kernel_size: %d\n", kernel_size);
    printf("\t most_similar: %s\n", most_similar ? "true" : "false");
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

    int nsamples = iftLabeledSetSize(S);
    int kernel_neighbors = iftMin(graph->num_nodes, kernel_size);
    int tensor_size = aggregation_opt == IFT_GFLIM_AGGREGATION_CONCAT ? graph->num_feats * kernel_neighbors : 2 * graph->num_feats;
    iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);

    iftLabeledSet *seed;
    Z->nclasses = 0;
    int s = 0;
    seed = S;

    while (seed != NULL)
    {
        Z->sample[s].id = seed->elem;
        Z->sample[s].truelabel = seed->marker;
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        iftIntArray *neighbors = iftFLIMGraphGetKNeighborsAggregation(seed->elem, graph, kernel_neighbors, 1, true, attention_opt);
        double *feature_vector = iftGFLIMGraphAggregateFeatures(seed->elem, neighbors, graph, kernel_size, aggregation_opt);

        for (int i = 0; i < tensor_size; i++)
            Z->sample[s].feat[i] = (float)feature_vector[i];

        iftFree(feature_vector);
        iftDestroyIntArray(&neighbors);

        s++;
        seed = seed->next;
    }

    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);
    return (Z);
}

// ============ FUNCTIONS TO CREATE A LAYER ============ //

/**
 * @fn iftGFLIMComputeKernelsAndTrueLabels
 * @brief Compute the kernels and the true labels from the dataset.
 * @param Z The dataset with graph markers.
 * @param truelabel The true labels.
 * @return The matrix with the kernels.
 */
iftMatrix *iftGFLIMComputeKernelsAndTrueLabels(iftDataSet *Z, int *truelabel)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (Z == NULL)
        printf("\t Z: NULL \n");
    else
        printf("\t Z samples: %d, feats: %d, classes: %d \n", Z->nsamples, Z->nfeats, Z->nclasses);
    printf("\t truelabel: %s\n", truelabel == NULL ? "NULL" : "not NULL");
#endif
    iftMatrix *kernels = iftCreateMatrix(Z->nsamples, Z->nfeats);
    /* compute filters and their truelabels */
    for (int s = 0, col = 0; s < Z->nsamples; s++)
    {
        iftUnitNorm(Z->sample[s].feat, Z->nfeats);
        truelabel[s] = Z->sample[s].truelabel;
        for (int row = 0; row < Z->nfeats; row++)
            iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row] / Z->fsp.stdev[row];
        col++;
    }
    return kernels;
}

/**
 * @fn iftGFLIMComputeBias
 * @brief Compute the bias from the kernels and the dataset.
 * @param kernels The matrix with the kernels.
 * @param Z The dataset with graph markers.
 * @return The bias.
 */
float *iftGFLIMComputeBias(iftMatrix *kernels, iftDataSet *Z)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (kernels == NULL)
        printf("\t kernels: NULL \n");
    else
        printf("\t kernels size: %d %d \n", kernels->nrows, kernels->ncols);
    if (Z == NULL)
        printf("\t Z: NULL \n");
    else
        printf("\t Z samples: %d, feats: %d, classes: %d \n", Z->nsamples, Z->nfeats, Z->nclasses);
#endif

    float *bias = NULL;
    bias = iftAllocFloatArray(kernels->ncols);
    for (int col = 0; col < kernels->ncols; col++)
    {
        for (int row = 0; row < Z->nfeats; row++)
            bias[col] -= (Z->fsp.mean[row] * iftMatrixElem(kernels, col, row));
    }
    return bias;
}

/**
 * @fn iftGFLIMCreateLayerModel
 * @brief Create a layer model.
 * @param graph The input graph.
 * @param S The set of graph markers.
 * @param label_img0 The image-graph mapping in "layer 0" (from 1). It can be null if stride is IFT_GFLIM_NOSTRIDE.
 * @param label_img The image-graph mapping for the input graph (from 1). It can be null if stride is IFT_GFLIM_NOSTRIDE.
 * @param layer The layer configuration.
 * @param stdev_factor The standard deviation factor.
 * @param stride The stride to map the image markers to the graph markers.
 * @param[out] bias A pointer to an array. After the function, it will be filled with the bias values. Is allocated inside the function.
 * @param[out] truelabel A pointer to an array. After the function, it will be filled with the bias values. Is allocated inside the function.
 * @param attention_opt The attention option. @see iftGFLIMGraphAttention.
 * @param aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @note The function allocates the bias and truelabel arrays. The caller is responsible for freeing them.
 * @return The matrix with the kernels.
 */
iftMatrix *iftGFLIMCreateLayerModel(iftGFLIMGraph *graph, iftLabeledSet *S,
                                    iftImage *label_img0, iftImage *label_img,
                                    iftGFLIMLayer layer, float stdev_factor,
                                    iftGFLIMStrideType stride,
                                    float **bias, int **truelabel,
                                    iftGFLIMGraphAttention attention_opt,
                                    iftGFLIMAggregation aggregation_opt)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (S == NULL)
        printf("\t S: NULL \n");
    else
        printf("\t S size: %d \n", iftLabeledSetSize(S));
    if (label_img0 == NULL)
        printf("\t label_img0: NULL \n");
    else
        printf("\t label_img0 size: %d %d %d \n", label_img0->xsize, label_img0->ysize, label_img0->zsize);
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    printf("\t layer: kernel_size: %d, nclusters_per_bkg_marker: %d, nclusters_per_obj_marker: %d \n", layer.kernel_size, layer.nkernels_per_marker, layer.nkernels_per_marker);
    printf("\t stdev_factor: %f\n", stdev_factor);
    switch (stride)
    {
    case IFT_GFLIM_NOSTRIDE:
        printf("\t stride: IFT_GFLIM_NOSTRIDE \n");
        break;

    case IFT_GFLIM_IMAGESTRIDE:
        printf("\t stride: IFT_GFLIM_IMAGESTRIDE \n");
        break;

    case IFT_GFLIM_GRAPHSTRIDE:
        printf("\t stride: IFT_GFLIM_GRAPHSTRIDE \n");
        break;
    }
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t truelabel: %s\n", truelabel == NULL ? "NULL" : "not NULL");
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

    switch (stride)
    {
    case IFT_GFLIM_GRAPHSTRIDE:
        if (graph->num_nodes != iftMaximumValue(label_img0))
            iftGFLIMUpdateGraphMarkers(S, label_img, label_img0);
        break;

    case IFT_GFLIM_IMAGESTRIDE:
        if (label_img->xsize != label_img0->xsize || label_img->ysize != label_img0->ysize)
            iftGFLIMUpdateGraphMarkers(S, label_img, label_img0);
        break;

    case IFT_GFLIM_NOSTRIDE:
        break;

    default:
        iftError("Invalid stride %d", __func__, stride);
        break;
    }

    iftDataSet *Z = iftGFLIMExtractLabelDataSet(graph, S, layer.kernel_size, attention_opt, aggregation_opt);

#ifdef PRINT_VALS
    printf("Dataset %s\n", basename);
    for (int i = 0; i < Z->nsamples; i++)
    {
        printf("Sample %d: ", i);
        for (int j = 0; j < Z->nfeats; j++)
            printf("%f ", Z->sample[i].feat[j]);
        printf("\n");
    }
    printf("\n");
#endif

    iftNormalizeDataSetByZScoreInPlace(Z, NULL, stdev_factor);

#ifdef PRINT_VALS
    printf("Z-norm mean: ");
    for (int i = 0; i < Z->nfeats; i++)
        printf("%f ", Z->fsp.mean[i]);
    printf("\n");
    printf("Z-norm stdev: ");
    for (int i = 0; i < Z->nfeats; i++)
        printf("%f ", Z->fsp.stdev[i]);
    printf("\n");

    printf("Kernels after z-norm: \n");
    for (int i = 0; i < Z->nsamples; i++)
    {
        for (int j = 0; j < Z->nfeats; j++)
            printf("%f ", Z->sample[i].feat[j]);
        printf("\n");
    }
    printf("\n");
#endif

    (*truelabel) = iftAllocIntArray(Z->nsamples);
    iftMatrix *kernels = iftGFLIMComputeKernelsAndTrueLabels(Z, *truelabel);

#ifdef PRINT_VALS
    printf("Kernels after unit-norm: \n");
    for (int i = 0; i < kernels->nrows; i++)
    {
        for (int j = 0; j < kernels->ncols; j++)
            printf("%f ", iftMatrixElem(kernels, i, j));
        printf("\n");
    }
    printf("\n");
#endif

    /* compute biases and update filters */
    (*bias) = iftGFLIMComputeBias(kernels, Z);

#ifdef PRINT_VALS
    printf("Kernels after bias: \n");
    for (int i = 0; i < kernels->nrows; i++)
    {
        for (int j = 0; j < kernels->ncols; j++)
            printf("%f ", iftMatrixElem(kernels, i, j));
        printf("\n");
    }
    printf("\n");

    printf("Bias: \n");
    for (int i = 0; i < kernels->ncols; i++)
        printf("%f ", (*bias)[i]);
    printf("\n");
#endif

    iftDestroyDataSet(&Z);
    /*if (T != S) iftDestroyLabeledSet(&T);
    else T = NULL;*/
    return kernels;
}

// ============ FUNCTIONS FOR UNION FIND ALGORITHM IN GRAPH-BASED STRIDE ============ //

int element_link(int x, int y, int *rank, int *parent)
{
    // If necessary, rename variables to ensure that
    // x has rank at least as large as that of y
    if (rank[x] < rank[y])
    {
        int t = x;
        x = y;
        y = t;
    }

    // If necessary, increment the rank of x
    if (rank[x] == rank[y])
        rank[x] = rank[x] + 1;

    // Make x the new root
    parent[y] = x;
    return x;
}

int element_find(int x, int *parent)
{
    if (parent[x] != x)
        parent[x] = element_find(parent[x], parent);
    return parent[x];
}

int union_set(int x, int y, int *parent, int *rank)
{
    // Replace nodes by roots
    x = element_find(x, parent);
    y = element_find(y, parent);

    if (x == y)
    {
        return -1; // x and y are already in the same set
    }

    return element_link(x, y, rank, parent);
}

// ============ FUNCTIONS FOR STRIDE ============ //

/**
 * @fn iftGFLIMImageStride
 * @brief Compute the image-based stride
 * @param graph The input graph.
 * @param input_labels The input labels image.
 * @param output_labels The output labels image.
 * @param stride The stride value.
 * @return The output graph after stride.
 */
iftGFLIMGraph *iftGFLIMImageStride(iftGFLIMGraph *graph,
                                  iftImage *input_labels,
                                  iftImage *output_labels,
                                  int stride)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (input_labels == NULL)
        printf("\t input_labels: NULL \n");
    else
        printf("\t input_labels size: %d %d %d \n", input_labels->xsize, input_labels->ysize, input_labels->zsize);
    if (output_labels == NULL)
        printf("\t output_labels: NULL \n");
    else
        printf("\t output_labels size: %d %d %d \n", output_labels->xsize, output_labels->ysize, output_labels->zsize);
    printf("\t stride: %d\n", stride);
#endif

    iftBMap *visited = iftCreateBMap(graph->num_nodes);
    iftAdjRel *A;

    // reduce labels image according to stride and compute mean color
    int num_nodes = 0;
    iftVoxel u;
    for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
    {
        for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
        {
            for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
            {
                int p = iftMGetVoxelIndex(input_labels, u);
                int label = input_labels->val[p];
                if (!iftBMapValue(visited, label - 1))
                {
                    iftBMapSet1(visited, label - 1);
                    num_nodes++;
                }
            }
        }
    }

    int map_nodes[graph->num_nodes]; // map old to new node index
    int new_index = 0;
    for (int old_index = 0; old_index < graph->num_nodes; old_index++)
    {
        if (iftBMapValue(visited, old_index))
        {
            map_nodes[old_index] = new_index;
            new_index++;
        }
        else
            map_nodes[old_index] = -1;
    }

    if (new_index != num_nodes)
        iftError("Number of nodes is different from expected: %d %d", __func__, new_index, num_nodes);

    iftDestroyBMap(&visited);

    iftGFLIMGraph *output_graph = iftCreateGFLIMGraph(graph->num_feats, num_nodes);
    iftBMap *neighbors[num_nodes];

    if (iftIs3DImage(input_labels))
        A = iftSpheric(sqrtf(3.0));
    else
        A = iftCircular(sqrtf(2.0));

    for (int i = 0; i < num_nodes; i++)
        neighbors[i] = iftCreateBMap(num_nodes);

    int q = 0;
    for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
    {
        for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
        {
            for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
            {
                int p = iftMGetVoxelIndex(input_labels, u);
                int old_label = input_labels->val[p] - 1;
                int new_label = map_nodes[old_label];
                iftGFLIMGraphNode *node = &(output_graph->nodes[new_label]);

                output_labels->val[q] = new_label + 1;
                int feat = 0;
                while (feat < graph->num_feats)
                {
                    node->feats[feat] += graph->nodes[old_label].feats[feat];
                    feat++;
                }

                node->numPixels++;
                q++;
            }
        }
    }

    q = 0;
    for (u.z = 0; u.z < input_labels->zsize; u.z = u.z + stride)
    {
        for (u.y = 0; u.y < input_labels->ysize; u.y = u.y + stride)
        {
            for (u.x = 0; u.x < input_labels->xsize; u.x = u.x + stride)
            {
                int p = iftMGetVoxelIndex(input_labels, u);
                int old_label = input_labels->val[p] - 1;
                int new_label = map_nodes[old_label];
                iftGFLIMGraphNode *node = &(output_graph->nodes[new_label]);

                iftVoxel p_voxel = iftGetVoxelCoord(output_labels, q);
                for (int i = 1; i < A->n; i++)
                {
                    iftVoxel q_voxel = iftGetAdjacentVoxel(A, p_voxel, i);
                    if (iftValidVoxel(output_labels, q_voxel))
                    {
                        int q_index = iftMGetVoxelIndex(output_labels, q_voxel);
                        int q_label = output_labels->val[q_index] - 1;
                        if (new_label > q_label && !iftBMapValue(neighbors[new_label], q_label))
                        {
                            iftGFLIMGraphInsertNeighborNode(node, q_label);
                            iftGFLIMGraphInsertNeighborNode(&(output_graph->nodes[q_label]), new_label);
                            iftBMapSet1(neighbors[new_label], q_label);
                            output_graph->num_edges++;
                        }
                    }
                }
                q++;
            }
        }
    }

    //  for each node, compute its mean color
    iftGFLIMGraphNode *node = output_graph->nodes;
    for (int spx_index = 0; spx_index < num_nodes; spx_index++)
    {
        for (int band_index = 0; band_index < output_graph->num_feats; band_index++)
            node->feats[band_index] /= (double)node->numPixels;
        node++;
        iftDestroyBMap(&neighbors[spx_index]);
    }
    node = NULL;
    iftDestroyAdjRel(&A);

    return output_graph;
}

/**
 * @fn iftGFLIMGraphStride
 * @brief Compute the graph-based stride
 * @param graph The input graph.
 * @param stride The stride value.
 * @param nodes_map Empty array to map from nodes in one layer to another.
 * @return The graph after stride.
 * @note nodes_map must be allocated with graph->num_nodes elements.
 * @note The function will fill nodes_map with the mapping from the input graph to the output graph.
 */
iftGFLIMGraph *iftGFLIMGraphStride(iftGFLIMGraph *graph, float stride, int *nodes_map)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    printf("\t stride: %f\n", stride);
    printf("\t nodes_map: %s\n", nodes_map == NULL ? "NULL" : "not NULL");
#endif

    int num_nodes_out = floorf((float)graph->num_nodes / stride);
    double *distance = iftAllocDoubleArray(graph->num_nodes);
    int closest_neigh[graph->num_nodes];
    iftDHeap *heap = iftCreateDHeap(graph->num_nodes, distance);

    int rank[graph->num_nodes];
    int parent[graph->num_nodes];

    // weight nodes accoding to the minimum distance to their neighbors
    for (int i = 0; i < graph->num_nodes; i++)
    {
        iftGFLIMGraphNode *node = &(graph->nodes[i]);
        iftSet *adj = node->adj;

        double min_dist = INFINITY;
        int cn = -1;
        while (adj != NULL)
        {
            double val = 0;
            for (int j = 0; j < graph->num_feats; j++)
            { // euclidean distance
                val += (node->feats[j] - graph->nodes[adj->elem].feats[j]) * (node->feats[j] - graph->nodes[adj->elem].feats[j]);
            }
            val = sqrt(val);
            if (val < min_dist)
            {
                min_dist = val;
                cn = adj->elem;
            }
            adj = adj->next;
        }

        closest_neigh[i] = cn;
        distance[i] = min_dist;
        iftInsertDHeap(heap, i);
        rank[i] = 1;
        parent[i] = i;
        nodes_map[i] = -1;
    }
    
    // merge nodes until achieve the required number of nodes
    int removed_nodes = 0;
    while (!iftEmptyDHeap(heap) && removed_nodes != graph->num_nodes - num_nodes_out)
    {
        int x = iftRemoveDHeap(heap);
        int y = closest_neigh[x];

        int p = union_set(x, y, parent, rank);
        if (p == -1)
            continue;
        removed_nodes++;
    }
    iftFree(distance);
    iftDestroyDHeap(&heap);


    if (removed_nodes != graph->num_nodes - num_nodes_out)
    {
        iftWarning("The number of output nodes was changed from %d to %d", __func__, num_nodes_out, graph->num_nodes - removed_nodes);
        num_nodes_out = graph->num_nodes - removed_nodes;
    }

    iftGFLIMGraph *output_graph = iftCreateGFLIMGraph(graph->num_feats, num_nodes_out);
    int j = 0;
    for (int i = 0; i < graph->num_nodes; i++)
    {
        int p = element_find(i, parent);
        if (nodes_map[p] == -1)
        {
            nodes_map[p] = j;
            j++;
        }
        int q = nodes_map[p];
        nodes_map[i] = q;

        iftGFLIMGraphNode *node = &(output_graph->nodes[q]);
        iftGFLIMGraphNode *remove_node = &(graph->nodes[i]);

        // merge node features
        for (int b = 0; b < graph->num_feats; b++)
        {
            node->feats[b] *= node->numPixels;
            node->feats[b] += (remove_node->feats[b] * remove_node->numPixels);
            node->feats[b] /= (node->numPixels + remove_node->numPixels);
        }

        // merge their number of pixels
        node->numPixels += remove_node->numPixels;

        // merge neighbors
        iftSet *adj = remove_node->adj;
        while (adj != NULL)
        {
            int adj_label = nodes_map[adj->elem];
            if (adj->elem < i &&
                adj_label != q &&
                (iftUnionSetElem(&(node->adj), adj_label) == 1))
            {
                node->numNeighbors++;
                iftGFLIMGraphInsertNeighborNode(&(output_graph->nodes[adj_label]), q);
            }
            adj = adj->next;
        }
    }

    return output_graph;
}

/**
 * @fn iftGFLIMStrideBatch
 * @brief Perform stride in a batch of graphs.
 * @param[in,out] graph The input graph array (the batch). After the function, it will be filled with the output graphs.
 * @param[in] layer The layer configuration.
 * @param[in] in_labels_dir The input labels directory.
 * @param[in] out_labels_dir The output labels directory.
 * @param[in] basename The name of the images.
 * @param[in] nbasenames The number of names in basename.
 * @param[in] stride The stride type. @see iftGFLIMStrideType.
 */
void iftGFLIMStrideBatch(iftGFLIMGraphArray *graph, iftGFLIMLayer layer,
                         char *in_labels_dir, char *out_labels_dir,
                         char **basename, int nbasenames,
                         iftGFLIMStrideType stride)
{
    if (stride == IFT_GFLIM_NOSTRIDE || layer.pool_stride == 1 || (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride < 2))
        return;

    for (int i = 0; i < nbasenames; i++)
    {
        char filename[512];
        iftGFLIMGraph *output_graph = NULL;

        sprintf(filename, "%s/%s.pgm", in_labels_dir, basename[i]);
        iftImage *input_labels = iftReadImageByExt(filename);

        if (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride >= 2)
        {
            iftImage *output_labels = NULL;
            int new_x = ceilf((float)input_labels->xsize / layer.pool_stride);
            int new_y = ceilf((float)input_labels->ysize / layer.pool_stride);
            int new_z = iftMax(ceilf((float)input_labels->zsize / layer.pool_stride), 1);
            output_labels = iftCreateImage(new_x, new_y, new_z);
            output_graph = iftGFLIMImageStride(graph->val[i], input_labels, output_labels, floorf(layer.pool_stride));
            iftDestroyGFLIMGraph(&(graph->val[i]));
            graph->val[i] = output_graph;
            output_graph = NULL;
            /* save output */
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[i]);
            iftWriteImageByExt(output_labels, filename);
            iftDestroyImage(&output_labels);
        }
        if (stride == IFT_GFLIM_GRAPHSTRIDE)
        {
            int nodes_map[graph->val[i]->num_nodes];
            output_graph = iftGFLIMGraphStride(graph->val[i], layer.pool_stride, nodes_map);
            for (int j = 0; j < input_labels->n; j++)
                input_labels->val[j] = nodes_map[input_labels->val[j] - 1] + 1;
            iftDestroyGFLIMGraph(&(graph->val[i]));
            graph->val[i] = output_graph;
            output_graph = NULL;
            /* save output */
            sprintf(filename, "%s/%s.pgm", out_labels_dir, basename[i]);
            iftWriteImageByExt(input_labels, filename);
        }
        iftDestroyImage(&input_labels);
    }
}

void iftGFLIMStrideBatch_v2(iftGFLIMGraphArray *graph, iftGFLIMLayer layer,
                            iftImageArray *labels_img, iftGFLIMStrideType stride)
{
    if (stride == IFT_GFLIM_NOSTRIDE || layer.pool_stride == 1 || (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride < 2))
        return;

    if (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride >= 2)
    {
        for (int i = 0; i < graph->n; i++)
        {
            int new_x = ceilf((float)labels_img->val[i]->xsize / layer.pool_stride);
            int new_y = ceilf((float)labels_img->val[i]->ysize / layer.pool_stride);
            int new_z = iftMax(ceilf((float)labels_img->val[i]->zsize / layer.pool_stride), 1);
            iftImage *output_labels = iftCreateImage(new_x, new_y, new_z);
            iftGFLIMGraph *output_graph = iftGFLIMImageStride(graph->val[i], labels_img->val[i], output_labels, floorf(layer.pool_stride));
            iftDestroyImage(&(labels_img->val[i]));
            labels_img->val[i] = output_labels;
            output_labels = NULL;
            iftDestroyGFLIMGraph(&(graph->val[i]));
            graph->val[i] = output_graph;
            output_graph = NULL;
        }
    }
    else
    {
        if (stride == IFT_GFLIM_GRAPHSTRIDE)
        {
            for (int i = 0; i < graph->n; i++)
            {
                iftGFLIMGraph *output_graph = NULL;
                int nodes_map[graph->val[i]->num_nodes];
                output_graph = iftGFLIMGraphStride(graph->val[i], layer.pool_stride, nodes_map);
                for (int j = 0; j < labels_img->val[i]->n; j++)
                    labels_img->val[i]->val[j] = nodes_map[labels_img->val[i]->val[j] - 1] + 1;
                iftDestroyGFLIMGraph(&(graph->val[i]));
                graph->val[i] = output_graph;
                output_graph = NULL;
            }
        }
    }
}

// ============ FUNCTIONS TO ENCODE LAYER ============ //

/**
 * @fn iftGFLIMConvLayer
 * @brief Compute convolution between a graph and a set of kernels.
 * @param graph The input graph.
 * @param kernels The set of kernels.
 * @param kernel_size The kernel size.
 * @param attention_opt The attention option. @see iftGFLIMGraphAttention.
 * @param kernel_labels The kernel labels.
 * @param aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @param maintain_kernels Used with reduce option. Vector with nnodes size in which each position indicates whether the channel should be maintained or not.
 * @param reduce If true, use attention values to remove some channels from the output graph.
 * @return A mimage with the convolution result.
 */
iftMImage *iftGFLIMConvLayer(iftGFLIMGraph *graph, iftMatrix *kernels,
                             int kernel_size, iftGFLIMGraphAttention attention_opt,
                             int *kernel_labels, iftGFLIMAggregation aggregation_opt,
                             /*bool *maintain_kernels,*/ bool reduce)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (kernels == NULL)
        printf("\t kernels: NULL \n");
    else
        printf("\t kernels size: %d %d \n", kernels->nrows, kernels->ncols);
    printf("\t kernel_size: %d\n", kernel_size);
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t kernel_labels: %s\n", kernel_labels == NULL ? "NULL" : "not NULL");
    printf("\t aggregation_opt: %d\n", aggregation_opt);
    printf("\t maintain_kernels: %s\n", maintain_kernels == NULL ? "NULL" : "not NULL");
    printf("\t reduce: %s\n", reduce ? "true" : "false");
#endif

    /*iftIntArray **neighbors_all = NULL;
    if (attention_opt == IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL)
    {
        neighbors_all = (iftIntArray **)calloc(graph->num_nodes, sizeof(iftIntArray *));
        for (int p = 0; p < graph->num_nodes; p++)
            neighbors_all[p] = iftCreateIntArray(kernel_size);
    }*/

    iftMatrix *XI = iftGFLIMGraphToNeighborhoodFeatureMatrix(graph, kernel_size, attention_opt, aggregation_opt/*, neighbors_all*/);

#ifdef PRINT_VALS
    printf("XI: %d %d \n", XI->nrows, XI->ncols);
    for (int i = 0; i < XI->nrows; i++)
    {
        for (int j = 0; j < XI->ncols; j++)
            printf("%f ", iftMatrixElem(XI, j, i));
        printf("\n");
    }
    printf("\n");
#endif

    iftMatrix *XJ = iftMultMatrices(XI, kernels);
    iftMatrix *A = NULL;

    switch (attention_opt)
    {
    /*case IFT_FLIMGRAPH_ATTENTION_NEIGHB_SIMIL:
        if (kernel_labels == NULL)
            iftError("kernel_labels is NULL", __func__);
        iftMatrix *XK = iftCreateMatrix(XJ->ncols, XJ->nrows);

        int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
        for (int i = 1; i < XJ->ncols; i++)
        {
            if (kernel_labels[i] < minLabel)
                minLabel = kernel_labels[i];
            if (kernel_labels[i] > nlabels)
                nlabels = kernel_labels[i];
        }
        nlabels = nlabels - minLabel + 1;

        for (int p = 0; p < graph->num_nodes; p++)
        {
            for (int l = 0; l < nlabels; l++)
            {
                iftMatrix *X = iftGFLIMGraphAttentionNeighborSimilarity(XJ, neighbors_all[p], kernel_labels, l + minLabel);
                iftMatrix *aux = iftCreateMatrix(1, X->ncols);

                int k = 0;
                for (int b = 0; b < XJ->ncols; b++)
                {
                    if (kernel_labels[b] == l + minLabel)
                    {
                        iftMatrixElem(aux, 0, k) = iftMatrixElem(XJ, b, p);
                        k++;
                    }
                }

                iftMatrix *H = iftMultMatrices(X, aux);
                iftDestroyMatrix(&X);
                iftDestroyMatrix(&aux);

                k = 0;
                for (int b = 0; b < XJ->ncols; b++)
                {
                    if (kernel_labels[b] == l + minLabel)
                    {
                        iftMatrixElem(XK, b, p) = iftMatrixElem(H, 0, k) + iftMatrixElem(XJ, b, p);
                        k++;
                    }
                }
                iftDestroyMatrix(&H);
            }

            iftDestroyIntArray(&neighbors_all[p]);
        }
        iftDestroyMatrix(&XJ);
        iftFree(neighbors_all);
        XJ = XK;
        XK = NULL;
        break;*/

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_TEST:

        if (kernel_labels == NULL)
            iftError("kernel_labels is NULL", __func__);
        A = iftGFLIMGraphAttentionClassDominance_v2(XJ, graph, kernel_labels);
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE:
        if (kernel_labels == NULL)
            iftError("kernel_labels is NULL", __func__);
        A = iftGFLIMGraphAttentionClassDominance(XJ, graph, kernel_labels);
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_POWER2:
        if (kernel_labels == NULL)
            iftError("kernel_labels is NULL", __func__);
        A = iftGFLIMGraphAttentionClassDominancePower2(XJ, graph, kernel_labels);
        break;

    case IFT_FLIMGRAPH_ATTENTION_CLASS_DOMINANCE_SOFTMAX_NEIGHB:
        if (kernel_labels == NULL)
            iftError("kernel_labels is NULL", __func__);
        A = iftGFLIMGraphAttentionClassDominance_softmaxNeigh(XJ, graph, kernel_labels);
        break;

    // no need for a default case here, since some attention strategies are applied before the convolution
    }

    if (A != NULL)
    {
        if (reduce)
        {
            iftMatrix *XA = iftMatricesMultiplicationPointWise(XJ, A);
            iftDestroyMatrix(&A);

            // compute the sum of the difference between the original and the attention matrix
            // 'sumAttention' contain the sum of the difference between the XJ and XA
            iftMatrix *X_diff = iftMatricesSubtractionPointWise(XA, XJ);
            iftMatrix *sumAttention = iftComputeSumVectorBycolumn(X_diff);
            
            // compute the number of labels
            int nlabels = kernel_labels[0], minLabel = kernel_labels[0];
            for (int i = 1; i < XJ->ncols; i++)
            {
                if (kernel_labels[i] < minLabel)
                    minLabel = kernel_labels[i];
                if (kernel_labels[i] > nlabels)
                    nlabels = kernel_labels[i];
            }
            nlabels = nlabels - minLabel + 1;
            
            for (int l = minLabel + 1; l <= nlabels; l++)
            {
                // compute the number of kernels and min and max values of the attention for the current label
                float min_val = IFT_INFINITY_FLT;
                float max_val = IFT_INFINITY_FLT_NEG;
                int nkernels = 0;
                for (int i = 0; i < XJ->ncols; i++)
                {
                    if (kernel_labels[i] == l){
                        iftMatrixElem(sumAttention, i, 0) /= XJ->nrows;
                        min_val = iftMin(min_val, iftMatrixElem(sumAttention, i, 0));
                        max_val = iftMax(max_val, iftMatrixElem(sumAttention, i, 0));
                        nkernels++;
                    }
                }
                
                // normalize the attention values for the current label and put them in 'otsu_img'
                iftImage *otsu_img = iftCreateImage(nkernels, 1, 1);
                for (int i = 0, j = 0; i < XJ->ncols; i++)
                {
                    if (kernel_labels[i] == l){
                        iftMatrixElem(sumAttention, i, 0) = 255*(iftMatrixElem(sumAttention, i, 0) - min_val)/(max_val - min_val);
                        otsu_img->val[j++] = (int)iftMatrixElem(sumAttention, i, 0);
                    }
                }

                // perform otsu thresholding on the attention difference values of kernels with the same labels
                int otsu_threshold = iftOtsu(otsu_img);
                iftDestroyImage(&otsu_img);

                for (int i = 0; i < XJ->ncols; i++)
                {
                    // maintain the attention values of the kernels that are below the otsu threshold (i.e., their values decrease after the attention)
                    if (kernel_labels[i] == l && (int)iftMatrixElem(sumAttention, i, 0) <= otsu_threshold)
                    {
                        for (int j = 0; j < XJ->nrows; j++)
                            iftMatrixElem(XJ, i, j) = iftMatrixElem(XA, i, j);
                    }
                }
            }
            iftDestroyMatrix(&XA);

            iftDestroyMatrix(&sumAttention);
        }
        else
        {
            iftMatrix *XA = iftMatricesMultiplicationPointWise(XJ, A);
            iftDestroyMatrix(&A);
            iftDestroyMatrix(&XJ);
            XJ = XA;
            XA = NULL;
        }
    }

#ifdef PRINT_VALS
    printf("XJ: %d %d \n", XJ->nrows, XJ->ncols);
    for (int i = 0; i < XJ->nrows; i++)
    {
        for (int j = 0; j < XJ->ncols; j++)
            printf("%f ", iftMatrixElem(XJ, j, i));
        printf("\n");
    }
    printf("\n");
#endif

    iftDestroyMatrix(&XI);
    iftMImage *activ = iftMatrixToMImage(XJ, XJ->nrows,
                                         1, 1,
                                         XJ->ncols, 'c');
#ifdef PRINT_VALS
    printf("activ: %d %d %d %ld \n", activ->xsize, activ->ysize, activ->zsize, activ->m);
    for (int i = 0; i < activ->n; i++)
    {
        for (int j = 0; j < activ->m; j++)
            printf("%f ", activ->val[i][j]);
        printf("\n");
    }
    printf("\n");
#endif

    iftDestroyMatrix(&XJ);
    return activ;
}

/**
 * @fn iftGFLIMSumMImgArrayToFloatArray
 * @brief Sum the bias to a mimage array.
 * @param mimgArray The input mimage array.
 * @param bias The bias to be summed.
 * @param inplace If true, the bias is summed in place.
 */
iftMImageArray *iftGFLIMSumMImgArrayToFloatArray(iftMImageArray *mimgArray, float *bias,
                                                 bool inplace)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (mimgArray == NULL)
        printf("\t mimgArray: NULL \n");
    else
        printf("\t mimgArray size: %d \n", mimgArray->n);
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t inplace: %s\n", inplace ? "true" : "false");
#endif

    iftMImageArray *mimgArrayOut = mimgArray;

    if (!inplace)
        mimgArrayOut = iftCreateMImageArray(mimgArray->n);

    for (int i = 0; i < mimgArray->n; i++)
    {
        mimgArrayOut->val[i] = iftGFLIMSumMImgToFloatArray(mimgArray->val[i], bias, inplace);
    }

    return mimgArrayOut;
}

/**
 * @fn iftGFLIMConvLayerBatch
 * @brief Convolution layer batch.
 * @param graph The input graph.
 * @param kernels The set of kernels.
 * @param kernel_size The kernel size.
 * @return The mimage array with the convolution result.
 */
iftMImageArray *iftGFLIMConvLayerBatch(iftGFLIMGraphArray *graph, iftMatrix *kernels, int kernel_size,
                                       iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                                       iftGFLIMAggregation aggregation_opt, /*bool **maintain_kernels,*/ bool reduce)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph size: %d \n", graph->n);
    if (kernels == NULL)
        printf("\t kernels: NULL \n");
    else
        printf("\t kernels size: %d %d \n", kernels->nrows, kernels->ncols);
    printf("\t kernel_size: %d\n", kernel_size);
#endif

    iftMImageArray *activ = iftCreateMImageArray(graph->n);
    for (int i = 0; i < graph->n; i++)
    {
        activ->val[i] = iftGFLIMConvLayer(graph->val[i], kernels, kernel_size, attention_opt, kernel_labels, aggregation_opt, /*maintain_kernels == NULL? NULL : maintain_kernels[i],*/ reduce);
    }
    return activ;
}

/**
 * @fn iftGFLIMPoolingArray
 * @brief Pooling layer.
 * @param activ_array The input mimage array.
 * @param graph The input graph.
 * @param pool_type The pooling type.
 * @param pool_size The pooling size.
 * @param pool_stride The pooling stride.
 * @note The pooling is applied to the graph nodes.
 */
void iftGFLIMPoolingArray(iftMImageArray *activ_array, iftGFLIMGraphArray *graph,
                          char *pool_type, int pool_size, float pool_stride)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (activ_array == NULL)
        printf("\t activ_array: NULL \n");
    else
        printf("\t activ_array size: %d \n", activ_array->n);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph size: %d \n", graph->n);
    printf("\t pool_type: %s\n", pool_type);
    printf("\t pool_size: %d \n", pool_size);
    printf("\t pool_stride: %f\n", pool_stride);
#endif

    for (int i = 0; i < activ_array->n; i++)
    {
        iftMImageToGFLIMGraphInplace(activ_array->val[i], graph->val[i]);
        iftGFLIMPoolingLayer(graph->val[i], pool_type, pool_size, pool_stride);
    }
}

/**
 * @fn iftGFLIMEncodeLayer
 * @brief Encode a layer. It computes the convolution, pooling and stride (if any).
 * @param[in] graph The input graph.
 * @param[in] Kernels The set of kernels.
 * @param[in] bias The bias.
 * @param[in] layer The layer configuration.
 * @param[in] stride The stride.
 * @param[in] input_labels The image with the superpixels of the input graph. It maps the image pixels to the graph nodes (from 1).
 * @param[out] output_labels A pointer to an iftImage*. Is allocated inside the function. After the function, it will be filled with the image-graph mapping values (from 1).
 * @param[in] attention_opt The attention option. @see iftGFLIMGraphAttention.
 * @param[in] kernel_labels The kernel labels.
 * @param[in] aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @param[in] reduce If true, use attention values to remove some channels from the output graph.
 * @return The output graph after encoding the layer.
 * @note If stride is IFT_GFLIM_NOSTRIDE the input_labels and output_labels can be NULL.
 */
iftGFLIMGraph *iftGFLIMEncodeLayer(iftGFLIMGraph *graph,
                                  iftMatrix *Kernels, float *bias,
                                  iftGFLIMLayer layer, iftGFLIMStrideType stride,
                                  iftImage *input_labels, iftImage **output_labels,
                                  iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                                  iftGFLIMAggregation aggregation_opt,
                                  bool reduce)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (Kernels == NULL)
        printf("\t Kernels: NULL \n");
    else
        printf("\t Kernels size: %d %d \n", Kernels->nrows, Kernels->ncols);
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t stride: %d\n", stride);
    if (input_labels == NULL)
        printf("\t input_labels: NULL \n");
    else
        printf("\t input_labels size: %d %d %d \n", input_labels->xsize, input_labels->ysize, input_labels->zsize);
    printf("\t output_labels: %s\n", output_labels == NULL ? "NULL" : "not NULL");
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t kernel_labels: %s\n", kernel_labels == NULL ? "NULL" : "not NULL");
    printf("\t aggregation_opt: %d\n", aggregation_opt);
    printf("\t reduce: %s\n", reduce ? "true" : "false");
#endif

    /*bool *channels_maintain = NULL;
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && reduce)
        channels_maintain = iftAllocBoolArray(Kernels->ncols);*/

    iftMImage *activ = iftGFLIMConvLayer(graph, Kernels, layer.kernel_size, attention_opt, kernel_labels, aggregation_opt, /*channels_maintain,*/ reduce);

    iftMImage *activ_relu = NULL;
    /*if (channels_maintain != NULL)
    {
        float *bias_aux = iftAllocFloatArray(activ->m);
        if(bias != NULL)
        {
            for (int i = 0, j = 0; i < Kernels->ncols; i++)
            {
                if (channels_maintain[i])
                {
                    bias_aux[j] = bias[i];
                    j++;
                }
            }
        }
        if (layer.relu)
            activ_relu = iftMReLU(activ, bias_aux, true);
        else
        {
            if (bias_aux != NULL)
                activ_relu = iftGFLIMSumMImgToFloatArray(activ, bias_aux, true);
            else
                activ_relu = activ;
        }
        if (bias_aux != NULL && bias_aux != bias)
            iftFree(bias_aux);
        
        iftFree(channels_maintain);
    }
    else
    {*/
        if (layer.relu)
            activ_relu = iftMReLU(activ, bias, true);
        else
        {
            if (bias != NULL)
                activ_relu = iftGFLIMSumMImgToFloatArray(activ, bias, true);
            else
                activ_relu = activ;
        }
    //}
    activ = NULL;


#ifdef PRINT_VALS
    printf("activ_relu: %d %d %d %ld \n", activ_relu->xsize, activ_relu->ysize, activ_relu->zsize, activ_relu->m);
    for (int i = 0; i < activ_relu->n; i++)
    {
        for (int j = 0; j < activ_relu->m; j++)
            printf("%f ", activ_relu->val[i][j]);
        printf("\n");
    }
    printf("\n");
#endif

    /* pooling */
    iftGFLIMGraph *pooling_graph = iftMImageToGFLIMGraph(activ_relu, graph);
    iftDestroyMImage(&activ_relu);
    iftGFLIMPoolingLayer(pooling_graph, layer.pool_type, layer.pool_size, 1);

    /* stride model */
    /*iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(1);
    input->val[0] = pooling_graph;
    iftImageArray *labels_img = iftCreateImageArray(1);
    labels_img->val[0] = input_labels;
    iftGFLIMStrideBatch_v2(input, layer, labels_img, stride);
    pooling_graph = input->val[0];
    input->val[0] = NULL;
    iftDestroyGFLIMGraphArray(&input);
    (*output_labels) = labels_img->val[0];
    labels_img->val[0] = NULL;
    iftDestroyImageArray(&labels_img);*/


    iftGFLIMGraph *output_graph = NULL;
    if (stride == IFT_GFLIM_NOSTRIDE || layer.pool_stride == 1 || (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride < 2))
    {
        output_graph = pooling_graph;
        pooling_graph = NULL;
    }

    if (stride == IFT_GFLIM_IMAGESTRIDE && layer.pool_stride >= 2)
    {
        int new_x = ceilf(input_labels->xsize / (float)layer.pool_stride);
        int new_y = ceilf(input_labels->ysize / (float)layer.pool_stride);
        int new_z = iftMax(ceilf(input_labels->zsize / (float)layer.pool_stride), 1);

        (*output_labels) = iftCreateImage(new_x, new_y, new_z);
        output_graph = iftGFLIMImageStride(pooling_graph, input_labels, (*output_labels), floorf(layer.pool_stride));
    }

    if (stride == IFT_GFLIM_GRAPHSTRIDE && layer.pool_stride > 1)
    {
        int nodes_map[pooling_graph->num_nodes];
        (*output_labels) = iftCreateImage(input_labels->xsize, input_labels->ysize, input_labels->zsize);
        output_graph = iftGFLIMGraphStride(pooling_graph, layer.pool_stride, nodes_map);
        for (int j = 0; j < input_labels->n; j++)
            (*output_labels)->val[j] = nodes_map[input_labels->val[j] - 1] + 1;
    }

    if (pooling_graph != NULL)
        iftDestroyGFLIMGraph(&pooling_graph);

    if (layer.pool_stride == 1 && stride != IFT_GFLIM_NOSTRIDE)
        (*output_labels) = iftCopyImage(input_labels);

    return output_graph;
}

/**
 * @fn iftGFLIMEncodeLayerBatch
 * @brief Encode a layer for a batch of graphs.
 * @param[in,out] input The input graph array (the batch). After the function, it will be filled with the output graphs.
 * @param[in] Kmerged The set of kernels.
 * @param[in] layer The layer configuration.
 * @param[in] in_labels_dir The input labels directory. It can be NULL if stride is IFT_GFLIM_NOSTRIDE.
 * @param[in] out_labels_dir The output labels directory. It can be NULL if stride is IFT_GFLIM_NOSTRIDE.
 * @param[in] bias_merged An array with bias values.
 * @param[in] basename An array with the name of the images. It can be NULL if stride is IFT_GFLIM_NOSTRIDE.
 * @param[in] nbasenames The number of names in basename.
 * @param[in] stride The stride type.
 * @param[in] layer_index The layer index.
 * @param[in] attention_opt The attention option. @see iftGFLIMGraphAttention.
 * @param[in] kernel_labels The kernel labels.
 * @param[in] aggregation_opt The aggregation option. @see iftGFLIMAggregation.
 * @param[in] reduce If true, use attention values to remove some channels from the output graph.
 * @note The stride is applied only if the pool_stride in layer config. is greater than 1.
 */
void iftGFLIMEncodeLayerBatch(iftGFLIMGraphArray *input,
                              iftMatrix *Kmerged, iftGFLIMLayer layer,
                              char *in_labels_dir, char *out_labels_dir,
                              float *bias_merged,
                              char **basename, int nbasenames,
                              iftGFLIMStrideType stride, int layer_index,
                              iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                              iftGFLIMAggregation aggregation_opt, bool reduce)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (input == NULL)
        printf("\t input: NULL \n");
    else
        printf("\t input size: %d \n", input->n);
    if (Kmerged == NULL)
        printf("\t Kmerged: NULL \n");
    else
        printf("\t Kmerged size: %d %d \n", Kmerged->nrows, Kmerged->ncols);
    if (stride != IFT_GFLIM_NOSTRIDE)
    {
        printf("\t in_labels_dir: %s\n", in_labels_dir);
        printf("\t out_labels_dir: %s\n", out_labels_dir);
        printf("\t basename: %s\n", basename == NULL ? "NULL" : "not NULL");
        printf("\t nbasenames: %d\n", nbasenames);
    }
    printf("\t bias_merged: %s\n", bias_merged == NULL ? "NULL" : "not NULL");
    printf("\t stride: %d\n", stride);
    printf("\t layer_index: %d\n", layer_index);
#endif

    /*bool **maintain_kernels = NULL;
    if (attention_opt == IFT_FLIMGRAPH_ATTENTION_NONE && reduce)
    {
        maintain_kernels = iftAlloc(input->n, sizeof(bool *));
        for (int i = 0; i < input->n; i++)
            maintain_kernels[i] = iftAlloc(Kmerged->ncols, sizeof(bool));
    }*/

    /* convolution */
    iftMImageArray *activ = iftGFLIMConvLayerBatch(input, Kmerged, layer.kernel_size, attention_opt, kernel_labels, aggregation_opt, /*maintain_kernels,*/ reduce);

    /* activation and bias */
    iftMImageArray *activ_relu = activ;
    /*if (maintain_kernels != NULL && bias_merged != NULL)
    {
        for (int i = 0; i < activ_relu->n; i++)
        {
            float *bias_aux = iftAllocFloatArray(activ_relu->val[i]->m);
            for (int k = 0, j = 0; k < Kmerged->ncols; k++)
            {
                if (maintain_kernels[i][k])
                {
                    bias_aux[j] = bias_merged[k];
                    j++;
                }
            }
            if (layer.relu)
                activ_relu->val[i] = iftMReLU(activ_relu->val[i], bias_aux, true);
            else
                activ_relu->val[i] = iftGFLIMSumMImgToFloatArray(activ_relu->val[i], bias_aux, true);

            iftFree(bias_aux);
        }
        for (int i = 0; i < input->n; i++)
            iftFree(maintain_kernels[i]);
        iftFree(maintain_kernels);
    }
    else
    {*/
        if (layer.relu)
            activ_relu = iftMReLUArray(activ, bias_merged, layer.relu);
        else
        {
            if (bias_merged != NULL)
            {
                activ_relu = iftGFLIMSumMImgArrayToFloatArray(activ, bias_merged, true);
            }
        }
    //}

    activ = NULL;

    /* pooling */
    iftGFLIMPoolingArray(activ_relu, input, layer.pool_type, layer.pool_size, 1);
    iftDestroyMImageArray(&activ_relu);

    /* stride */
    iftGFLIMStrideBatch(input, layer, in_labels_dir, out_labels_dir, basename, nbasenames, stride);
}

void iftGFLIMEncodeLayerBatch_v2(iftGFLIMGraphArray *input,
                                 iftMatrix *Kmerged, iftGFLIMLayer layer,
                                 iftImageArray *labels_img,
                                 float *bias_merged,
                                 iftGFLIMStrideType stride,
                                 iftGFLIMGraphAttention attention_opt, int *kernel_labels,
                                 iftGFLIMAggregation aggregation_opt, bool reduce)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (input == NULL)
        printf("\t input: NULL \n");
    else
        printf("\t input size: %d \n", input->n);
    if (Kmerged == NULL)
        printf("\t Kmerged: NULL \n");
    else
        printf("\t Kmerged size: %d %d \n", Kmerged->nrows, Kmerged->ncols);
    if (labels_img == NULL)
        printf("\t labels_img: NULL \n");
    else
        printf("\t labels_img size: %d \n", (int)labels_img->n);
    printf("\t bias_merged: %s\n", bias_merged == NULL ? "NULL" : "not NULL");
    printf("\t stride: %d\n", stride);
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t kernel_labels: %s\n", kernel_labels == NULL ? "NULL" : "not NULL");
    printf("\t aggregation_opt: %d\n", aggregation_opt);
    printf("\t reduce: %s\n", reduce ? "true" : "false");
#endif

    /*bool **maintain_kernels = NULL;
    if (attention_opt != IFT_FLIMGRAPH_ATTENTION_NONE && reduce)
    {
        maintain_kernels = iftAlloc(input->n, sizeof(bool *));
        for (int i = 0; i < input->n; i++)
            maintain_kernels[i] = iftAlloc(Kmerged->ncols, sizeof(bool));
    }*/

    /* convolution */
    iftMImageArray *activ = iftGFLIMConvLayerBatch(input, Kmerged, layer.kernel_size, attention_opt, kernel_labels, aggregation_opt, /*maintain_kernels,*/ reduce);

    /* activation and bias */
    iftMImageArray *activ_relu = activ;
    /*if (maintain_kernels != NULL && bias_merged != NULL)
    {
        for (int i = 0; i < activ_relu->n; i++)
        {
            float *bias_aux = iftAllocFloatArray(activ_relu->val[i]->m);
            for (int k = 0, j = 0; k < Kmerged->ncols; k++)
            {
                if (maintain_kernels[i][k])
                {
                    bias_aux[j] = bias_merged[k];
                    j++;
                }
            }
            if (layer.relu)
                activ_relu->val[i] = iftMReLU(activ_relu->val[i], bias_aux, true);
            else
                activ_relu->val[i] = iftGFLIMSumMImgToFloatArray(activ_relu->val[i], bias_aux, true);

            iftFree(bias_aux);
        }
        for (int i = 0; i < input->n; i++)
            iftFree(maintain_kernels[i]);
        iftFree(maintain_kernels);
    }
    else
    {*/
        if (layer.relu)
            activ_relu = iftMReLUArray(activ, bias_merged, true);
        else
        {
            if (bias_merged != NULL)
            {
                activ_relu = iftGFLIMSumMImgArrayToFloatArray(activ, bias_merged, true);
            }else
                activ_relu = activ;
        }
    //}
    activ = NULL;

    /* pooling */
    iftGFLIMPoolingArray(activ_relu, input, layer.pool_type, layer.pool_size, 1);
    iftDestroyMImageArray(&activ_relu);

    /* stride */
    iftGFLIMStrideBatch_v2(input, layer, labels_img, stride);
}

// ============ FUNCTIONS TO PROCESS RESIDUAL CONNECTIONS ============ //

int* iftGFLIMGetFinerResidualLayer(iftGFLIMArch *arch)
{
    /*
    Given the current layer, find the finest layer that has a skip connection to it.
    If there is no skip connection to the current layer, return the current layer (from 1).
    */

    int *finer_layer = iftAllocIntArray(arch->nlayers);
    for(int l = 0; l < arch->nlayers; l++)
        finer_layer[l] = l+1;

    for(int curr_layer = 0; curr_layer < arch->nlayers; curr_layer++)
    {
        for(int prev_layer = 0; prev_layer < curr_layer; prev_layer++)
        {
            if (arch->layer[curr_layer].skip_connection[prev_layer] > 0 && finer_layer[prev_layer] < finer_layer[curr_layer])
                finer_layer[curr_layer] = finer_layer[prev_layer];
        }
    }
    
    return finer_layer;
}

iftMatrix *iftGFLIMUpsampleFeatureMatrix(iftMatrix *X, 
                                    int *nodes_map, iftIntArray *kernels_map, 
                                    int nnodes, int nfeats)
{
    iftMatrix *X_up = iftCreateMatrix(nfeats, nnodes);
    for(int i = 0; i < nnodes; i++)
    {
        for(int j = 0; j < nfeats; j++)
        {
            iftMatrixElem(X_up, j, i) = iftMatrixElem(X, kernels_map->val[j], nodes_map[i]);
        }
    }
    return X_up;
}

float *iftGFLIMOtsuFromFeatureMatrix(iftMatrix *X)
{
    /*
    Compute the otsu threshold for each feature/channel of the feature matrix X.
    */
    float *otsu_thresholds = iftAllocFloatArray(X->ncols);
    for(int j = 0; j < X->ncols; j++)
    {
        // Just so it is easier to compute the otsu
        iftImage *feat_img = iftCreateImage(X->nrows, 1, 1);
        float max_feat = 0.0;
        for(int i = 0; i < X->nrows; i++)
        {
            feat_img->val[i] = iftMatrixElem(X, j, i) * 255;
            if (feat_img->val[i] > max_feat)
                max_feat = feat_img->val[i];
        }

        if (max_feat > 0.0)
        {
            // Compute otsu threshold as a way to separate foreground bands from background bands
            otsu_thresholds[j] = (float)iftOtsu(feat_img) / 255.0;
        }
        else
            otsu_thresholds[j] = 0.0;

        iftDestroyImage(&feat_img);
    }
    return otsu_thresholds;
}

void iftGFLIMSumValueConditionToFeatureMatrix(iftMatrix *X, float *values)
{
    /*
    Sum a value to all elements of the feature matrix X that are above a certain threshold.
    */
    for(int j = 0; j < X->ncols; j++)
    {
        float thresh = values[j];
        if(thresh != 0)
        {
            for(int i = 0; i < X->nrows; i++)
            {
                if(iftMatrixElem(X, j, i) > thresh)
                    iftMatrixElem(X, j, i) += thresh;
            }
        }
    }
}

float **iftGFLIMStdFromFeatureMatrix(iftMatrix **X, int n, int finer_scale)
{
    /*
    compute the std per node and channel across multiple feature matrices
    */
    float **stds = iftAlloc(X[finer_scale]->nrows, sizeof(float *));
    for(int i = 0; i < X[finer_scale]->nrows; i++)
        stds[i] = iftAllocFloatArray(X[finer_scale]->ncols);

    for(int i = 0; i < X[finer_scale]->nrows; i++)
    {
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            float sum = 0.0;
            float sum_sqr = 0.0;

            for(int s = 0; s < n; s++)
            {
                float val = iftMatrixElem(X[s], j, i);
                sum += val;
                sum_sqr += val * val;
            }

            float mean = sum / n;
            float var = (sum_sqr / n) - (mean * mean);
            stds[i][j] = sqrtf(var);
        }
    }
    return stds;
}

iftMatrix *iftGFLIMMergeMatrices_dotProd(iftMatrix **X, int n, 
                                        int **nodes_map, iftIntArray **kernels_map, 
                                        int finer_scale, bool use_otsu, bool use_std)
{
    /*
    Merge two feature matrices by dot product.
    */
    iftMatrix *merged = iftCopyMatrix(X[finer_scale]);

    if(use_otsu)
    {
        float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(merged);
        iftGFLIMSumValueConditionToFeatureMatrix(merged, otsu_thresholds);
        iftFree(otsu_thresholds);
    }
    
    for(int i = 0; i < n; i++)
    {
        if(i == finer_scale) continue;
        iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(X[i], nodes_map[i], kernels_map[i], X[finer_scale]->nrows, X[finer_scale]->ncols);

        if(use_otsu)
        {
            float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(tmp);
            iftGFLIMSumValueConditionToFeatureMatrix(tmp, otsu_thresholds);
            iftFree(otsu_thresholds);
        }

        // dot product
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            for(int k = 0; k < X[finer_scale]->nrows; k++)
            {
                iftMatrixElem(merged, j, k) *= iftMatrixElem(tmp, j, k);
            }
        }
        iftDestroyMatrix(&tmp);
    }

    if(use_std)
    {
        float **stds = iftGFLIMStdFromFeatureMatrix(X, n, finer_scale);
        for(int i = 0; i < merged->nrows; i++)
        {
            float *stds_i = stds[i];
            for(int j = 0; j < merged->ncols; j++)
            {
                iftMatrixElem(merged, j, i) *= (1 - *stds_i);
                stds_i++;
            }
            stds_i = NULL;
            iftFree(stds[i]);
        }
        iftFree(stds);
    }

    return merged;
}

iftMatrix *iftGFLIMMergeMatrices_dotProd_sumScale0(iftMatrix **X, int n, 
                                        int **nodes_map, iftIntArray **kernels_map, 
                                        int finer_scale, bool use_otsu, bool use_std)
{
    /*
    Merge two feature matrices by dot product and sum it with the finer scale.
    */
    iftMatrix *merged = iftCopyMatrix(X[finer_scale]);

    if(use_otsu)
    {
        float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(merged);
        iftGFLIMSumValueConditionToFeatureMatrix(merged, otsu_thresholds);
        iftFree(otsu_thresholds);
    }
    
    for(int i = 0; i < n; i++)
    {
        if(i == finer_scale) continue;
        iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(X[i], nodes_map[i], kernels_map[i], X[finer_scale]->nrows, X[finer_scale]->ncols);

        if(use_otsu)
        {
            float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(tmp);
            iftGFLIMSumValueConditionToFeatureMatrix(tmp, otsu_thresholds);
            iftFree(otsu_thresholds);
        }

        // dot product
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            for(int k = 0; k < X[finer_scale]->nrows; k++)
            {
                iftMatrixElem(merged, j, k) *= iftMatrixElem(tmp, j, k);
            }
        }
        iftDestroyMatrix(&tmp);
    }

    if(use_std)
    {
        float **stds = iftGFLIMStdFromFeatureMatrix(X, n, finer_scale);
        for(int i = 0; i < merged->nrows; i++)
        {
            float *stds_i = stds[i];
            for(int j = 0; j < merged->ncols; j++)
            {
                iftMatrixElem(merged, j, i) *= (1 - *stds_i);
                stds_i++;
            }
            stds_i = NULL;
            iftFree(stds[i]);
        }
        iftFree(stds);
    }

    for(int i = 0; i < X[finer_scale]->nrows; i++)
    {
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            iftMatrixElem(merged, j, i) += iftMatrixElem(X[finer_scale], j, i);
        }
    }
    return merged;
}

iftMatrix *iftGFLIMMergeMatrices_dotProd_sumScalei(iftMatrix **X, int n, 
                                        int **nodes_map, iftIntArray **kernels_map, 
                                        int finer_scale, bool use_otsu, bool use_std)
{
    /*
    Merge two feature matrices by dot product and sum it with the last scale.
    */
    iftMatrix *merged = iftCopyMatrix(X[finer_scale]);

    if(use_otsu)
    {
        float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(merged);
        iftGFLIMSumValueConditionToFeatureMatrix(merged, otsu_thresholds);
        iftFree(otsu_thresholds);
    }
    
    for(int i = 0; i < n; i++)
    {
        if(i == finer_scale) continue;
        iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(X[i], nodes_map[i], kernels_map[i], X[finer_scale]->nrows, X[finer_scale]->ncols);

        if(use_otsu)
        {
            float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(tmp);
            iftGFLIMSumValueConditionToFeatureMatrix(tmp, otsu_thresholds);
            iftFree(otsu_thresholds);
        }

        // dot product
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            for(int k = 0; k < X[finer_scale]->nrows; k++)
            {
                iftMatrixElem(merged, j, k) *= iftMatrixElem(tmp, j, k);
            }
        }
        iftDestroyMatrix(&tmp);
    }

    if(use_std)
    {
        float **stds = iftGFLIMStdFromFeatureMatrix(X, n, finer_scale);
        for(int i = 0; i < merged->nrows; i++)
        {
            float *stds_i = stds[i];
            for(int j = 0; j < merged->ncols; j++)
            {
                iftMatrixElem(merged, j, i) *= (1 - *stds_i);
                stds_i++;
            }
            stds_i = NULL;
            iftFree(stds[i]);
        }
        iftFree(stds);
    }

    iftMatrix *sum_matrix = X[finer_scale];
    if(X[n-1]->nrows != X[finer_scale]->nrows || X[n-1]->ncols != X[finer_scale]->ncols)
        sum_matrix = iftGFLIMUpsampleFeatureMatrix(X[n-1], nodes_map[n-1], kernels_map[n-1], X[finer_scale]->nrows, X[finer_scale]->ncols);
    for(int i = 0; i < sum_matrix->nrows; i++)
    {
        for(int j = 0; j < sum_matrix->ncols; j++)
        {
            iftMatrixElem(merged, j, i) += iftMatrixElem(sum_matrix, j, i);
        }
    }
    if(X[n-1]->nrows != X[finer_scale]->nrows || X[n-1]->ncols != X[finer_scale]->ncols)
        iftDestroyMatrix(&sum_matrix);
    sum_matrix = NULL;
    return merged;
}

iftMatrix *iftGFLIMMergeMatrices_sum(iftMatrix **X, int n, 
                                        int **nodes_map, iftIntArray **kernels_map, 
                                        int finer_scale, bool use_otsu, bool use_std,
                                        int reinforce_scale)
{
    /*
    Merge two feature matrices by sum
    */
    iftMatrix *merged = iftCopyMatrix(X[finer_scale]);

    if(use_otsu)
    {
        float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(merged);
        iftGFLIMSumValueConditionToFeatureMatrix(merged, otsu_thresholds);
        iftFree(otsu_thresholds);
    }
    
    for(int i = 0; i < n; i++)
    {
        if(i == finer_scale) continue;
        iftMatrix *tmp = iftGFLIMUpsampleFeatureMatrix(X[i], nodes_map[i], kernels_map[i], X[finer_scale]->nrows, X[finer_scale]->ncols);

        if(use_otsu)
        {
            float *otsu_thresholds = iftGFLIMOtsuFromFeatureMatrix(tmp);
            iftGFLIMSumValueConditionToFeatureMatrix(tmp, otsu_thresholds);
            iftFree(otsu_thresholds);
        }

        // sum
        for(int j = 0; j < X[finer_scale]->ncols; j++)
        {
            for(int k = 0; k < X[finer_scale]->nrows; k++)
            {
                iftMatrixElem(merged, j, k) += iftMatrixElem(tmp, j, k);
            }
        }
        iftDestroyMatrix(&tmp);
    }

    if(use_std)
    {
        float **stds = iftGFLIMStdFromFeatureMatrix(X, n, finer_scale);
        for(int i = 0; i < merged->nrows; i++)
        {
            float *stds_i = stds[i];
            for(int j = 0; j < merged->ncols; j++)
            {
                iftMatrixElem(merged, j, i) *= (1 - *stds_i);
                stds_i++;
            }
            stds_i = NULL;
            iftFree(stds[i]);
        }
        iftFree(stds);
    }

    if(reinforce_scale > 0)
    {
        iftMatrix *sum_matrix = X[finer_scale];
        if(X[reinforce_scale]->nrows != X[finer_scale]->nrows || X[reinforce_scale]->ncols != X[finer_scale]->ncols)
            sum_matrix = iftGFLIMUpsampleFeatureMatrix(X[reinforce_scale], nodes_map[reinforce_scale], kernels_map[reinforce_scale], X[finer_scale]->nrows, X[finer_scale]->ncols);
        for(int i = 0; i < sum_matrix->nrows; i++)
        {
            for(int j = 0; j < sum_matrix->ncols; j++)
                iftMatrixElem(merged, j, i) += iftMatrixElem(sum_matrix, j, i);
        }
        if(X[reinforce_scale]->nrows != X[finer_scale]->nrows || X[reinforce_scale]->ncols != X[finer_scale]->ncols)
            iftDestroyMatrix(&sum_matrix);
        sum_matrix = NULL;
    }
    
    return merged;
}


iftIntArray **iftGFLIMCompute_kernel_maps_bkp(char *filename, int *nscales)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
        iftError("Could not read file: %s",  __func__, filename);
    
    int c;
    int nch = 0;
    int size = 256;
    char *buf = malloc(size);
    int nspaces = 0, nlines = 0;
    if(buf == NULL)
        iftError("out of memory", __func__);

    // read the file and reallocate buffer if needed
    while((c = getchar()) != EOF)
    {
        if(nch >= size-1)
        {
            size += 256;
            buf = realloc(buf, size);
            if(buf == NULL)
                iftError("out of memory", __func__);
        }
        if(c == ' ') nspaces++;
        if(c == '\n') nlines++;
        buf[nch++] = c;
    }
    buf[nch++] = '\0';
    fclose(fp);

    // convert the buffer into a matrix of kernels
    int **kernel_maps_scale0 = iftAlloc(nlines, sizeof(int *));
    int pos = 0; // character index in the buffer
    int kernels_count = 0; // kernel index in the current line
    int lines_count = 0; // current line index
    int nkernels0 = nspaces/nlines + 1; // number of kernels in the first line
    kernel_maps_scale0[0] = iftAllocIntArray(nkernels0);
    while(pos < nch && buf[pos] != '\n')
    {
        int end = pos; // end of the current number

        // find the end of the current number
        while(end < nch && buf[end] != '\n' && buf[end] != ' ') end++;
        
        if(buf[end] == ' ') // convert in int
        {
            buf[end] = '\0';
            kernel_maps_scale0[lines_count][kernels_count++] = atoi(&buf[pos]);
            buf[end] = ' ';
        }
        else{ // last element of the line
            if(lines_count < nlines - 1)
            {
                lines_count++;
                kernel_maps_scale0[lines_count] = iftAllocIntArray(nkernels0);
                kernels_count = 0;
            }
        }
        pos = end + 1;
    }
    free(buf);

    iftIntArray **kernel_maps = iftAlloc(nlines, sizeof(iftIntArray *));
    *nscales = nlines;
    // compute the kernel maps for all scales
    int nkernelsi = 0;
    kernel_maps[0] = iftCreateIntArray(nkernels0);
    for(int i=0; i < nkernels0; i++)
    {
        int k0 = kernel_maps_scale0[0][i] - 1;
        if(k0 == i)
        {
            kernel_maps[0]->val[nkernelsi] = nkernelsi;
            nkernelsi++;
        }else
            kernel_maps[0]->val[nkernelsi] = kernel_maps[0]->val[k0];
    }

    for(int l = 1; l < nlines; l++)
    {
        nkernelsi = 0;
        kernel_maps[l] = iftCreateIntArray(nkernels0);
        for(int i=0; i < nkernels0; i++)
        {
            int k0 = kernel_maps_scale0[l][i] - 1;
            int k0_prev = kernel_maps_scale0[l-1][i] - 1;
            if(k0_prev == i)
            {
                if(k0 == k0_prev)
                {
                    kernel_maps[l]->val[nkernelsi] = nkernelsi;
                    nkernelsi++;
                }else{
                    int j = 0, k = k0;
                    while(j < l)
                    {
                        k = kernel_maps[j]->val[k];
                        j++;
                    }
                    k = kernel_maps[l]->val[k];
                    kernel_maps[l]->val[nkernelsi] = k;
                    nkernelsi++;
                }
            }
        }
        iftResizeIntArray(&(kernel_maps[l]), nkernelsi);
    }
    for(int l = 0; l < nlines; l++)
        iftFree(kernel_maps_scale0[l]);
    iftFree(kernel_maps_scale0);

    return kernel_maps;
}

char* read_file(const char* filename) { 
    FILE* file = fopen(filename, "r"); 
    if (!file) { 
        perror("Failed to open file"); 
        return NULL; 
    } 
 
    // Allocate initial memory for the file content 
    size_t capacity = 1024; 
    char* content = malloc(capacity); 
    if (!content) { 
        perror("Failed to allocate memory"); 
        fclose(file); 
        return NULL; 
    } 
 
    size_t length = 0; 
    size_t bytesRead; 
 
    // Read the file in chunks 
    while ((bytesRead = fread(content + length, sizeof(char), 1024, file)) > 0) { 
        length += bytesRead; 
 
        // If we reach capacity, resize the buffer 
        if (length + 1024 > capacity) { 
            capacity *= 2; // Double the capacity 
            content = realloc(content, capacity); 
            if (!content) { 
                perror("Failed to reallocate memory"); 
                fclose(file); 
                return NULL; 
            } 
        } 
    } 
 
    // Null-terminate the string 
    if (length < capacity) { 
        content[length] = '\0'; 
    } else { 
        content = realloc(content, length + 1); 
        content[length] = '\0'; 
    } 
 
    fclose(file); 
    return content; 
} 

iftIntArray **iftGFLIMCompute_kernel_maps(char *filename)
{
    
    char *buf = read_file(filename);

    int nlines = 100, line_count = 0;
    const char line_delimiter[] = "\n";
    const char number_delimiter[] = " ";
    char *line, *endptr; 
    int nkernels0 = 100, kernel_count = 0;

    char* line_saveptr = NULL;
    char* number_saveptr = NULL;

    int **kernel_maps_scale0 = iftAlloc(nlines, sizeof(int *));

    // Get the first line 
    line = strtok_r(buf, line_delimiter, &line_saveptr); 

    while(line != NULL)
    {
        // break into numbers
        char* inner_token = strtok_r(line, number_delimiter, &number_saveptr);

        // allocate memory for more lines if needed
        if(line_count >= nlines)
        {
            nlines += 100;
            kernel_maps_scale0 = realloc(kernel_maps_scale0, nlines * sizeof(int *));
            if(kernel_maps_scale0 == NULL)
                iftError("out of memory", __func__);
        }

        kernel_maps_scale0[line_count] = iftAllocIntArray(nkernels0);
        kernel_count = 0;

        // get each number
        while (inner_token != NULL) {

            // allocate memory for more kernels if needed
            if(kernel_count >= nkernels0)
            {
                nkernels0 += 100;
                kernel_maps_scale0[line_count] = realloc(kernel_maps_scale0[line_count], nkernels0 * sizeof(int));
                if(kernel_maps_scale0[line_count] == NULL)
                    iftError("out of memory", __func__);
            }
            kernel_maps_scale0[line_count][kernel_count] = (int) strtol(inner_token,&endptr,10);
            kernel_count++;
            inner_token = strtok_r(NULL, number_delimiter, &number_saveptr);
        }
        if(line_count == 0)
            nkernels0 = kernel_count;
        line_count++;
        line = strtok_r(NULL, line_delimiter, &line_saveptr);
    }
    nlines = line_count;
    free(buf);

    iftIntArray **kernel_maps = iftAlloc(nlines, sizeof(iftIntArray *));
    
    // compute the kernel maps for all scales
    int nkernelsi = 0;
    kernel_maps[0] = iftCreateIntArray(nkernels0);
    for(int i=0; i < nkernels0; i++)
    {
        int k0 = kernel_maps_scale0[0][i] - 1;
        if(k0 == i)
        {
            kernel_maps[0]->val[nkernelsi] = nkernelsi;
            nkernelsi++;
        }else
            kernel_maps[0]->val[nkernelsi] = kernel_maps[0]->val[k0];
    }

    for(int l = 1; l < nlines; l++)
    {
        nkernelsi = 0;
        kernel_maps[l] = iftCreateIntArray(nkernels0);
        for(int i=0; i < nkernels0; i++)
        {
            int k0 = kernel_maps_scale0[l][i] - 1;
            int k0_prev = kernel_maps_scale0[l-1][i] - 1;
            if(k0_prev == i)
            {
                if(k0 == k0_prev)
                {
                    kernel_maps[l]->val[nkernelsi] = nkernelsi;
                    nkernelsi++;
                }else{
                    int j = 0, k = k0;
                    while(j < l)
                    {
                        k = kernel_maps[j]->val[k];
                        j++;
                    }
                    k = kernel_maps[l]->val[k];
                    kernel_maps[l]->val[nkernelsi] = k;
                    nkernelsi++;
                }
            }
        }
        iftResizeIntArray(&(kernel_maps[l]), nkernelsi);
    }
    for(int l = 0; l < nlines; l++)
        iftFree(kernel_maps_scale0[l]);
    iftFree(kernel_maps_scale0);

    return kernel_maps;
}


void iftGFLIMMaxNormGraph(iftGFLIMGraph *graph)
{
    // Compute max values for normalization
    float *max_features = iftAllocFloatArray(graph->num_feats);
    for(int j = 0; j < graph->num_feats; j++)
        max_features[j] = graph->nodes[0].feats[j];
    
    for(int i = 1; i < graph->num_nodes; i++)
    {
        for(int j = 0; j < graph->num_feats; j++)
        {
            if(graph->nodes[i].feats[j] > max_features[j])
                max_features[j] = graph->nodes[i].feats[j];
        }
    }

    // Normalize each feature to [0,1] with a max normalization
    for(int i = 0; i < graph->num_nodes; i++)
    {
        for(int j = 0; j < graph->num_feats; j++)
        {
            if(max_features[j] > 0)
                graph->nodes[i].feats[j] /= max_features[j];
            else
                graph->nodes[i].feats[j] = 0;
        }
    }
    iftFree(max_features);
}

iftIntArray *iftGFLIMMergeKernelScales(iftIntArray **kernel_maps, int nscales, 
                                    int init, int end)
{
    if(end < init) iftError("end must be greater than or equal to init", __func__);
    if(end > nscales) iftError("end must be less than nscales", __func__);
    
    iftIntArray *merged_map = NULL;

    if(init == end)
    {
        if(init < nscales)
        {
            merged_map = iftCreateIntArray(kernel_maps[init]->n);
            for(int k = 0; k < kernel_maps[init]->n; k++)
                merged_map->val[k] = k; //kernel_maps[init]->val[k];
            return merged_map;
        }
        else
            iftError("init must be less than nscales", __func__);
    }

    merged_map = iftCreateIntArray(kernel_maps[init]->n);
    for(int k = 0; k < kernel_maps[init]->n; k++)
        merged_map->val[k] = kernel_maps[init]->val[k];

    for(int i = init+1; i < end; i++)
    {
        for(int j = 0; j < merged_map->n; j++)
        {
            int k = merged_map->val[j];
            merged_map->val[j] = kernel_maps[i]->val[k];
        }   
    }
    return merged_map;
}


int *iftGFLIMComputeNodeMap(iftImage *spx_imgi, iftImage *spx_img0){
    /*
    Compute a mapping from the nodes of spx_img0 to the nodes of spx_imgi.
    */
    int max_spx = iftMaximumValue(spx_img0);
    int *node_map = iftAllocIntArray(max_spx);

    for(int p = 0; p < spx_img0->n; p++)
    {
        int label = spx_img0->val[p] - 1;
        if(label < 0)
            iftError("spx_img0 has a label %d at voxel %d", __func__, spx_img0->val[p], p);
        node_map[label] = spx_imgi->val[p] - 1;
        if(node_map[label] < 0)
            iftError("spx_imgi has a label %d at voxel %d", __func__, spx_imgi->val[p], p);
    }
    return node_map;
}

void iftGFLIMSkipConnection(iftGFLIMArch *arch, int layer, char *kernels_file, 
                            char *model_dir, iftFileSet *fs)
{
    // get the layers (from 1) that will be used in the skip connection of the current layer
    int *skip_layers = iftAllocIntArray(layer);
    int nskip = 0;
    for(int l = 1; l < layer; l++){
        if(arch->layer[layer-1].skip_connection[l-1] > 0)
        {
            skip_layers[nskip] = l;
            nskip++;
        }   
    }
    
    // if there is no skip connection, return
    if(nskip == 0)
    {
        iftFree(skip_layers);
        return;
    }

    // get the finer residual layer (with more nodes) to be used as reference in the upsampling
    int *finer_layers = iftGFLIMGetFinerResidualLayer(arch);

    int finer_scale_index = 0; // this index refers to the matrices, kernel_maps, and load maps loaded

    // compute the kernel maps for the layers that will be used in the skip connection
    iftIntArray **kernels_maps_all = iftGFLIMCompute_kernel_maps(kernels_file); // compute the kernel maps for all layers
    iftIntArray **kernels_maps = iftAlloc(nskip+1, sizeof(iftIntArray *));
    for(int j = 0; j < nskip; j++)
    {
        // the layer j may have skip connections to layers finer than itself, so we need to find its finer scale and map it from finer_layer
        int i = skip_layers[j];
        kernels_maps[j] = iftGFLIMMergeKernelScales(kernels_maps_all, layer, finer_layers[layer-1]-1, finer_layers[i-1]-1);
        if(finer_scale_index == 0 && finer_layers[layer-1] == finer_layers[i-1]) // if the skip layer is the finer layer, store its index
            finer_scale_index = j;
    }
    // create a mapping from the kernels of the finer layer to those of the current layer
    kernels_maps[nskip] = iftGFLIMMergeKernelScales(kernels_maps_all, layer, finer_layers[layer-1]-1, layer-1);
    for(int i = 0; i < arch->nlayers-1; i++)
        iftDestroyIntArray(&kernels_maps_all[i]);
    iftFree(kernels_maps_all);

    for (int file_id = 0; file_id < fs->n; file_id++)
    {
        char ext[10], filename[512];
        sprintf(ext, "%s", iftFileExt(fs->files[file_id]->path));
        char *basename = iftFilename(fs->files[file_id]->path, ext);
        
        sprintf(filename, "%s/labels%d/%s.pgm", model_dir, finer_layers[layer-1], basename);
        iftImage *spx_img0 = iftReadImageByExt(filename);

        int **node_maps = iftAlloc(nskip+1, sizeof(int *));
        iftMatrix **matrices = iftAlloc(nskip+1, sizeof(iftMatrix *));

        for(int j = 0; j < nskip; j++)
        {
            int i = skip_layers[j];
            sprintf(filename, "%s/layer%d/%s.json", model_dir, i, basename);
            iftGFLIMGraph *graph_skip = iftReadGFLIMGraph(filename);
            iftGFLIMMaxNormGraph(graph_skip);
            matrices[j] = iftGFLIMGraphToNodeFeaturesMatrix(graph_skip);
            iftDestroyGFLIMGraph(&graph_skip);

            sprintf(filename, "%s/labels%d/%s.pgm", model_dir, i, basename);
            iftImage *spx_img = iftReadImageByExt(filename);

            node_maps[j] = iftGFLIMComputeNodeMap(spx_img, spx_img0);
            iftDestroyImage(&spx_img);
        }
        sprintf(filename, "%s/layer%d/%s.json", model_dir, layer, basename);
        iftGFLIMGraph *graph_skip = iftReadGFLIMGraph(filename);
        iftGFLIMMaxNormGraph(graph_skip);
        matrices[nskip] = iftGFLIMGraphToNodeFeaturesMatrix(graph_skip);
        iftDestroyGFLIMGraph(&graph_skip);

        sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer, basename);
        iftImage *spx_img = iftReadImageByExt(filename);
        node_maps[nskip] = iftGFLIMComputeNodeMap(spx_img, spx_img0);

        iftDestroyImage(&spx_img);

        // merge the feature matrices
        iftMatrix *merged = NULL;
        if(strcmp(arch->layer[layer-1].skip_type, "dotprod") == 0)
            merged = iftGFLIMMergeMatrices_dotProd(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type, "dotprod_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScale0(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type, "dotprod_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScalei(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, -1);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, 0);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, nskip);
        else
            iftError("Unknown merge function: %s",  __func__, arch->layer[layer-1].skip_type);

        for(int j = 0; j < nskip+1; j++)
        {
            iftDestroyMatrix(&matrices[j]);
            iftFree(node_maps[j]);
        }
        iftFree(matrices);
        iftFree(node_maps);

        // copy the merged features to the graph of the current layer
        sprintf(filename, "%s/layer%d/%s.json", model_dir, finer_layers[layer-1], basename);
        iftGFLIMGraph *graph_finer = iftReadGFLIMGraph(filename);
        if(graph_finer->num_nodes != merged->nrows || graph_finer->num_feats != merged->ncols)
            iftError("Graph and merged matrix dimensions do not match: (%ld, %ld) != (%d, %d)", __func__, 
                    graph_finer->num_nodes, graph_finer->num_feats, merged->nrows, merged->ncols); 
        for(int i = 0; i < graph_finer->num_nodes; i++)
        {
            for(int j = 0; j < graph_finer->num_feats; j++)
                graph_finer->nodes[i].feats[j] = iftMatrixElem(merged, j, i);
        }
        // min-max normalization per channel
        /*for(int j = 0; j < graph_finer->num_feats; j++)
        {
            float min = iftMatrixElem(merged, j, 0);
            float max = iftMatrixElem(merged, j, 0);
            for(int i = 1; i < graph_finer->num_nodes; i++)
            {
                if(iftMatrixElem(merged, j, i) < min)
                    min = iftMatrixElem(merged, j, i);
                if(iftMatrixElem(merged, j, i) > max)
                    max = iftMatrixElem(merged, j, i);
            }
            for(int i = 0; i < graph_finer->num_nodes; i++)
                graph_finer->nodes[i].feats[j] = (iftMatrixElem(merged, j, i) - min) / (max - min);
        }*/
        iftDestroyMatrix(&merged);

        // save the graph with the skip connections
        sprintf(filename, "%s/layer%d/%s.json", model_dir, layer, basename);
        iftWriteGFLIMGraph(graph_finer, filename);
        iftDestroyGFLIMGraph(&graph_finer);

        // save the superpixel image of the finer layer
		sprintf(filename, "%s/labels%d/%s.pgm", model_dir, layer, basename);
		iftWriteImageByExt(spx_img0, filename);
        iftDestroyImage(&spx_img0);

        iftFree(basename);
   }
    iftFree(skip_layers);
    for(int j = 0; j < nskip+1; j++)
        iftDestroyIntArray(&kernels_maps[j]);
    iftFree(kernels_maps);
    iftFree(finer_layers);
}

void iftGFLIMSkipConnectionBatch(iftGFLIMGraphArray *input, iftImageArray *label_imgs,
                            iftFileSet *fs, int first, int last, int layer, 
                            iftGFLIMArch *arch, char *kernels_file, 
                            char *model_dir)
{
    // get the layers that will be used in the skip connection of the current layer
    int *skip_layers = iftAllocIntArray(layer);
    int nskip = 0;
    for(int l = 1; l < layer; l++){
        if(arch->layer[layer-1].skip_connection[l-1] > 0)
        {
            skip_layers[nskip] = l;
            nskip++;
        }   
    }
    // if there is no skip connection, return
    if(nskip == 0)
    {
        iftFree(skip_layers);
        return;
    }

    // get the finer residual layer (with more nodes) to be used as reference in the upsampling
    int *finer_layers = iftGFLIMGetFinerResidualLayer(arch);

    int finer_scale_index = 0; // this index refers to the matrices, kernel_maps, and load maps loaded

    // compute the kernel maps for the layers that will be used in the skip connection
    iftIntArray **kernels_maps_all = iftGFLIMCompute_kernel_maps(kernels_file); // compute the kernel maps for all layers
    iftIntArray **kernels_maps = iftAlloc(nskip+1, sizeof(iftIntArray *));
    for(int j = 0; j < nskip; j++)
    {
        // the layer j may have skip connections to layers finer than itself, so we need to find its finer scale and map it from finer_layer
        int i = skip_layers[j];
        kernels_maps[j] = iftGFLIMMergeKernelScales(kernels_maps_all, layer, finer_layers[layer-1]-1, finer_layers[i-1]-1);
        if(finer_scale_index == 0 && finer_layers[layer-1] == finer_layers[i-1]) // if the skip layer is the finer layer, store its index
            finer_scale_index = j;
    }
    // create a mapping from the kernels of the finer layer to those of the current layer
    kernels_maps[nskip] = iftGFLIMMergeKernelScales(kernels_maps_all, layer, finer_layers[layer-1]-1, layer-1);
    for(int i = 0; i < arch->nlayers-1; i++)
        iftDestroyIntArray(&kernels_maps_all[i]);
    iftFree(kernels_maps_all);

    for (int k = 0; k <= last - first; k++)
    {
        int file_id = k + first;
        char ext[10], filename[512];
        sprintf(ext, "%s", iftFileExt(fs->files[file_id]->path));
        char *basename = iftFilename(fs->files[file_id]->path, ext);

        sprintf(filename, "%s/labels%d/%s.pgm", model_dir, finer_layers[layer-1], basename);
        iftImage *spx_img0 = iftReadImageByExt(filename);

        int **node_maps = iftAlloc(nskip+1, sizeof(int *));
        iftMatrix **matrices = iftAlloc(nskip+1, sizeof(iftMatrix *));

        for(int j = 0; j < nskip; j++)
        {
            int i = skip_layers[j];
            sprintf(filename, "%s/layer%d/%s.json", model_dir, i, basename);
            iftGFLIMGraph *graph_skip = iftReadGFLIMGraph(filename);
            iftGFLIMMaxNormGraph(graph_skip);
            matrices[j] = iftGFLIMGraphToNodeFeaturesMatrix(graph_skip);
            iftDestroyGFLIMGraph(&graph_skip);

            sprintf(filename, "%s/labels%d/%s.pgm", model_dir, i, basename);
            iftImage *spx_img = iftReadImageByExt(filename);
            
            node_maps[j] = iftGFLIMComputeNodeMap(spx_img, spx_img0);
            iftDestroyImage(&spx_img);
        }
        iftGFLIMMaxNormGraph(input->val[k]);
        matrices[nskip] = iftGFLIMGraphToNodeFeaturesMatrix(input->val[k]);
        node_maps[nskip] = iftGFLIMComputeNodeMap(label_imgs->val[k], spx_img0);
        iftDestroyImage(&(label_imgs->val[k]));
        label_imgs->val[k] = spx_img0;
        spx_img0 = NULL;

        // merge the feature matrices
        iftMatrix *merged = NULL;
        if(strcmp(arch->layer[layer-1].skip_type, "dotprod") == 0)
            merged = iftGFLIMMergeMatrices_dotProd(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type, "dotprod_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScale0(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type , "dotprod_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_dotProd_sumScalei(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, -1);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum_sumscale0") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, 0);
        else if(strcmp(arch->layer[layer-1].skip_type, "sum_sumscalei") == 0)
            merged = iftGFLIMMergeMatrices_sum(matrices, nskip+1, node_maps, kernels_maps, finer_scale_index, false, false, nskip);
        else
            iftError("Unknown merge function: %s", __func__, arch->layer[layer-1].skip_type);

        for(int j = 0; j < nskip+1; j++)
        {
            iftDestroyMatrix(&matrices[j]);
            iftFree(node_maps[j]);
        }
        iftFree(matrices);
        iftFree(node_maps);

        // copy the merged features to the graph of the current layer
        sprintf(filename, "%s/layer%d/%s.json", model_dir, finer_layers[layer-1], basename);
        iftDestroyGFLIMGraph(&(input->val[k]));
        iftGFLIMGraph *graph_finer = iftReadGFLIMGraph(filename);
        if(graph_finer->num_nodes != merged->nrows || graph_finer->num_feats != merged->ncols)
            iftError("Graph and merged matrix dimensions do not match: (%ld, %ld) != (%d, %d)", __func__, 
                    graph_finer->num_nodes, graph_finer->num_feats, merged->nrows, merged->ncols); 
        for(int i = 0; i < graph_finer->num_nodes; i++)
        {
            for(int j = 0; j < graph_finer->num_feats; j++)
                graph_finer->nodes[i].feats[j] = iftMatrixElem(merged, j, i);
        }
        // min-max normalization per channel
        /*for(int j = 0; j < graph_finer->num_feats; j++)
        {
            float min = iftMatrixElem(merged, j, 0);
            float max = iftMatrixElem(merged, j, 0);
            for(int i = 1; i < graph_finer->num_nodes; i++)
            {
                if(iftMatrixElem(merged, j, i) < min)
                    min = iftMatrixElem(merged, j, i);
                if(iftMatrixElem(merged, j, i) > max)
                    max = iftMatrixElem(merged, j, i);
            }
            for(int i = 0; i < graph_finer->num_nodes; i++)
                graph_finer->nodes[i].feats[j] = (iftMatrixElem(merged, j, i) - min) / (max - min);
        }*/
        iftDestroyMatrix(&merged);
        input->val[k] = graph_finer;
        graph_finer = NULL;
        iftFree(basename);
    }
    iftFree(skip_layers);
    for(int j = 0; j < nskip+1; j++)
        iftDestroyIntArray(&kernels_maps[j]);
    iftFree(kernels_maps);
    iftFree(finer_layers);
}



// ============ FUNCTIONS TO DECODE FEATURES ============ //

iftFImage *TestAdaptativeDecoder2(iftGFLIMGraph *graph, int *channel_label, int label)
{
    float *weight = iftAllocFloatArray(graph->num_feats);
    float *mean = iftAllocFloatArray(graph->num_feats);

    int num_pixels = 0;
    /* Estimate the mean activation per channel */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
            mean[b] += graph->nodes[p].feats[b] * graph->nodes[p].numPixels;
        num_pixels += graph->nodes[p].numPixels;
    }

    // Just so it is easier to compute the otsu
    iftImage *means_image = iftCreateImage(graph->num_feats, 1, 1);
    float max_mean = 0.0;
    for (int b = 0; b < graph->num_feats; b++)
    {
        mean[b] /= (float)num_pixels;
        means_image->val[b] = mean[b] * 255;
        if (means_image->val[b] > max_mean)
            max_mean = means_image->val[b];
    }

    if (max_mean > 0.0)
    {
        // Compute otsu threshold as a way to separate foreground bands from background bands
        float mu = 0.0, stdev = 0.0;
        for (int b = 0; b < graph->num_feats; b++)
            mu += mean[b];
        mu /= graph->num_feats;
        int means_otsu = iftOtsu(means_image); 

        // Compute the standard deviation
        for (int b = 0; b < graph->num_feats; b++)
            stdev += (mean[b] - mu) * (mean[b] - mu);
        stdev = sqrtf(stdev / graph->num_feats) * sqrtf(stdev / graph->num_feats);

        // Compute the low and high threshold values
        float low_end, high_end;
        if (means_otsu > 0.1)
            low_end = means_otsu - stdev;
        else
            low_end = means_otsu;
        high_end = means_otsu + stdev;

        /* Estimate the weight of each channel */
        //iftImage *band = iftCreateImage(graph->num_nodes, 1, 1);
        iftImage *band = iftCreateImage(num_pixels, 1, 1);
        for (int b = 0; b < graph->num_feats; b++)
        {
            // Get one activation (a channel/band from mimg) and convert it from 0-1 to 0-255
            for (int p = 0, x = 0; p < graph->num_nodes; p++) {
                float feat = graph->nodes[p].feats[b];
                for (int q = 0; q < graph->nodes[p].numPixels; q++, x++){
                    band->val[x] = feat * 255;
                }
            }

            // find the maximum value of the band
            int max_band = iftMaximumValue(band); // the difference to the img version of this decoder is here. The otsu here consider node values, i.e, the nodes' sizes are not considered.

            if (max_band > 0.0)
            {
                // Compute another otsu to separate foreground pixels from background pixels
                int otsu_band = iftOtsu(band);

                // Compute the foreground proportion of the band 
                // as the proportion of pixels above otsu_band
                float fg_proportion = 0;
                for (int p = 0; p < band->n; p++)
                {
                    if (band->val[p] >= otsu_band) 
                        fg_proportion += 1;//graph->nodes[p].numPixels;
                }
                fg_proportion /= num_pixels;
                
                // weight the band according to the high/low threshold and the foreground proportion
                if (/*mean[b] >= high_end &&*/ fg_proportion > 0.2 && channel_label[b] == 1)
                {
                    if(label != 0) weight[b] = -1.0;
                    else weight[b] = 1.0;
                }
                else if (/*mean[b] <= low_end &&*/ fg_proportion < 0.1 && channel_label[b] != 1)
                {
                    if(label != 0) weight[b] = 1.0;
                    else weight[b] = -1.0;
                }
                else
                {
                    weight[b] = 0.0;
                }
            }
            else
            {
                weight[b] = 0.0;
            }
        }
        iftDestroyImage(&band);
    }
    else
    {
        for (int b = 0; b < graph->num_feats; b++)
            weight[b] = 0.0;
    }

    iftFree(mean);
    iftDestroyImage(&means_image);

    iftFImage *salie = iftCreateFImage(graph->num_nodes, 1, 1);

    /* decode layer */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
        {
            //if (channel_label[b] != -1) // only foreground channels are weigthed
            salie->val[p] += graph->nodes[p].feats[b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }

    iftFree(weight);

    return (salie);
}


iftFImage *TestAdaptativeDecoder(iftGFLIMGraph *graph, int *channel_label, int label)
{
    float *weight = iftAllocFloatArray(graph->num_feats);
    float *mean = iftAllocFloatArray(graph->num_feats);

    int num_pixels = 0;
    /* Estimate the mean activation per channel */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
            mean[b] += graph->nodes[p].feats[b] * graph->nodes[p].numPixels;
        num_pixels += graph->nodes[p].numPixels;
    }

    // Just so it is easier to compute the otsu
    iftImage *means_image = iftCreateImage(graph->num_feats, 1, 1);
    float max_mean = 0.0;
    for (int b = 0; b < graph->num_feats; b++)
    {
        mean[b] /= (float)num_pixels;
        means_image->val[b] = mean[b] * 255;
        if (means_image->val[b] > max_mean)
            max_mean = means_image->val[b];
    }

    if (max_mean > 0.0)
    {
        // Compute otsu threshold as a way to separate foreground bands from background bands
        float mu = 0.0, stdev = 0.0;
        for (int b = 0; b < graph->num_feats; b++)
            mu += mean[b];
        mu /= graph->num_feats;
        int means_otsu = iftOtsu(means_image); 

        // Compute the standard deviation
        for (int b = 0; b < graph->num_feats; b++)
            stdev += (mean[b] - mu) * (mean[b] - mu);
        stdev = sqrtf(stdev / graph->num_feats) * sqrtf(stdev / graph->num_feats);

        // Compute the low and high threshold values
        float low_end, high_end;
        if (means_otsu > 0.1)
            low_end = means_otsu - stdev;
        else
            low_end = means_otsu;
        high_end = means_otsu + stdev;

        /* Estimate the weight of each channel */
        //iftImage *band = iftCreateImage(graph->num_nodes, 1, 1);
        iftImage *band = iftCreateImage(num_pixels, 1, 1);
        for (int b = 0; b < graph->num_feats; b++)
        {
            // Get one activation (a channel/band from mimg) and convert it from 0-1 to 0-255
            for (int p = 0, x = 0; p < graph->num_nodes; p++) {
                float feat = graph->nodes[p].feats[b];
                for (int q = 0; q < graph->nodes[p].numPixels; q++, x++){
                    band->val[x] = feat * 255;
                }
            }

            // find the maximum value of the band
            int max_band = iftMaximumValue(band); // the difference to the img version of this decoder is here. The otsu here consider node values, i.e, the nodes' sizes are not considered.

            if (max_band > 0.0)
            {
                // Compute another otsu to separate foreground pixels from background pixels
                int otsu_band = iftOtsu(band);

                // Compute the foreground proportion of the band 
                // as the proportion of pixels above otsu_band
                float fg_proportion = 0;
                for (int p = 0; p < band->n; p++)
                {
                    if (band->val[p] >= otsu_band) 
                        fg_proportion += 1;//graph->nodes[p].numPixels;
                }
                fg_proportion /= num_pixels;
                
                // weight the band according to the high/low threshold and the foreground proportion
                if (/*mean[b] >= high_end &&*/ fg_proportion > 0.2 && channel_label[b] == 1)
                {
                    if(label != 0) weight[b] = -1.0;
                    else weight[b] = 1.0;
                }
                else if (/*mean[b] <= low_end &&*/ fg_proportion < 0.1 && channel_label[b] != 1)
                {
                    if(label != 0) weight[b] = 1.0;
                    else weight[b] = -1.0;
                }
                else
                {
                    weight[b] = 0.0;
                }
            }
            else
            {
                weight[b] = 0.0;
            }
        }
        iftDestroyImage(&band);
    }
    else
    {
        for (int b = 0; b < graph->num_feats; b++)
            weight[b] = 0.0;
    }

    iftFree(mean);
    iftDestroyImage(&means_image);

    iftFImage *salie = iftCreateFImage(graph->num_nodes, 1, 1);

    /* decode layer */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
        {
            //if (channel_label[b] != -1) // only foreground channels are weigthed
            salie->val[p] += graph->nodes[p].feats[b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }

    iftFree(weight);

    return (salie);
}


/**
 * @brief 
 */
iftFImage *HybridAdaptativeDecoderGraph(iftGFLIMGraph *graph, float *channel_label, int label)
{
    float *weight = iftAllocFloatArray(graph->num_feats);
    float *mean = iftAllocFloatArray(graph->num_feats);

    int num_pixels = 0;
    /* Estimate the mean activation per channel */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
            mean[b] += graph->nodes[p].feats[b] * graph->nodes[p].numPixels;
        num_pixels += graph->nodes[p].numPixels;
    }

    // Just so it is easier to compute the otsu
    iftImage *means_image = iftCreateImage(graph->num_feats, 1, 1);
    float max_mean = 0.0;
    for (int b = 0; b < graph->num_feats; b++)
    {
        mean[b] /= (float)num_pixels;
        means_image->val[b] = mean[b] * 255;
        if (means_image->val[b] > max_mean)
            max_mean = means_image->val[b];
    }

    if (max_mean > 0.0)
    {
        // Compute otsu threshold as a way to separate foreground bands from background bands
        float mu = 0.0, stdev = 0.0;
        for (int b = 0; b < graph->num_feats; b++)
            mu += mean[b];
        mu /= graph->num_feats;
        int means_otsu = iftOtsu(means_image); 

        // Compute the standard deviation
        for (int b = 0; b < graph->num_feats; b++)
            stdev += (mean[b] - mu) * (mean[b] - mu);
        stdev = sqrtf(stdev / graph->num_feats) * sqrtf(stdev / graph->num_feats);

        // Compute the low and high threshold values
        float low_end, high_end;
        if (means_otsu > 0.1)
            low_end = means_otsu - stdev;
        else
            low_end = means_otsu;
        high_end = means_otsu + stdev;

        /*printf("low_end: %f, high_end: %f, means_otsu: %d, stdev: %f, mu: %f \n", 
            low_end, high_end, means_otsu, stdev, mu);*/

        /* Estimate the weight of each channel */
        //iftImage *band = iftCreateImage(graph->num_nodes, 1, 1);
        iftImage *band = iftCreateImage(num_pixels, 1, 1);
        for (int b = 0; b < graph->num_feats; b++)
        {
            // Get one activation (a channel/band from mimg) and convert it from 0-1 to 0-255
            for (int p = 0, x = 0; p < graph->num_nodes; p++) {
                float feat = graph->nodes[p].feats[b];
                for (int q = 0; q < graph->nodes[p].numPixels; q++, x++){
                    band->val[x] = feat * 255;
                }
            }

            // find the maximum value of the band
            int max_band = iftMaximumValue(band); // the difference to the img version of this decoder is here. The otsu here consider node values, i.e, the nodes' sizes are not considered.

            if (max_band > 0.0)
            {
                // Compute another otsu to separate foreground pixels from background pixels
                int otsu_band = iftOtsu(band);

                // Compute the foreground proportion of the band 
                // as the proportion of pixels above otsu_band
                float fg_proportion = 0;
                for (int p = 0; p < band->n; p++)
                {
                    if (band->val[p] >= otsu_band) 
                        fg_proportion += 1;//graph->nodes[p].numPixels;
                }
                fg_proportion /= num_pixels;

                // weight the band according to the high/low threshold and the foreground proportion
                if (mean[b] >= high_end && fg_proportion > 0.2)
                {
                    weight[b] = -1.0;
                }
                else if (mean[b] <= low_end && fg_proportion < 0.1)
                {
                    weight[b] = +1.0;
                }
                else
                {
                    weight[b] = 0.0;
                }
            }
            else
            {
                weight[b] = 0.0;
            }
        }
        iftDestroyImage(&band);
    }
    else
    {
        for (int b = 0; b < graph->num_feats; b++)
            weight[b] = 0.0;
    }

    iftFree(mean);
    iftDestroyImage(&means_image);

    iftFImage *salie = iftCreateFImage(graph->num_nodes, 1, 1);

    /* decode layer */
    for (int p = 0; p < graph->num_nodes; p++)
    {
        for (int b = 0; b < graph->num_feats; b++)
        {
            if (channel_label[b] != -1) // only foreground channels are weigthed
                salie->val[p] += graph->nodes[p].feats[b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }

    iftFree(weight);

    return (salie);
}


/**
 * @brief Decode the MImage using the hybrid adaptive decoder
 * @param mimg MImage to decode
 * @param channel_label Array with the channel labels
 * @return iftFImage* Decoded image
 */
iftFImage *HybridAdaptativeDecoder(iftMImage *mimg, float *channel_label)
{
    float *weight = iftGFLIMTriStateAdaptativeWeigth(mimg); // get the same weigths as the tri-state decoder

    iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

    /* decode layer */
    for (int p = 0; p < mimg->n; p++)
    {
        for (int b = 0; b < mimg->m; b++)
        {
            if (channel_label[b] != -1) // only foreground channels are weigthed
                salie->val[p] += mimg->val[p][b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }

    iftFree(weight);

    return (salie);
}

/**
 * @fn iftGFLIMGaussianDecoder
 * @return The decoded graph.
 */
iftFImage *iftGFLIMGaussianDecoder(iftGFLIMGraph *graph, int *channel_label)
{

    iftFImage *salie = iftCreateFImage(graph->num_nodes, 1, 1);
    float *weight = iftAllocFloatArray(graph->num_feats);
    int nobj = 0, nbkg = 0;

    for (int b = 0; b < graph->num_feats; b++)
    {
        if (channel_label[b] > 1)
        { /* object channel */
            nobj++;
        }
        else
        { /* background channel */
            nbkg++;
        }
    }

    /* Estimate saliency of each pixel */

    for (int p = 0; p < graph->num_nodes; p++)
    {
        /* estimate mean and stdev of activations across channels */

        float mean[2] = {0.0, 0.0}, var[2] = {0.0, 0.0};
        
        iftSet *adj = graph->nodes[p].adj;
        float sum_size = 0;
        float sum_feats[2] = {0.0, 0.0};
        float sum_feats_2[2] = {0.0, 0.0};

        while(adj != NULL)
        {
            double *feats = graph->nodes[adj->elem].feats;
            float size = (float)graph->nodes[adj->elem].numPixels;
            sum_size += size;

            for (int b = 0; b < graph->num_feats; b++)
            {
                if (channel_label[b] == 1)
                { /* background channel */
                    sum_feats[1] += feats[b] * size;
                    sum_feats_2[1] += pow(feats[b] * size, 2);
                }
                else
                { /* object channel */
                    sum_feats[0] += feats[b] * size;
                    sum_feats_2[0] += pow(feats[b] * size, 2);
                }
            }
            adj = adj->next;
        }

        float neighb_size = graph->nodes[p].numNeighbors * sum_size;

        mean[0] = sum_feats[0] / (nobj * neighb_size);
        mean[1] = sum_feats[1] / (nbkg * neighb_size);
        
        var[0] = (sum_feats_2[0] / (nobj * neighb_size)) - pow(mean[0], 2);
        var[1] = (sum_feats_2[1] / (nbkg * neighb_size)) - pow(mean[1], 2);

        /* It is expected two Gaussian distributions G0 and G1 of
           activations around mean[0] and mean[1]. We may expect that
           object pixels generate mean[0] >> mean[1] and the other way
           around for background pixels. We may use the the probability
           values of these distributions to estimate weights. */

        float P0, P1;
        double *feats = graph->nodes[p].feats;
        for (int b = 0; b < graph->num_feats; b++)
        {
            // obj
            if (var[0] <= IFT_EPSILON) var[0] = 1.0;
            P0 = exp(-powf(feats[b] - mean[0], 2) / (2.0 * var[0]));
            
            // bkg
            if (var[1] <= IFT_EPSILON) var[1] = 1.0;
            P1 = exp(-powf(feats[b] - mean[1], 2) / (2.0 * var[1]));

            if ((channel_label[b] > 1) && (P0 > P1))
            {
                weight[b] = 1; // P0;
            }
            else
            {
                if ((channel_label[b] == 1) && (P0 < P1))
                {
                    weight[b] = -1; // P1
                }
                else
                {
                    weight[b] = 0;
                }
            }
        }

        /* Estimate the salience map using one neuron per pixel -- it is
           no longer a point-wise convolution */

        for (int b = 0; b < graph->num_feats; b++)
        {
            salie->val[p] += feats[b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }
    iftFree(weight);

    return (salie);
}


/**
 * @fn iftGFLIMGaussianDecoder
 * @brief Decode the features using a Gaussian decoder.
 * @param mimg The input mimage.
 * @param adj_radius The adjacency radius.
 * @param channel_label An array with the kernel labels.
 * @return The decoded image.
 */
iftFImage *iftGFLIMGaussianDecoder_bkp(iftMImage *mimg, float adj_radius, float *channel_label)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (mimg == NULL)
        printf("\t mimg: NULL \n");
    else
        printf("\t mimg size: %d %d %d %ld \n", mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
    printf("\t adj_radius: %f\n", adj_radius);
    printf("\t channel_label: %s\n", channel_label == NULL ? "NULL" : "not NULL");
#endif

    iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
    float *weight = iftAllocFloatArray(mimg->m);
    int nobj = 0, nbkg = 0;

    for (int b = 0; b < mimg->m; b++)
    {
        if (channel_label[b] > 0)
        { /* object channel */
            nobj++;
        }
        else
        { /* background channel */
            nbkg++;
        }
    }

    /* Estimate saliency of each pixel */

    iftAdjRel *A = NULL;
    if (iftIs3DMImage(mimg))
        A = iftSpheric(adj_radius);
    else
        A = iftCircular(adj_radius);

    for (int p = 0; p < mimg->n; p++)
    {
        /* estimate mean and stdev of activations across channels */

        float mean[2] = {0.0, 0.0}, var[2] = {0.0, 0.0};
        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        for (int i = 0; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftMValidVoxel(mimg, v))
            {
                int q = iftMGetVoxelIndex(mimg, v);

                for (int b = 0; b < mimg->m; b++)
                {
                    if (channel_label[b] > 0)
                    { /* object channel */
                        mean[0] += mimg->val[q][b];
                    }
                    else
                    { /* background channel */
                        mean[1] += mimg->val[q][b];
                    }
                }
            }
        }
        mean[0] /= (nobj * A->n);
        mean[1] /= (nbkg * A->n);

        for (int i = 0; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftMValidVoxel(mimg, v))
            {
                int q = iftMGetVoxelIndex(mimg, v);

                for (int b = 0; b < mimg->m; b++)
                {
                    if (channel_label[b] > 0)
                    { /* object channel */
                        var[0] += powf(mimg->val[q][b] - mean[0], 2);
                    }
                    else
                    { /* background channel */
                        var[1] += powf(mimg->val[q][b] - mean[1], 2);
                    }
                }
            }
        }
        var[0] = var[0] / (nobj * A->n);
        var[1] = var[1] / (nbkg * A->n);

        /* It is expected two Gaussian distributions G0 and G1 of
           activations around mean[0] and mean[1]. We may expect that
           object pixels generate mean[0] >> mean[1] and the other way
           around for background pixels. We may use the the probability
           values of these distributions to estimate weights. */

        float P0, P1;
        for (int b = 0; b < mimg->m; b++)
        {
            if (var[0] > IFT_EPSILON)
                P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
            else
            {
                var[0] = 1.0;
                P0 = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
            }
            if (var[1] > IFT_EPSILON)
                P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
            else
            {
                var[1] = 1.0;
                P1 = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
            }

            if ((channel_label[b] > 0) && (P0 > P1))
            {
                weight[b] = 1; // P0;
            }
            else
            {
                if ((channel_label[b] < 0) && (P0 < P1))
                {
                    weight[b] = -1; // P1
                }
                else
                {
                    weight[b] = 0;
                }
            }
        }

        /* Estimate the salience map using one neuron per pixel -- it is
           no longer a point-wise convolution */

        for (int b = 0; b < mimg->m; b++)
        {
            salie->val[p] += mimg->val[p][b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0;
    }
    iftDestroyAdjRel(&A);
    iftFree(weight);

    return (salie);
}

float *iftGFLIMDecodeNode(iftGFLIMGraphNode node, iftGFLIMGraph *graph,
                          int nlabels, int min_label, int *channel_label,
                          float lambda)
{
    /* estimate mean of activations across channels */
    float mean[2] = {0.0, 0.0};
    float nkernels[2] = {0.0, 0.0};
    for (int b = 0; b < graph->num_feats; b++)
    {
        int label = channel_label[b] == -1 ? 0 : 1;
        float mean_feat = node.feats[b];

        iftSet *adj = node.adj;
        while (adj != NULL)
        {
            mean_feat += graph->nodes[adj->elem].feats[b];
            adj = adj->next;
        }
        mean[label] += mean_feat;
        nkernels[label]++;
    }
    
    mean[0] /= (nkernels[0] * (node.numNeighbors + 1));
    mean[1] /= (nkernels[1] * (node.numNeighbors + 1));

    float *weight = iftAllocFloatArray(graph->num_feats); // do not compute weight for background
    for (int b = 0; b < graph->num_feats; b++)
    {
        if (channel_label[b] > 0 && mean[0] > mean[1])
            weight[b] = 1;
        else
        {
            if (channel_label[b] < 0 && mean[0] < mean[1])
                weight[b] = -1;
            else
                weight[b] = 0;
        }
    }

    return weight;
}

// float *iftGFLIMDecodeNode(iftGFLIMGraphNode node, iftGFLIMGraph *graph,
//                         int nlabels, int min_label, int *channel_label,
//                         float lambda)
// {
//     /* estimate mean of activations across channels */
//     float *mean = iftAllocFloatArray(nlabels);
//     for (int b = 0; b < graph->num_feats; b++)
//     {
//         int label = channel_label[b] - min_label;
//         float mean_feat = node.feats[b];

//         iftSet *adj = node.adj;
//         while (adj != NULL)
//         {
//             mean_feat += graph->nodes[adj->elem].feats[b];
//             adj = adj->next;
//         }
//         mean[label] += (mean_feat / (node.numNeighbors + 1));
//     }

//     float *weight = iftAllocFloatArray(nlabels-1); // do not compute weight for background
//     for(int l = 1; l < nlabels; l++)
//     {
//         if (lambda * mean[l] >  mean[0]) weight[l-1] = 1;
//         else{
//             if(mean[l] < lambda * mean[0]) weight[l-1] = -1;
//             else weight[l-1] = 0;
//         }
//     }

//     return weight;
// }

/**
 * @fn iftGFLIMGraphAdaptiveDecoder
 * @brief Decode the features using a Graph Adaptive Decoder.
 * @param graph The input graph.
 * @param channel_label An array with the kernel labels.
 * @return The saliency of the nodes.
 * @note The channel_label is used to define the object and background channels.
 * @note This function normalizes the graph node features.
 */
float *iftGFLIMGraphAdaptiveDecoder(iftGFLIMGraph *graph, float *channel_weight)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    printf("\t channel_label: %s\n", channel_label == NULL ? "NULL" : "not NULL");
#endif

    //float lambda = 1.2;
    /* Normalize activations per channel */
    int nchannels[2] = {0, 0}; // number of channels for label
    for (int b = 0; b < graph->num_feats; b++)
    {
        nchannels[channel_weight[b] < 1 ? 1 : 0]++; // count the number of channels for each label
    }
    
    float *saliencyGraph = (float *)iftAlloc(graph->num_nodes, sizeof(float));

    /* Estimate saliency of each pixel */
//#pragma parallel for
    for (int p = 0; p < graph->num_nodes; p++)
    {
        iftGFLIMGraphNode node = graph->nodes[p];

        /* estimate mean of activations across channels for a node p */
        float mean[2] = {0.0, 0.0};
        for (int b = 0; b < graph->num_feats; b++)
        {
            float mean_feat = node.feats[b];
            iftSet *adj = node.adj;
            while (adj != NULL)
            {
                mean_feat += graph->nodes[adj->elem].feats[b];
                adj = adj->next;
            }
            mean[channel_weight[b] < 1 ? 1 : 0] += mean_feat;
        }
        mean[0] /= (nchannels[0] * (node.numNeighbors + 1));
        mean[1] /= (nchannels[1] * (node.numNeighbors + 1));

        for (int b = 0; b < graph->num_feats; b++)
        {
            // b is obj and mean_{obj} > mean_{bkg}
            if (channel_weight[b] > 0 && mean[0] > mean[1])
                saliencyGraph[p] += node.feats[b];
            else
            {
                // b is bkg and mean_{obj} < mean_{bkg}
                if (channel_weight[b] < 0 && mean[0] < mean[1])
                    saliencyGraph[p] -= node.feats[b];
            }
        }

        if (saliencyGraph[p] < 0) /* RELU */
            saliencyGraph[p] = 0;
    }

    return (saliencyGraph);
}

/**
 * @fn iftGFLIMMeanAdaptiveDecoder
 * @brief Decode the features using a Mean Adaptive Decoder.
 * @param graph The input graph.
 * @param channel_label An array with the kernel labels.
 * @return The saliency of the nodes.
 * @note The channel_label is used to define the object and background channels.
 * @note This function normalizes the graph node features.
 */
float *iftGFLIMMeanAdaptiveDecoder(iftGFLIMGraph *graph, float *channel_weight)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    printf("\t channel_label: %s\n", channel_label == NULL ? "NULL" : "not NULL");
#endif

    //float lambda = 1.2;
    /* Normalize activations per channel */
    int nchannels[2] = {0, 0}; // number of channels for label
    for (int b = 0; b < graph->num_feats; b++)
    {
        nchannels[channel_weight[b] < 1 ? 1 : 0]++; // count the number of channels for each label
    }
    
    float *saliencyGraph = (float *)iftAlloc(graph->num_nodes, sizeof(float));

    /* Estimate saliency of each pixel */
//#pragma parallel for
    for (int p = 0; p < graph->num_nodes; p++)
    {
        iftGFLIMGraphNode node = graph->nodes[p];

        /* estimate mean of activations across channels for a node p */
        float mean[2] = {0.0, 0.0};
        float neighbors_size = 0; // include the node itself
        for (int b = 0; b < graph->num_feats; b++)
        {
            float mean_feat = node.feats[b] * node.numPixels; // use the number of pixels to weight the mean
            neighbors_size += node.numPixels; // include the node itself
            iftSet *adj = node.adj;
            while (adj != NULL)
            {
                mean_feat += graph->nodes[adj->elem].feats[b] * graph->nodes[adj->elem].numPixels;
                neighbors_size += graph->nodes[adj->elem].numPixels; // include the neighbor
                adj = adj->next;
            }
            mean[channel_weight[b] < 1 ? 1 : 0] += mean_feat;
        }
        mean[0] /= (nchannels[0] * (node.numNeighbors + 1) * neighbors_size);
        mean[1] /= (nchannels[1] * (node.numNeighbors + 1) * neighbors_size);

        for (int b = 0; b < graph->num_feats; b++)
        {
            // b is obj and mean_{obj} > mean_{bkg}
            if (channel_weight[b] > 0 && mean[0] > mean[1])
                saliencyGraph[p] += node.feats[b];
            else
            {
                // b is bkg and mean_{obj} < mean_{bkg}
                if (channel_weight[b] < 0 && mean[0] < mean[1])
                    saliencyGraph[p] -= node.feats[b];
            }
        }

        if (saliencyGraph[p] < 0) /* RELU */
            saliencyGraph[p] = 0;
    }

    return (saliencyGraph);
}


/**
 * @fn iftGFLIMDecodeLayer
 * @brief Decode a layer.
 * @param graph The input graph.
 * @param label_img The image with the superpixels of the input graph. It maps the image pixels to the graph nodes (from 1).
 * @param label_img0 The image with the superpixels of the original image. It maps the image pixels to the graph nodes (from 1).
 * @param truelabels An array with the kernel labels.
 * @param nlabels The number of labels.
 * @param decoder The decoder type. @see iftGFLIMDecoderType.
 * @return The decoded image.
 * @note The truelabels array is used to define the object and background channels.
 * @note The label_img0 is used to interpolate the saliency map.
 * @note The decoder type is used to define the decoder algorithm.
 * @note The output image is normalized to 4095 (12 bits) or 255 (8 bits) and interpolated to the original image size.
 */
iftImageArray *iftGFLIMDecodeLayer(iftGFLIMGraph *graph,
                                   iftImage *label_img, iftImage *label_img0,
                                   int *truelabels, int nlabels,
                                   iftGFLIMDecoderType decoder)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    if (label_img == NULL)
        printf("\t label_img: NULL \n");
    else
        printf("\t label_img size: %d %d %d \n", label_img->xsize, label_img->ysize, label_img->zsize);
    if (label_img0 == NULL)
        printf("\t label_img0: NULL \n");
    else
        printf("\t label_img0 size: %d %d %d \n", label_img0->xsize, label_img0->ysize, label_img0->zsize);
    printf("\t truelabels: %s\n", truelabels == NULL ? "NULL" : "not NULL");
    printf("\t nlabels: %d\n", nlabels);
    printf("\t decoder: %d\n", decoder);
    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        printf("\t decoder: iftGFLIMTriStateAdaptativeDecoder\n");
        break;

    case IFT_GFLIM_HYBRID_DECODER:
        printf("\t decoder: HybridAdaptativeDecoder\n");
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        printf("\t decoder: GraphAdaptiveDecoder\n");
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        printf("\t decoder: GaussianDecoder\n");
        break;
    }
#endif

    float scale[3];
    iftFImage **salie = NULL;
    iftMImage *mimg = NULL;
    float *weight = NULL;
    float *saliencyGraph = NULL;
    iftImageArray *smap = NULL;

    if (IFT_GFLIM_TRISTATE_DECODER)
        nlabels = 1;

    if (nlabels == 0)
        iftError("Number of labels must be greater than 0: %d", __func__, nlabels);

    for (int b = 0; b < graph->num_feats; b++)
    {
        // get the maximum value of the channel
        double maxval = graph->nodes[0].feats[b];
        for (int p = 1; p < graph->num_nodes; p++)
            maxval = iftMax(graph->nodes[p].feats[b], maxval);

        // normalize the channel
        for (int p = 0; p < graph->num_nodes; p++)
        {
            if (maxval > 0)
                graph->nodes[p].feats[b] /= maxval;
            else
                graph->nodes[p].feats[b] = 0;
        }
    }

    salie = (iftFImage **)calloc(nlabels, sizeof(iftFImage *));

    switch (decoder)
    {
    case IFT_GFLIM_TRISTATE_DECODER:
        mimg = iftGFLIMGraphToMImage(graph, label_img);
        salie[0] = iftGFLIMTriStateAdaptativeDecoder(mimg);
        iftDestroyMImage(&mimg);
        break;

    case IFT_GFLIM_HYBRID_DECODER:
        if (truelabels == NULL)
            iftError("truelabels must be not NULL with hybrid decoder", __func__);

        mimg = iftGFLIMGraphToMImage(graph, label_img);
        weight = iftAllocFloatArray(graph->num_feats);
        for (int l = 0; l < nlabels; l++)
        {
            /* set weights */
            for (int k = 0; k < graph->num_feats; k++)
            {
                if ((l + 1) == truelabels[k])
                    weight[k] = +1;
                else
                    weight[k] = -1;
            }
            salie[l] = HybridAdaptativeDecoder(mimg, weight);
        }
        iftDestroyMImage(&mimg);
        iftFree(weight);
        break;

    case IFT_GFLIM_TEST_DECODER:
        if (truelabels == NULL)
            iftError("truelabels must be not NULL with test decoder", __func__);

        /* Estimate saliency of each pixel */
        for (int l = 0; l < nlabels; l++)
        {
            iftFImage *salie_graph = TestAdaptativeDecoder(graph, truelabels, l);

            salie[l] = iftCreateFImage(label_img->xsize, label_img->ysize, label_img->zsize);
            // fill the saliency image with the saliency of the graph nodes.
            for (int p = 0; p < label_img->n; p++)
                salie[l]->val[p] = salie_graph->val[label_img->val[p] - 1];
            
            iftFree(salie_graph);
        }
        break;

    case IFT_GFLIM_GRAPHADAPTIVE_DECODER:
        if (truelabels == NULL)
            iftError("truelabels must be not NULL with graph adaptive decoder", __func__);
        weight = iftAllocFloatArray(graph->num_feats);
        for (int l = 0; l < nlabels; l++)
        {
            /* set weights */
            for (int k = 0; k < graph->num_feats; k++)
            {
                if ((l + 1) == truelabels[k])
                    weight[k] = +1;
                else
                    weight[k] = -1;
            }
            saliencyGraph = iftGFLIMGraphAdaptiveDecoder(graph, weight);
            salie[l] = iftCreateFImage(label_img->xsize, label_img->ysize, label_img->zsize);
            // fill the saliency image with the saliency of the graph nodes.
            for (int p = 0; p < label_img->n; p++)
            {
                salie[l]->val[p] = saliencyGraph[label_img->val[p] - 1];
            }
            iftFree(saliencyGraph);
        }
        iftFree(weight);
        break;

    case IFT_GFLIM_MEAN_DECODER:
        if (truelabels == NULL)
            iftError("truelabels must be not NULL with graph adaptive decoder", __func__);
        weight = iftAllocFloatArray(graph->num_feats);
        for (int l = 0; l < nlabels; l++)
        {
            /* set weights */
            for (int k = 0; k < graph->num_feats; k++)
            {
                if ((l + 1) == truelabels[k])
                    weight[k] = +1;
                else
                    weight[k] = -1;
            }
            saliencyGraph = iftGFLIMMeanAdaptiveDecoder(graph, weight);
            salie[l] = iftCreateFImage(label_img->xsize, label_img->ysize, label_img->zsize);
            // fill the saliency image with the saliency of the graph nodes.
            for (int p = 0; p < label_img->n; p++)
            {
                salie[l]->val[p] = saliencyGraph[label_img->val[p] - 1];
            }
            iftFree(saliencyGraph);
        }
        iftFree(weight);
        break;

    case IFT_GFLIM_GAUSSIAN_DECODER:
        if (truelabels == NULL)
            iftError("truelabels must be not NULL with gaussian decoder", __func__);
        for (int l = 0; l < nlabels; l++)
        {
            iftFImage *salie_graph = iftGFLIMGaussianDecoder(graph, truelabels);
            salie[l] = iftCreateFImage(label_img->xsize, label_img->ysize, label_img->zsize);
            // fill the saliency image with the saliency of the graph nodes.
            for (int p = 0; p < label_img->n; p++)
                salie[l]->val[p] = salie_graph->val[label_img->val[p] - 1];
            
            iftFree(salie_graph);
        }
        break;

    default:
        iftError("Invalid decoder type: %d", __func__, decoder);
        break;
    }

    smap = iftCreateImageArray(nlabels);

    scale[0] = (float)label_img0->xsize / (float)label_img->xsize;
    scale[1] = (float)label_img0->ysize / (float)label_img->ysize;
    scale[2] = (float)label_img0->zsize / (float)label_img->zsize;

    for (int l = 0; l < nlabels; l++)
    {
        int Imax;
        /* Interpolate and normalize the salience maps */
        iftFImage *smap_interp = NULL;
        if (iftIs3DImage(label_img))
        {
            smap_interp = iftFInterp(salie[l], scale[0], scale[1], scale[2]);
            Imax = 4095;
        }
        else
        {
            smap_interp = iftFInterp2D(salie[l], scale[0], scale[1]);
            Imax = 255;
        }

        smap->val[l] = iftFImageToImage(smap_interp, Imax);
        iftDestroyFImage(&smap_interp);
        iftDestroyFImage(&salie[l]);
    }
    iftFree(salie);

    return smap;
}

// =========== FUNCTION TO EXTRACT FEATURES ============ //

/**
 * @fn iftGFLIMImageToFLIMGraphBatch
 * @brief Convert a batch of images to a batch of FLIM graphs.
 * @param fs The image file set.
 * @param labels_dir The labels directory.
 * @param first The first image index.
 * @param last The last image index.
 * @param initial_seeds The number of initial seeds.
 * @param nsuperpixels The number of superpixels.
 * @param basename An array with the name of the images.
 * @param graph_feat The graph feature. @see iftGFLIMGraphNodeFeature.
 * @param graph_type The graph type. @see iftGFLIMGraphSpxMethod.
 * @return The FLIM graph array.
 */
iftGFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch(iftFileSet *fs, char *labels_dir,
                                                 int first, int last,
                                                 int initial_seeds, int nsuperpixels,
                                                 char **basename,
                                                 iftGFLIMGraphNodeFeature graph_feat,
                                                 iftGFLIMGraphSpxMethod graph_type)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (fs == NULL)
        printf("\t fs: NULL \n");
    else
        printf("\t fs size: %ld \n", fs->n);
    printf("\t labels_dir: %s\n", labels_dir);
    printf("\t first: %d\n", first);
    printf("\t last: %d\n", last);
    printf("\t initial_seeds: %d\n", initial_seeds);
    printf("\t nsuperpixels: %d\n", nsuperpixels);
    printf("\t basename: %s\n", basename == NULL ? "NULL" : "not NULL");
    printf("\t graph_feat: %d\n", graph_feat);
    printf("\t graph_type: %d\n", graph_type);
#endif

    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(last - first + 1);

#pragma omp parallel for
    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        char ext[10], label_file[512];

        sprintf(ext, "%s", iftFileExt(fs->files[i]->path));
        basename[k] = iftFilename(fs->files[i]->path, ext);
        sprintf(label_file, "%s/%s.pgm", labels_dir, basename[k]);

        iftImage *img = iftReadImageByExt(fs->files[i]->path);
        iftImage *label_img = NULL;

        if (graph_type == IFT_FLIMGRAPH_TYPE_ERGC)
            initial_seeds = 0;
        input->val[k] = iftImageToGFLIMGraph(img, nsuperpixels, initial_seeds, &label_img, graph_feat, graph_type);

        iftWriteImageByExt(label_img, label_file);
        iftDestroyImage(&label_img);
        iftDestroyImage(&img);
    }
    return input;
}

iftGFLIMGraphArray *iftGFLIMImageToFLIMGraphBatch_v2(iftFileSet *fs,
                                                    int first, int last,
                                                    int initial_seeds, int nsuperpixels,
                                                    iftGFLIMGraphNodeFeature graph_feat,
                                                    iftGFLIMGraphSpxMethod graph_type,
                                                    iftImageArray *label_imgs)
{
    iftGFLIMGraphArray *input = iftCreateGFLIMGraphArray(last - first + 1);

    for (int k = 0; k <= last - first; k++)
    {
        int i = k + first;
        iftImage *img = iftReadImageByExt(fs->files[i]->path);
        input->val[k] = iftImageToGFLIMGraph(img, nsuperpixels, initial_seeds, &(label_imgs->val[k]), graph_feat, graph_type);
        iftDestroyImage(&img);
    }
    return input;
}
