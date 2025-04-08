
#include "iftGFLIM.h"

/* --------------------- GFLIM seeds (user drawn markers) ----------------------------- */

/**
 * @fn iftGFLIMRelabelConnectedComponentsInPlace
 * @brief Label the seeds based on their connected components on the graph.
 * @param graph Graph with the nodes and their weights.
 * @param S set of labeled graph nodes (the seeds).
 * @param obj_label_map The map of object labels.
 * @return iftLabeledSet* labeled set of seeds with a different label for each connected component.
 * @warning obj_label_map must be allocated before calling this function.
 * Its size must be equal to the number of nodes in the graph and its initial values are overwritten.
 */
void iftGFLIMRelabelConnectedComponentsInPlace(iftFLIMGraph *graph, iftLabeledSet *S, int *obj_label_map)
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
    printf("\t obj_label_map: %s \n", obj_label_map == NULL ? "NULL" : "not NULL");
#endif

    iftLabeledSet *seed = NULL;
    iftDHeap *queue;
    double *markers = iftAllocDoubleArray(graph->num_nodes);
    double max_label = IFT_INFINITY_DBL;
    queue = iftCreateDHeap((int)(graph->num_nodes), markers);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        markers[i] = -1;
        obj_label_map[i] = -1;
    }

    seed = S;
    while (seed != NULL)
    {
        int p = seed->elem;
        markers[p] = max_label;
        obj_label_map[p] = seed->label;
        iftInsertDHeap(queue, p);
        seed = seed->next;
    }

    double curr_label = 1.0;
    while (!iftEmptyDHeap(queue))
    {
        int node_index = iftRemoveDHeap(queue);

        if (markers[node_index] == max_label)
        { // unconnected seed
            markers[node_index] = curr_label;
            curr_label++;
        }
        double node_label = markers[node_index];

        iftSet *neigh = graph->nodes[node_index].adj;
        while (neigh != NULL)
        {
            // insert in heap neighbor nodes that are seeds
            // and were not visited yet (gray color)
            // and have the same label (foreground or background) as node_index
            if (markers[neigh->elem] == max_label &&
                queue->color[neigh->elem] == IFT_GRAY &&
                obj_label_map[node_index] == obj_label_map[neigh->elem])
            {
                markers[neigh->elem] = node_label; // indicate connectivity - they belong to the same marker
                iftGoUpDHeap(queue, queue->pos[neigh->elem]);
            }
            neigh = neigh->next;
        }
    }

    seed = S;
    while (seed != NULL)
    {
        seed->marker = (int)(markers[seed->elem]);
        seed = seed->next;
    }

    iftFree(markers);
    iftDestroyDHeap(&queue);
}

/**
 * @fn iftGFLIMExtractLabelDataSet
 * @brief Compute a dataset from the labeled nodes in the graph,
 * agreggating features from their neighbors.
 * @param graph Graph with the nodes and their weights.
 * @param S set of labeled graph nodes.
 * @param kernel_size Number of neighbors to compose a kernel.
 * @param attention_opt The attention option to compute the features. @see iftFLIMGraphAttention.
 * @param aggregation_opt The aggregation option. @see iftGFLIMAggregationType.
 * @return iftDataSet* kernel bank
 * @note The truelabel in the dataset is the label from the seeds.
 */
iftDataSet *iftGFLIMExtractLabelDataSet(iftFLIMGraph *graph,
                                        iftLabeledSet *S,
                                        int kernel_size,
                                        iftFLIMGraphAttention attention_opt,
                                        iftGFLIMAggregationType aggregation_opt)
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
    printf("\t attention_opt: %d\n", attention_opt);
    printf("\t aggregation_opt: %d\n", aggregation_opt);
#endif

    int nsamples = iftLabeledSetSize(S);
    int kernel_neighbors = iftMin(graph->num_nodes, kernel_size);
    int tensor_size = aggregation_opt == IFT_GFLIM_AGGREGATION_CONCAT ? graph->num_feats * kernel_neighbors : 2 * graph->num_feats;
    iftDataSet *Z = iftCreateDataSet(nsamples, tensor_size);

    iftLabeledSet *seed = S;
    int incr = 0;
    while (seed != NULL)
    {
        int l = seed->label;
        if (l == 0)
        {
            incr = 1;
            break;
        }
        seed = seed->next;
    }

    Z->nclasses = 0;
    int s = 0;
    seed = S;
    while (seed != NULL)
    {
        int p = seed->elem;
        Z->sample[s].id = s;
        Z->sample[s].truelabel = seed->label + incr; /* vary from 1 to c */
        if (Z->sample[s].truelabel > Z->nclasses)
            Z->nclasses = Z->sample[s].truelabel;

        // find the k most similar neighbors
        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(p, graph, kernel_neighbors,
                                                           1, true, attention_opt);

        // aggregate the features of the neighbors and fill missing neighbors with the values in band_filling
        double *feature_vector = iftFLIMgraphAggregateFeatures(seed->elem, neighbors,
                                                               graph, kernel_size,
                                                               aggregation_opt);

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

/**
 * @fn StatisticsFromAllSeedsGraph
 * @brief Get the mean and standard deviation from seeds files
 * @param fs_seeds FileSet with seeds files
 * @param dir Directory where the graph files are
 * @param mean Mean of the seeds (output)
 * @param stdev Standard deviation of the seeds (output)
 * @param stdev_factor Factor to add to the standard deviation
 */
void StatisticsFromAllSeedsGraph(iftFileSet *fs_seeds, char *dir, float *mean, float *stdev, float stdev_factor)
{
    int nseeds = 0, nfeats = 0;
    char *basename = NULL;
    char filename[200];
    iftFLIMGraph *graph = NULL;

    basename = iftFilename(fs_seeds->files[0]->path, "-seeds_graph.txt");
    sprintf(filename, "%s/%s.json", dir, basename);
    graph = iftReadFLIMGraph(filename);
    nfeats = graph->num_feats;
    iftFree(basename);
    float *sum_sqrd = iftAllocFloatArray(nfeats);
    iftDestroyFLIMGraph(&graph);

    for (int i = 0; i < fs_seeds->n; i++)
    {
        iftLabeledSet *seed = NULL;
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");

        // read the graph
        sprintf(filename, "%s/%s.json", dir, basename);
        graph = iftReadFLIMGraph(filename);
        iftFree(basename);
        if(nfeats != graph->num_feats)
            iftError("Number of features in the graph is different from the number of features in the seeds", __func__);

        // read the seeds
        iftLabeledSet *S = iftReadSeedsGraph(fs_seeds->files[i]->path);

        // compute the sum of the features of the seeds
        seed = S;
        while (seed != NULL)
        {
            int p = seed->elem;
            nseeds += 1;
            for (int b = 0; b < nfeats; b++)
            {
                mean[b] += graph->nodes[p].feats[b];
                sum_sqrd[b] += powf(graph->nodes[p].feats[b], 2);
            }
            seed = seed->next;
        }
        iftDestroyLabeledSet(&S);
        iftDestroyFLIMGraph(&graph);
    }

    // compute the mean of the features
    for (int b = 0; b < nfeats; b++)
    {
        mean[b] = mean[b] / nseeds;
        stdev[b] = sqrtf((sum_sqrd[b] / nseeds) - powf(mean[b], 2)) + stdev_factor;
    }
}

/* --------------------- GFLIM training ----------------------------- */

/**
 * @fn iftGFLIMSumMImgToFloatArray
 * @brief Sum the bias to a mimage.
 * @param mimg The input mimage.
 * @param bias The bias to be summed.
 * @param inplace If true, the bias is summed in place.
 * @return The sum of the bias to the mimage.
 */
iftMImage *iftGFLIMSumMImgToFloatArray(iftMImage *mimg, float *bias, bool inplace)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (mimg == NULL)
        printf("\t mimg: NULL \n");
    else
        printf("\t mimg size: %d %d %d %ld \n", mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
    printf("\t bias: %s\n", bias == NULL ? "NULL" : "not NULL");
    printf("\t inplace: %s\n", inplace ? "true" : "false");
#endif

    iftMImage *out = NULL;

    if (inplace)
    {
#pragma omp parallel for
        for (int p = 0; p < mimg->n; p++)
        {
            for (int b = 0; b < mimg->m; b++)
                mimg->val[p][b] = mimg->val[p][b] + bias[b];
        }
        out = mimg;
    }
    else
    {
        out = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
#pragma omp parallel for
        for (int p = 0; p < mimg->n; p++)
        {
            for (int b = 0; b < mimg->m; b++)
                out->val[p][b] = mimg->val[p][b] + bias[b];
        }
    }
    return out;
}

/**
 * @fn LearnKernelBankGraph
 * @brief Learn the kernel bank from the graph
 * @param graph Graph with the nodes and their weights.
 * @param M set of labeled graph nodes.
 * @param kernel_size Number of neighbors to compose a kernel.
 * @param nkernels_per_image Number of kernels per image.
 * @param nkernels_per_marker Number of kernels per marker.
 * @return iftMatrix* kernel bank
 */
iftMatrix *LearnKernelBankGraph(iftFLIMGraph *graph, iftLabeledSet *M,
                                int kernel_size,
                                int nkernels_per_image,
                                int nkernels_per_marker)
{
    // Z contém as informações dos kernels.
    // Z->sample contém um vetor com os kernels.
    // Z->nclasses = numero de marcadores
    // Cada posição é um kernel e contém o rótulo da semente que o gerou. Ou seja, pode conter kernels de mesmo rótulo
    // As features de cada kernel correspondem às cores da semente e de suas posições vizinhas
    iftDataSet *Z = iftGFLIMExtractLabelDataSet(graph, M, kernel_size, IFT_FLIMGRAPH_ATTENTION_NONE, IFT_GFLIM_AGGREGATION_CONCAT);

    // vetor de matrizes de kernels, Cada matriz é banco de kernels
    // cuja quantidade de matrizes (kernels) é igual ao número de marcadores
    iftMatrix **kernels = (iftMatrix **)calloc(Z->nclasses + 1, sizeof(iftMatrix *));
    int total_nkernels = 0;

    // para cada rótulo c, ou seja, para cada marcador
    // calcula seus kernels (uma matriz), armazenando-os em kernels[c]
    for (int c = 1; c <= Z->nclasses; c++)
    {
        iftDataSet *Z1 = iftExtractSamplesFromClass(Z, c);
        int nkernels = nkernels_per_marker;
        /* 0: kmeans, 1: iterated watershed and 2: OPF c clusters */
        kernels[c] = iftComputeKernelBank(Z1, &nkernels, 0, iftEuclideanDistance);
        total_nkernels += nkernels;
        iftDestroyDataSet(&Z1);
    }

    // kernelbank une os kernels de todos os marcadores
    // Seu tamanho é [nkernels][graph->num_feats * kernel_size[0] * kernel_size[1]],
    // em que nkernels é
    iftMatrix *kernelbank = iftCreateMatrix(total_nkernels, graph->num_feats * kernel_size);

    int k = 0;
    for (int c = 1; c <= Z->nclasses; c++)
    {
        for (int col = 0; col < kernels[c]->ncols; col++, k++)
        {
            for (int row = 0; row < kernels[c]->nrows; row++)
            {
                iftMatrixElem(kernelbank, k, row) = iftMatrixElem(kernels[c], col, row);
            }
        }
    }

    for (int c = 0; c <= Z->nclasses; c++)
        iftDestroyMatrix(&kernels[c]);
    iftFree(kernels);
    iftDestroyDataSet(&Z);

    if (kernelbank->ncols > nkernels_per_image)
    { /* force a number of kernels per image */
        iftMatrix *Rkernels = iftSelectRelevantKernelsByKmeans(kernelbank, nkernels_per_image, iftEuclideanDistance);
        iftDestroyMatrix(&kernelbank);
        kernelbank = Rkernels;
    }

    return (kernelbank);
}

/**
 * @fn FLIMGraphConvolutionalLayer
 * @brief Apply a convolutional layer to a set of graphs.
 * @param graph Array of graphs.
 * @param arch Architecture of the network.
 * @param layer Layer index in iftFLIMArch* (>= 0).
 * @param layer_index Layer value (> 0).
 * @param atrous_factor Atrous factor for the convolution.
 * @param param_dir Directory with the parameters of the layer.
 * @return iftFLIMGraphArray* Array of graphs after applying the convolutional layer.
 */
iftFLIMGraphArray *
FLIMGraphConvolutionalLayer(iftFLIMGraphArray *graph,
                            iftFLIMArch *arch,
                            int layer, int layer_index,
                            int atrous_factor, char *param_dir)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t ngraphs: %d\n", graph->n);
    printf("\t layer: %d\n", layer);
    printf("\t layer_index: %d\n", layer_index);
    printf("\t atrous_factor: %d\n", atrous_factor);
    printf("\t param_dir: %s\n", param_dir == NULL ? "NULL" : param_dir);
    printf("\t arch: %s\n", arch == NULL ? "NULL" : "NOT NULL");
#endif

    if (param_dir == NULL || !iftDirExists(param_dir))
        iftError("Model directory does not exist", __func__);
    if (graph->n < 1)
        iftError("Number of graphs is less than 1", __func__);
    if (layer < 0)
        iftError("Layer is less than 0", __func__);
    if (layer_index < 1)
        iftError("Layer index is less than 0", __func__);

    // Set input parameters
    int ninput_channels = graph->val[0]->num_feats;
    char *basename = NULL;
    iftFLIMGraphArray *output_graph = iftFLIMGraphCreateEmptyFLIMGraphArray(graph->n);

    // Read consensus parameters of the current layer

    char filename[200];
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftMatrix *kernelbank = iftReadMatrix(filename);
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftReadMeanStdev(filename, mean, stdev, ninput_channels);

    // BIAS: use environment variable to use the bias idea
    float *bias = NULL;
    char *use_bias = getenv("USE_BIAS");

    if (use_bias != NULL)
        bias = iftReadBias(filename);

    // Apply convolutional layer
    for (int i = 0; i < graph->n; i++)
    {
        iftMatrix *imgM = NULL;

        // normalize the graph
        if (use_bias == NULL) iftNormalizeFLIMGraphByZScore(graph->val[i], mean, stdev);
        imgM = iftFLIMGraphToFeatureMatrix(graph->val[i], arch->layer[layer].kernel_size[0] * arch->layer[layer].kernel_size[1], IFT_FLIMGRAPH_ATTENTION_NONE, IFT_GFLIM_AGGREGATION_CONCAT, NULL);

        // convolution
        iftMatrix *conv = iftMultMatrices(imgM, kernelbank); // (n, m*k) x (m*k, k*k) = (n, k*k)
        iftDestroyMatrix(&imgM);

        iftMImage *activ = iftMatrixToMImage(conv, conv->nrows, 1, 1, conv->ncols, 'c');
        iftDestroyMatrix(&conv);

        // activation and bias
        if (arch->layer[layer].relu)
            iftMReLU(activ, bias, true);
        else
        {
            if (bias != NULL)
                iftGFLIMSumMImgToFloatArray(activ, bias, true);
        }

        // pooling
        output_graph->val[i] = iftMImageToFLIMGraph(activ, graph->val[i]);
        iftGFLIMPoolingLayer(output_graph->val[i],
                             arch->layer[layer].pool_type,
                             arch->layer[layer].pool_size[0] *
                             arch->layer[layer].pool_size[1],
                             arch->layer[layer].pool_stride);

        iftDestroyMImage(&activ);
    }

    iftDestroyMatrix(&kernelbank);
    iftFree(mean);
    iftFree(stdev);

    if (use_bias != NULL)
        iftFree(bias);

    return output_graph;
}

/**
 * @fn ConsensusKernelbankGraph
 * @brief Create a consensus kernel bank from the seeds files
 * @param fs_seeds FileSet with seeds files
 * @param inputdata_dir Directory where the graph files are
 * @param noutput_channels Number of output channels
 * @param stdev_factor Standard deviation factor
 * @return iftMatrix* kernel bank
 */
iftMatrix *ConsensusKernelbankGraph(iftFileSet *fs_seeds,
                                    char *inputdata_dir, int noutput_channels,
                                    float stdev_factor)
{
    if (inputdata_dir == NULL || !iftDirExists(inputdata_dir))
        iftError("Input data directory does not exist", __func__);
    if (fs_seeds->n < 1)
        iftError("No kernels to create the consensus kernelbank", __func__);
    if (noutput_channels < 1)
        iftError("Number of output channels is less than 1", __func__);

    int nkernels = fs_seeds->n;

    if (nkernels <= 0)
        iftError("No kernels to create the consensus kernelbank: %d", __func__, nkernels);

    iftMatrix **kernels = (iftMatrix **)calloc(nkernels, sizeof(iftMatrix *));
    int ncols = 0;
    int nrows = 0;

    /* Load kernels from the training images */
    for (int i = 0; i < nkernels; i++)
    {
        char filename[200];
        char *basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "%s/%s-kernels.npy", inputdata_dir, basename);
        kernels[i] = iftReadMatrix(filename);
        ncols += kernels[i]->ncols;
        iftFree(basename);
    }
    nrows = kernels[0]->nrows;

    /* Copy all kernels into a single matrix */

    iftMatrix *Mkernels = iftCreateMatrix(ncols, nrows);
    int k = 0;
    for (int i = 0; i < nkernels; i++)
    {
        for (int col = 0; col < kernels[i]->ncols; col++, k++)
        {
            for (int row = 0; row < kernels[i]->nrows; row++)
                iftMatrixElem(Mkernels, k, row) = iftMatrixElem(kernels[i], col, row);
        }
    }

    for (int i = 0; i < nkernels; i++)
        iftDestroyMatrix(&kernels[i]);
    iftFree(kernels);

    /* Reduce the number of kernels into the desired number of output channels */

    iftMatrix *Rkernels = NULL;

    if (Mkernels->ncols <= noutput_channels)
        Rkernels = iftCopyMatrix(Mkernels);
    else
    {
        if (Mkernels->ncols > Mkernels->nrows)
            Rkernels = iftSelectRelevantKernelsByPCA(Mkernels, noutput_channels, stdev_factor);
        else
            Rkernels = iftSelectRelevantKernelsByKmeans(Mkernels, noutput_channels, iftEuclideanDistance);
    }

    iftDestroyMatrix(&Mkernels);
    return (Rkernels);
}

/**
 * @fn iftFLIMGraphLearnLayer
 * @brief Learn a layer
 * @param input_dir directory with input files
 * @param markers_dir directory with seeds files
 * @param param_dir directory with the model files
 * @param layer_index layer index
 * @param arch architecture
 * @param output_dir directory to write the output files
 * @note The input files must be graphs in .json format
 * @note The seeds files must be in .txt format (*-seeds_graph.txt)
 * @note The output_dir must be an existing directory
 * @note The architecture must be in .json format
 */
void iftGFLIMTrainLayer(char *input_dir, char *markers_dir,
                            char *param_dir, int layer_index,
                            iftFLIMArch *arch, char *output_dir)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t input_dir: %s\n", input_dir == NULL ? "NULL" : input_dir);
    printf("\t markers_dir: %s\n", markers_dir == NULL ? "NULL" : markers_dir);
    printf("\t param_dir: %s\n", param_dir == NULL ? "NULL" : param_dir);
    printf("\t layer_index: %d \n", layer_index);
    printf("\t output_dir: %s\n", output_dir == NULL ? "NULL" : output_dir);
#endif

    if (input_dir == NULL && !iftDirExists(input_dir))
        iftError("Input directory does not exist", __func__);
    if (markers_dir == NULL && !iftDirExists(markers_dir))
        iftError("Markers directory does not exist", __func__);
    if (param_dir == NULL && !iftDirExists(param_dir))
        iftError("Model directory does not exist", __func__);
    if (output_dir == NULL)
        iftError("Output directory is NULL", __func__);

    /* Set input parameters */
    iftFileSet *fs_activ = iftLoadFileSetFromDirOrCSV(input_dir, 1, 1); // original images (for layer 1)
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds_graph.txt", 1);
    iftFLIMGraphArray *output = NULL;
    int ninput_channels = 0;
    int ngraphs = fs_seeds->n;
    int atrous_factor = 1;
    char *basename = NULL;
    char filename[512], ext[10], tmp_dir[255];

    if (fs_activ->n < 1)
        iftError("No activation files found", __func__);
    if (fs_seeds->n < 1)
        iftError("No seeds files found", __func__);

    sprintf(tmp_dir, "%s/tmp", param_dir);
    iftMakeDir(tmp_dir);
    sprintf(ext, "%s", iftFileExt(fs_activ->files[0]->path));

    /* Generate input layer */

    // read activation files and write them in the temp directory
    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "%s/%s%s", input_dir, basename, ext);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);
        iftIsValidFLIMGraph(graph);

        sprintf(filename, "%s/%s.json", tmp_dir, basename);
        iftWriteFLIMGraph(graph, filename);
        if (i == 0)
            ninput_channels = graph->num_feats;

        iftFree(basename);
        iftDestroyFLIMGraph(&graph);
    }

    // compute mean and std of the graph node seeds
    float *mean = iftAllocFloatArray(ninput_channels);
    float *stdev = iftAllocFloatArray(ninput_channels);

    StatisticsFromAllSeedsGraph(fs_seeds, tmp_dir, mean, stdev, arch->stdev_factor);

    printf("\nLayer %d\n", layer_index);
    fflush(stdout);

    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");

        // read the input graph
        sprintf(filename, "%s/%s.json", tmp_dir, basename);
        iftFLIMGraph *graph = iftReadFLIMGraph(filename);

        printf("Processing file %s: %d of %d files\r", basename, i + 1, ngraphs);
        fflush(stdout);

        /* read seeds */
        iftLabeledSet *S = iftReadSeedsGraph(fs_seeds->files[i]->path);
        int obj_label_map[graph->num_nodes];
        iftGFLIMRelabelConnectedComponentsInPlace(graph, S, obj_label_map);

        /* Apply marker-based image normalization */
        iftNormalizeFLIMGraphByZScore(graph, mean, stdev);
        sprintf(filename, "%s/%s-norm.json", tmp_dir, basename);
        iftWriteFLIMGraph(graph, filename);

        /* Learn and save kernel bank */

        iftMatrix *kernelbank = LearnKernelBankGraph(graph, S,
                                                     arch->layer[0].kernel_size[0] * arch->layer[0].kernel_size[1],
                                                     arch->layer[0].nkernels_per_image,
                                                     arch->layer[0].nkernels_per_marker);

        iftDestroyFLIMGraph(&graph);
        iftDestroyLabeledSet(&S);
        sprintf(filename, "%s/%s-kernels.npy", tmp_dir, basename);
        iftWriteMatrix(kernelbank, filename);
        iftDestroyMatrix(&kernelbank);
        iftFree(basename);
    }

    /* Create a consensus layer (i.e., merge kernels) and save final kernel bank for layer l */
    iftMatrix *kernelbank = ConsensusKernelbankGraph(fs_seeds, tmp_dir, arch->layer[0].noutput_channels, arch->stdev_factor);

    /* BIAS: read bias array */
    char *use_bias = getenv("USE_BIAS");
    float *bias = NULL;

    if (use_bias != NULL)
    {
        bias = iftAllocFloatArray(kernelbank->ncols);
        for (int col = 0; col < kernelbank->ncols; col++)
        {
            for (int row = 0; row < kernelbank->nrows;)
            {
                for (int ch = 0; ch < ninput_channels; ch++)
                {
                    iftMatrixElem(kernelbank, col, row) =
                        iftMatrixElem(kernelbank, col, row) / stdev[ch];
                    bias[col] -= (mean[ch] * iftMatrixElem(kernelbank, col, row));
                    row++;
                }
            }
        }

        sprintf(filename, "%s/conv%d", param_dir, layer_index);
        iftWriteBias(filename, bias, kernelbank->ncols);
        iftFree(bias);
    }

    // write the final kernel bank
    sprintf(filename, "%s/conv%d-kernels.npy", param_dir, layer_index);
    iftWriteMatrix(kernelbank, filename);
    iftDestroyMatrix(&kernelbank);

    // write the mean and std of the labeled nodes
    sprintf(filename, "%s/conv%d", param_dir, layer_index);
    iftWriteMeanStdev(filename, mean, stdev, ninput_channels);
    iftFree(mean);
    iftFree(stdev);

    int pool_stride;

    /* Apply convolutional layer using the consensus kernel bank and the statistics from all markers */
    pool_stride = arch->layer[0].pool_stride;
    arch->layer[0].pool_stride = 1;
    for (int i = 0; i < ngraphs; i++)
    {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds_graph.txt");
        sprintf(filename, "%s/%s.json", tmp_dir, basename);
        iftFLIMGraphArray *graph = iftFLIMGraphCreateEmptyFLIMGraphArray(1);
        graph->val[0] = iftReadFLIMGraph(filename);
        output = FLIMGraphConvolutionalLayer(graph, arch, 0, layer_index, atrous_factor, param_dir);
        iftDestroyFLIMGraphArray(&graph);
        sprintf(filename, "%s/%s.json", output_dir, basename);
        iftWriteFLIMGraph(output->val[0], filename);
        iftDestroyFLIMGraphArray(&output);
        iftFree(basename);
    }
    arch->layer[0].pool_stride = pool_stride;

    // updating atrous factor with pooling of current layer
    if (arch->apply_intrinsic_atrous)
    {
        atrous_factor *= arch->layer[0].pool_stride;
        fflush(stdout);
    }
    // writing atrous factor on file
    sprintf(filename, "%s/intrinsic_atrous.txt", param_dir);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d", atrous_factor);
    fclose(fp);
    iftRemoveDir(tmp_dir);
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_activ);
}

/* --------------------- GFLIM extract features ----------------------------- */

/**
 * @fn FLIMGraphExtractFeaturesFromLayerPerBatch
 * @brief Extract features from a layer of a set of graphs
 * @param fs_graphs FileSet with the graph file names
 * @param first First graph index (in fs_graphs)
 * @param last Last graph index (in fs_graphs)
 * @param orig_dir Directory of the input
 * @param arch Architecture of the network
 * @param param_dir Directory of the model
 * @param layer_index Layer index (> 0)
 * @param feat_dir Directory to write the features
 * @note The input files must be graphs in .json format
 * @note The seeds files must be in .txt format
 * @note The feat_dir must be an existing directory
 */
void FLIMGraphExtractFeaturesFromLayerPerBatch(iftFileSet *fs_graphs, int first, int last,
                                               char *orig_dir, iftFLIMArch *arch,
                                               char *param_dir, int layer_index, char *feat_dir)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t first: %d\n", first);
    printf("\t last: %d\n", last);
    printf("\t orig_dir: %s\n", orig_dir == NULL ? "NULL" : orig_dir);
    printf("\t param_dir: %s\n", param_dir == NULL ? "NULL" : param_dir);
    printf("\t layer_index: %d\n", layer_index);
    printf("\t arch: %s\n", arch == NULL ? "NULL" : "NOT NULL");
    printf("\t feat_dir: %s\n", feat_dir == NULL ? "NULL" : feat_dir);
    printf("\t fs_graphs: %s\n", fs_graphs == NULL ? "NULL" : "NOT NULL");
    if (fs_graphs != NULL)
        printf("\t fs_graphs->n: %d\n", fs_graphs->n);
#endif

    if (!iftDirExists(param_dir))
        iftError("Model directory does not exist", __func__);
    if (!iftDirExists(orig_dir))
        iftError("Original directory does not exist", __func__);
    if (!iftDirExists(feat_dir))
        iftError("Feature directory does not exist", __func__);

    iftFLIMGraphArray *input = NULL, *output = NULL;
    char filename[200];
    int ngraphs = last - first + 1;
    char ext[20];
    sprintf(ext, "%s", iftFileExt(fs_graphs->files[0]->path));
    char **basename = (char **)calloc(ngraphs, sizeof(char *));
    input = iftFLIMGraphCreateEmptyFLIMGraphArray(ngraphs);
    output = NULL;

    /* Load batch */

    for (int i = first, k = 0; i <= last; i++, k++)
    {
        sprintf(filename, "%s/%s", orig_dir, fs_graphs->files[i]->path);
        input->val[k] = iftReadFLIMGraph(filename);
        basename[k] = iftFilename(fs_graphs->files[i]->path, ext);
    }

    /* For each layer do */

    output = FLIMGraphConvolutionalLayer(input, arch, 0, layer_index, 1, param_dir);

    for (int k = 0; k < ngraphs; k++)
    {
        iftDestroyFLIMGraph(&input->val[k]);
        input->val[k] = output->val[k];
        output->val[k] = NULL;
    }
    iftFree(output);

    for (int k = 0; k < ngraphs; k++)
    {
        sprintf(filename, "%s/%s.json", feat_dir, basename[k]);
        iftFree(basename[k]);
        iftWriteFLIMGraph(input->val[k], filename);
        iftDestroyFLIMGraph(&input->val[k]);
    }

    iftFree(input);
    iftFree(basename);
}

/**
 * @fn iftFLIMGraphExtractFeaturesFromLayer
 * @brief Extract features from a set of graphs using the weights of a layer
 * @param orig_dir Input directory
 * @param graph_list FileSet with the graph file names
 * @param arch Architecture of the network
 * @param param_dir Directory of the model
 * @param layer_index Layer index (> 0)
 * @param out_dir Directory to write the features
 * @param batchsize Batch size (> 0)
 * @note The input files must be graphs in .json format and contain the files in the graph_list
 * @note The out_dir must be an existing directory
 * @note The graph_list must be a valid file or directory
 */
void iftFLIMGraphExtractFeaturesFromLayer(char *orig_dir, char *graph_list, iftFLIMArch *arch,
                                          char *param_dir, int layer_index,
                                          char *out_dir, int batchsize)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t orig_dir: %s\n", orig_dir == NULL ? "NULL" : orig_dir);
    printf("\t graph_list: %s\n", graph_list == NULL ? "NULL" : graph_list);
    printf("\t param_dir: %s\n", param_dir == NULL ? "NULL" : param_dir);
    printf("\t layer_index: %d\n", layer_index);
    printf("\t out_dir: %s\n", out_dir == NULL ? "NULL" : out_dir);
    printf("\t arch: %s\n", arch == NULL ? "NULL" : "NOT NULL");
    printf("\t batchsize: %d\n", batchsize);
#endif

    if (orig_dir == NULL && !iftDirExists(orig_dir))
        iftError("Original directory does not exist", __func__);
    if (graph_list == NULL)
        iftError("Graph directory/file is NULL", __func__);
    if (param_dir == NULL && !iftDirExists(param_dir))
        iftError("Model directory does not exist", __func__);
    if (out_dir == NULL)
        iftError("Feature directory is NULL", __func__);

    iftFileSet *fs_graphs = iftLoadFileSetFromDirOrCSV(graph_list, 1, 1);
    int ngraphs = fs_graphs->n;
    int nbatches = 0;
    int nremaining_graphs = ngraphs;

    if (fs_graphs->n < 1)
        iftError("Graph list is empty", __func__);

    /* process in batch */
    batchsize = iftMin(batchsize, ngraphs);
    nbatches = ngraphs / batchsize;
    nremaining_graphs = nremaining_graphs - nbatches * batchsize;
    for (int batch = 0; batch < nbatches; batch++)
    {
        int first = batch * batchsize, last = first + batchsize - 1;
        printf("Processing batch %d of %d batches\n", batch + 1, nbatches);
        fflush(stdout);
        FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, out_dir);
    }
    int first = ngraphs - nremaining_graphs, last = ngraphs - 1;
    if (first <= last)
    {
        printf("Processing remaining %d images\n", last - first + 1);
        fflush(stdout);
        FLIMGraphExtractFeaturesFromLayerPerBatch(fs_graphs, first, last, orig_dir, arch, param_dir, layer_index, out_dir);
    }

    iftDestroyFileSet(&fs_graphs);
}

/* --------------------- GFLIM pooling ----------------------------- */

/**
 * @brief Compute the average pooling from a FLIMGraph
 * @param graph FLIMGraph 
 * @param pool_size Size of the pooling
 * @param atrous_factor Atrous factor for the pooling
 * @param stride Stride of the pooling
 */
iftMImage *iftFLIMGraphAtrousAveragePooling(iftFLIMGraph *graph,
                                            int pool_size, 
                                            int atrous_factor, 
                                            int stride)
{
    iftMImage *pool = iftCreateMImage(graph->num_nodes, 1, 1, graph->num_feats);

#pragma omp parallel for
    for (int p = 0; p < graph->num_nodes; p++)
    {
        // find the k most similar neighbors of the node p
        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(p, graph,
                                                        pool_size, atrous_factor,
                                                        true, IFT_FLIMGRAPH_ATTENTION_NONE);

        // aggregate the features of the neighbors 
        double *feature_vector = iftFLIMgraphAggregateFeatures(p, neighbors,
                                                                graph, pool_size,
                                                                IFT_GFLIM_AGGREGATION_CONCAT);
        
        for (int i = 0, j = 0; i < pool_size; i++)
        {
            for (int b = 0; b < graph->num_feats; b++, j++)
                pool->val[p][b] += (float)feature_vector[j];
        }
        iftFree(feature_vector);
        iftDestroyIntArray(&neighbors);

        for (int b = 0; b < graph->num_feats; b++)
            pool->val[p][b] /= (float)graph->num_feats;
        
    }

    if (stride > 1)
        iftWarning("Stride > 1 not implemented yet", __func__);

    return pool;
}

/**
 * @brief Compute the max pooling from a FLIMGraph
 * @param graph FLIMGraph 
 * @param pool_size Size of the pooling
 * @param atrous_factor Atrous factor for the pooling
 * @param stride Stride of the pooling
 */
iftMImage *iftFLIMGraphAtrousMaxPooling(iftFLIMGraph *graph,
                                        int pool_size,
                                        int atrous_factor,
                                        int stride)
{
    iftMImage *pool = iftCreateMImage(graph->num_nodes,1, 1, graph->num_feats);
    iftSetMImage(pool, IFT_INFINITY_FLT_NEG);

    for (int p = 0; p < graph->num_nodes; p++)
    {
        // find the k most similar neighbors of the node p
        iftIntArray *neighbors = iftGetKNeighborsFLIMGraph(p, graph, pool_size,
                                                           atrous_factor, true, 
                                                           IFT_FLIMGRAPH_ATTENTION_NONE);

        // aggregate the features of the neighbors 
        double *feature_vector = iftFLIMgraphAggregateFeatures(p, neighbors,
                                                            graph, pool_size,
                                                            IFT_GFLIM_AGGREGATION_CONCAT);
        for (int i = 0, j = 0; i < pool_size; i++)
        {
            for (int b = 0; b < graph->num_feats; b++, j++)
                pool->val[p][b] = iftMax(pool->val[p][b], (float)feature_vector[j]);
        }
        iftFree(feature_vector);
        iftDestroyIntArray(&neighbors);
    }

    if (stride > 1)
        iftWarning("Stride > 1 not implemented yet", "iftFLIMGraphMaxPooling");

    return pool;
}

/**
 * @fn iftGFLIMPoolingLayer
 * @brief Pooling layer.
 * @param graph Graph in which the pooling is applied.
 * @param pool_type The pooling type (no_pool, avg_pool, max_pool)
 * @param pool_size The pooling size (number of neighbors to apply pooling)
 * @param pool_stride The pooling stride (not implemented yet).
 */
void iftGFLIMPoolingLayer(iftFLIMGraph *graph,
                          char *pool_type, 
                          int pool_size, 
                          int pool_stride)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    if (graph == NULL)
        printf("\t graph: NULL \n");
    else
        printf("\t graph nodes: %ld, feats: %ld, edges: %ld \n", graph->num_nodes, graph->num_feats, graph->num_edges);
    printf("\t pool_type: %s\n", pool_type);
    printf("\t pool_size: %d %d %d\n", pool_size[0], pool_size[1], pool_size[2]);
    printf("\t pool_stride: %d\n", pool_stride);
#endif

    iftMImage *pool = NULL;

    if (strcmp(pool_type, "no_pool") != 0)
    {
        if (strcmp(pool_type, "avg_pool") == 0)
        {
            pool = iftFLIMGraphAtrousAveragePooling(graph,
                                                    pool_size,
                                                    1,
                                                    pool_stride);
        }
        else
        {
            if (strcmp(pool_type, "max_pool") == 0)
            {
                pool = iftFLIMGraphAtrousMaxPooling(graph,
                                                    pool_size,
                                                    1,
                                                    pool_stride);
            }
            else
                iftError("Invalid pooling %s", "pooling", pool_type);
        }

        iftMImageToFLIMGraphInplace(pool, graph);
        iftDestroyMImage(&pool);
    }
}

/* --------------------- GFLIM decoder ----------------------------- */

/**
 * @brief Use the tri-state adaptive decoder to compute weights for channels of a MImage
 * @param mimg MImage to compute the weights
 * @return float* Array of weights for each MImage band
 */
float *TriStateAdaptativeWeigth(iftMImage *mimg)
{
    float *weight = iftAllocFloatArray(mimg->m);
    float *mean = iftAllocFloatArray(mimg->m);
    float *maxval = iftAllocFloatArray(mimg->m);

    /* Normalize activations */
    for (int b = 0; b < mimg->m; b++)
        maxval[b] = iftMMaximumValue(mimg, b);
    for (int p = 0; p < mimg->n; p++)
    {
        for (int b = 0; b < mimg->m; b++)
        {
            if (maxval[b] > 0.0)
                mimg->val[p][b] /= maxval[b];
            else
                mimg->val[p][b] = 0.0;
        }
    }
    iftFree(maxval);

    /* Estimate the mean activation per channel */
    for (int p = 0; p < mimg->n; p++)
    {
        for (int b = 0; b < mimg->m; b++)
            mean[b] += mimg->val[p][b];
    }

    // Just so it is easier to compute the otsu
    iftImage *means_image = iftCreateImage(mimg->m, 1, 1);
    float max_mean = 0.0;
    for (int b = 0; b < mimg->m; b++)
    {
        mean[b] /= mimg->n;
        means_image->val[b] = mean[b] * 255;
        if (means_image->val[b] > max_mean)
            max_mean = means_image->val[b];
    }

    if (max_mean > 0.0)
    {
        // Compute otsu threshold as a way to separate foreground bands from background bands
        float mu = 0.0, stdev = 0.0;
        for (int b = 0; b < mimg->m; b++)
            mu += mean[b];
        mu /= mimg->m;
        int means_otsu = iftOtsu(means_image); 

        // Compute the standard deviation
        for (int b = 0; b < mimg->m; b++)
            stdev += (mean[b] - mu) * (mean[b] - mu);
        stdev = sqrtf(stdev / mimg->m) * sqrtf(stdev / mimg->m);

        // Compute the low and high threshold values
        float low_end, high_end;
        if (means_otsu > 0.1)
            low_end = means_otsu - stdev;
        else
            low_end = means_otsu;
        high_end = means_otsu + stdev;

        /* Estimate the weight of each channel */
        iftImage *band = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
        for (int b = 0; b < mimg->m; b++)
        {
            // Get one activation (a channel/band from mimg) and convert it from 0-1 to 0-255
            for (int p = 0; p < band->n; p++) 
                band->val[p] = mimg->val[p][b] * 255;

            // find the maximum value of the band
            int max_band = iftMaximumValue(band);

            if (max_band > 0.0)
            {
                // Compute another otsu to separate foreground pixels from background pixels
                int otsu_band = iftOtsu(band);

                // Compute the foreground proportion of the band 
                // as the proportion of pixels above otsu_band
                float fg_proportion = 0;
                for (int p = 0; p < band->n; p++)
                {
                    if (band->val[p] >= otsu_band) fg_proportion += 1;
                }
                fg_proportion /= band->n;

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
        for (int b = 0; b < mimg->m; b++)
            weight[b] = 0.0;
    }

    iftFree(mean);
    iftDestroyImage(&means_image);

    return (weight);
}

/**
 * @brief Decode the MImage using the tri-state adaptive decoder
 * @param mimg MImage to decode
 * @return iftFImage* Decoded image
 */
iftFImage *TriStateAdaptativeDecoder(iftMImage *mimg)
{
    float *weight = TriStateAdaptativeWeigth(mimg);

    iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

    /* decode layer */
    for (int p = 0; p < mimg->n; p++)
    {
        for (int b = 0; b < mimg->m; b++)
        {
            salie->val[p] += mimg->val[p][b] * weight[b];
        }
        if (salie->val[p] < 0)
            salie->val[p] = 0; 
    }

    iftFree(weight);

    return (salie);
}

/* --------------------- GFLIM others ----------------------------- */

/**
 * @brief Given an activation directory (source) with files in .mimg or .json format,
 * write the activations' images in destination directory with extension 'ext' (pgm or png)
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param source        File with features. Can be in .mimg or .json format.
 * @param destination   Directory to write the images.
 * @param ext           Extension of the output images (pgm or png).
 * @param labels_dir    Directory with the superpixel labels map (pgm).
 */
void iftWriteActivationsImg(char *source, char *destination, char *ext, char *labels_dir)
{
#ifdef IFT_DEBUG
    printf("%s \n", __func__);
    printf("\t source: %s\n", source == NULL ? "NULL" : source);
    printf("\t destination: %s\n", destination == NULL ? "NULL" : destination);
    printf("\t ext: %s\n", ext == NULL ? "NULL" : ext);
    printf("\t labels_dir: %s\n", labels_dir == NULL ? "NULL" : labels_dir);
#endif

    if (source == NULL || !iftDirExists(source))
        iftError("Source directory does not exist", __func__);
    if (destination == NULL)
        iftError("Destination directory is NULL", __func__);
    if (ext == NULL)
        iftError("Extension is NULL", __func__);
    if (labels_dir == NULL || !iftDirExists(labels_dir))
        iftError("Label directory does not exist", __func__);

    char ext_input[10];

    iftMakeDir(destination);

    iftFileSet *fs_imgs = iftLoadFileSetFromDir(source, 1);
    if (fs_imgs->n < 1)
        iftError("Source directory is empty", __func__);
    for (int i = 0; i < fs_imgs->n; i++)
    {
        sprintf(ext_input, "%s", iftFileExt(fs_imgs->files[i]->path));
        char *basename = iftFilename(fs_imgs->files[i]->path, ext_input);
        if (labels_dir != NULL)
        {
            iftFLIMGraph *graph = iftReadFLIMGraph(fs_imgs->files[i]->path);
            char filename[200];
            sprintf(filename, "%s/%s.pgm", labels_dir, basename);
            iftImage *img = iftReadImageByExt(filename);

            iftImage **activations = iftFLIMGraphToActivations(graph, img);
            iftDestroyImage(&img);

            int num_activations = (int)graph->num_feats;

            sprintf(filename, "%s/%s", destination, basename);
            iftMakeDir(filename);

#pragma omp parallel for shared(activations, num_activations, destination, basename, ext)
            for (int j = 0; j < num_activations; j++)
            {
                char filename2[200];
                sprintf(filename2, "%s/%s/kernel-%03d.%s", destination, basename, j + 1, ext);
                iftWriteImageByExt(activations[j], filename2);
            }

            for (int j = 0; j < num_activations; j++)
                iftDestroyImage(&(activations[j]));

            iftFree(activations);
            iftDestroyFLIMGraph(&graph);
        }
        else
        {
            iftMImage *mimg = iftReadMImage(fs_imgs->files[i]->path);

#pragma omp parallel for
            for (int j = 0; j < mimg->m; j++)
            {
                char filename2[200];
                iftImage *band = iftMImageToImage(mimg, 255, j);
                iftImage *band_resize = NULL;
                if (iftIs3DImage(band))
                    band_resize = iftGetXYSlice(band, band->zsize / 2);
                else
                    band_resize = iftCopyImage(band);
                iftDestroyImage(&band);
                sprintf(filename2, "%s/%s", destination, basename);
                iftMakeDir(filename2);
                sprintf(filename2, "%s/%s/kernel-%03d.%s", destination, basename, j + 1, ext);
                iftWriteImageByExt(band_resize, filename2);
                iftDestroyImage(&band_resize);
            }

            iftDestroyMImage(&mimg);
        }
        iftFree(basename);
    }
    iftDestroyFileSet(&fs_imgs);
}
