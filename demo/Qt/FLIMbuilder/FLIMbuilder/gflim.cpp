#include "gflim.h"
#include <stdlib.h>

GFLIM::GFLIM()
{
    this->images_dir = "";
    this->param_dir = "";
    this->seeds_dir = "";
    this->object_dir = "";
    this->graphs_dir = "";

    this->num_nodes = 1000;
    this->num_init_seeds = 8000;
    this->create_graphs = true;

    this->device = -1;
    this->ift_arch = nullptr;
}

GFLIM::~GFLIM()
{
    iftDestroyFLIMArch(&this->ift_arch);
}

bool GFLIM::IsArchitectureValid()
{
    /* Standard global keys */
    // stdev_factor
    QMessageBox msgbox;
    if (!this->arch_json.contains("stdev_factor"))
    {
        QErrorMessage *message = new QErrorMessage();
        message->showMessage(QString("Failed to load architecture. Key \"stdev_factor\" not found"));
        delete message;
        msgbox.setText(QString("Failed to load architecture. Key \"stdev_factor\" not found"));
        msgbox.exec();
        return false;
    }
    else
    {
        if (!this->arch_json.value(QString("stdev_factor")).isDouble())
        {
            QErrorMessage *message = new QErrorMessage();
            message->showMessage(QString("Stdev_factor must be a float point value."));
            delete message;
            msgbox.setText(QString("Stdev_factor must be a float point value."));
            msgbox.exec();
            return false;
        }
    }
    // nlayers
    if (!this->arch_json.contains("nlayers"))
    {
        QErrorMessage *message = new QErrorMessage();
        message->showMessage(QString("Failed to load architecture. Key \"nlayers\" not found"));
        delete message;
        msgbox.setText(QString("Failed to load architecture. Key \"nlayers\" not found"));
        msgbox.exec();
        return false;
    }
    else
    {
        if (this->arch_json.value(QString("nlayers")).toInt(-1) == -1)
        {
            QErrorMessage *message = new QErrorMessage();
            message->showMessage(QString("Number of layers must be a positive integer."));
            delete message;
            msgbox.setText(QString("Number of layers must be a positive integer."));
            msgbox.exec();
            return false;
        }
    }

    /* Checking layer key structures */
    int nlayers = this->arch_json.value("nlayers").toInt();
    QErrorMessage *message = new QErrorMessage();
    for (int l = 0; l < nlayers; l++)
    {
        if (!this->arch_json.contains(QString("layer%1").arg(l + 1)))
        {
            message->showMessage(QString("Layer %1 missing.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Layer %1 missing.").arg(l + 1));
            msgbox.exec();
            return false;
        }

        QJsonObject jsonLayer = this->arch_json[QString("layer%1").arg(l + 1)].toObject();
        if (!jsonLayer.contains(QString("conv")))
        {
            message->showMessage(QString("Layer %1 missing \"conv\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Layer %1 missing \"conv\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        if (!jsonLayer.contains(QString("relu")))
        {
            message->showMessage(QString("Layer %1 missing \"relu\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Layer %1 missing \"relu\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (!jsonLayer.value(QString("relu")).isBool())
            {
                message->showMessage(QString("Relu key of Layer %1 must be boolean.").arg(l + 1));
                delete message;
                msgbox.setText(QString("Relu key of Layer %1 must be boolean.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonLayer.contains(QString("pooling")))
        {
            message->showMessage(QString("Layer %1 missing \"pooling\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Layer %1 missing \"pooling\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }

        QJsonObject jsonConv = jsonLayer.value(QString("conv")).toObject();
        if (!jsonConv.contains(QString("kernel_size")))
        {
            message->showMessage(QString("Conv structure of layer %1 missing \"kernel_size\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Conv structure of layer %1 missing \"kernel_size\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (!jsonConv.value(QString("kernel_size")).isArray())
            {
                message->showMessage(QString("Kernel size key of Layer %1 must be an array.").arg(l + 1));
                delete message;
                msgbox.setText(QString("Kernel size key of Layer %1 must be an array.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonConv.contains(QString("nkernels_per_marker")))
        {
            message->showMessage(QString("Conv structure of layer %1 missing \"nkernels_per_marker\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Conv structure of layer %1 missing \"nkernels_per_marker\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (jsonConv.value(QString("nkernels_per_marker")).toInt(-1) == -1)
            {
                message->showMessage(QString("The nkernels_per_marker key of Layer %1 must be a positive integer.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The nkernels_per_marker key of Layer %1 must be a positive integer.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonConv.contains(QString("dilation_rate")))
        {
            message->showMessage(QString("Conv structure of layer %1 missing \"dilation_rate\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Conv structure of layer %1 missing \"dilation_rate\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (!jsonConv.value(QString("dilation_rate")).isArray())
            {
                message->showMessage(QString("The dilation_rate key of Layer %1 must be an array.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The dilation_rate key of Layer %1 must be an array.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonConv.contains(QString("nkernels_per_image")))
        {
            message->showMessage(QString("Conv structure of layer %1 missing \"nkernels_per_image\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Conv structure of layer %1 missing \"nkernels_per_image\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (jsonConv.value(QString("nkernels_per_image")).toInt(-1) == -1)
            {
                message->showMessage(QString("The nkernels_per_image key of Layer %1 must be a positive integer.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The nkernels_per_image key of Layer %1 must be a positive integer.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonConv.contains(QString("noutput_channels")))
        {
            message->showMessage(QString("Conv structure of layer %1 missing \"noutput_channels\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Conv structure of layer %1 missing \"noutput_channels\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (jsonConv.value(QString("noutput_channels")).toInt(-1) == -1)
            {
                message->showMessage(QString("The noutput_channels key of Layer %1 must be a positive integer.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The noutput_channels key of Layer %1 must be a positive integer.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }

        QJsonObject jsonPooling = jsonLayer.value(QString("pooling")).toObject();
        if (!jsonPooling.contains(QString("type")))
        {
            message->showMessage(QString("Pooling structure of layer %1 missing \"type\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Pooling structure of layer %1 missing \"type\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (!jsonPooling.value(QString("type")).isString())
            {
                message->showMessage(QString("The type key of Layer %1 must be a string.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The type key of Layer %1 must be a string.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonPooling.contains(QString("size")))
        {
            message->showMessage(QString("Pooling structure of layer %1 missing \"size\" key.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Pooling structure of layer %1 missing \"size\" key.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (!jsonPooling.value(QString("size")).isArray())
            {
                message->showMessage(QString("Pooling size key of Layer %1 must be an array.").arg(l + 1));
                delete message;
                msgbox.setText(QString("Pooling size key of Layer %1 must be an array.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
        if (!jsonPooling.contains(QString("stride")))
        {
            message->showMessage(QString("Stride key of Layer %1 must be an array.").arg(l + 1));
            delete message;
            msgbox.setText(QString("Stride key of Layer %1 must be an array.").arg(l + 1));
            msgbox.exec();
            return false;
        }
        else
        {
            if (jsonPooling.value(QString("stride")).toInt(-1) == -1)
            {
                message->showMessage(QString("The stride key of Layer %1 must be a positive integer.").arg(l + 1));
                delete message;
                msgbox.setText(QString("The stride key of Layer %1 must be a positive integer.").arg(l + 1));
                msgbox.exec();
                return false;
            }
        }
    }

    delete message;
    return true;
}

bool GFLIM::modelExists(int layer)
{
    QDir model_dir(this->param_dir);
    QFileInfoList list = model_dir.entryInfoList(QStringList() << "LayerParams" << QString("conv%1-kernels.npy").arg(layer), QDir::Files);

    if (list.count() == 0)
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

bool GFLIM::TrainLayer(int layer)
{
    // Setting layer input as previous layer output
    QString layer_input;
    QString layer_output;

    if (layer == 1)
        layer_input = this->graphs_dir;
    else
    {
        QDir weight_dir(this->param_dir);
        QFileInfoList weight_list = weight_dir.entryInfoList(QStringList() << "Weights" << QString("conv%1-*").arg(layer - 1), QDir::Files);

        // previous layer has no weights
        if(!this->modelExists(layer-1))
        {
            qWarning("Could not found the previous layer's weights.");
            return FALSE;
        }
        layer_input = this->param_dir + "/layer" + QVariant(layer - 1).toString() + "/";
    }

    layer_output = this->param_dir + "/layer" + QVariant(layer).toString() + "/";

    if (iftDirExists(layer_output.toUtf8().data()))
        iftRemoveDir(layer_output.toUtf8().data());

    iftMakeDir(layer_output.toUtf8().data());

    // Creating architecture file for layer
    QFile *save_arch = new QFile(this->param_dir + QString("arch_layer%1.json").arg(layer));
    if (!save_arch->open(QIODevice::WriteOnly))
    {
        qWarning("Could not open arch file");
        return FALSE;
    }

    QJsonObject json_layer;
    json_layer.insert("stdev_factor", this->arch_json.value(QString("stdev_factor")));
    json_layer.insert("nlayers", 1);
    json_layer.insert("apply_intrinsic_atrous", this->arch_json.value(QString("apply_intrinsic_atrous")));
    json_layer.insert("layer1", this->arch_json.value(QString("layer") + QVariant(layer).toString()));

    QJsonDocument *save = new QJsonDocument(json_layer);
    save_arch->write(save->toJson());
    save_arch->close();

    // Reading architecture
    iftFLIMArch *tmp_arch = iftReadFLIMArch((this->param_dir + QString("arch_layer%1.json").arg(layer)).toUtf8().data());

    printf("input:%s seeds_dir:%s model:%s layer:%d output:%s \n", layer_input.toUtf8().data(),
           this->seeds_dir.toUtf8().data(),
           this->param_dir.toUtf8().data(),
           layer,
           layer_output.toUtf8().data());

    // char *activ_dir, char *markers_dir, char *param_dir, int layer_index, iftFLIMArch *arch, char *output_dir
    iftGFLIMTrainLayer(layer_input.toUtf8().data(),
                       this->seeds_dir.toUtf8().data(),
                       this->param_dir.toUtf8().data(),
                       layer,
                       tmp_arch,
                       layer_output.toUtf8().data());


    // Writting architecture in case it changed
    delete save;
    delete save_arch;
    this->arch_path = this->param_dir + "/arch.json";

    this->arch_json[QString("layer%1").arg(layer)].toObject()["noutput_channels"] = tmp_arch->layer[layer].noutput_channels;
    save_arch = new QFile(this->arch_path);

    if (!save_arch->open(QIODevice::WriteOnly))
    {
        qWarning("Could not open arch file");
        return FALSE;
    }

    save = new QJsonDocument(this->arch_json);
    save_arch->write(save->toJson());
    save_arch->close();
    delete save;

    iftDestroyFLIMArch(&tmp_arch);

    return TRUE;
}

/**
 * @brief Convert a graph (in json format) to a .mimg file using the superpixel labels map as auxiliar.
 * @author  Isabela Borlido.
 * @date    May, 2021.
 * @param graph_dir    Directory with graph files (in json format).
 * @param labels_dir   Directory with the superpixel labels map.
 * @param output_dir   Directory to write the .mimg file.
 * @note The labels must be in .pgm or .ppm format
 * @note The graph must be in .json format
 */
void GFLIM::GraphToMIMG(char *graph_dir, char *labels_dir, char *output_dir)
{
    char ext[10];
    iftFileSet *fs_graphs = iftLoadFileSetFromDir(graph_dir, 1);
    iftFileSet *fs_labels = iftLoadFileSetFromDir(labels_dir, 1);

    if (fs_graphs->n > fs_labels->n)
    {
        printf("Error: number of graphs is greater than number of labels\n");
        exit(1);
    }

    sprintf(ext, "%s", iftFileExt(fs_labels->files[0]->path));

    for (int i = 0; i < fs_graphs->n; i++)
    {
        char *basename = NULL;
        char filename[200], label_file[200];

        basename = iftFilename(fs_graphs->files[i]->path, ".json");

        sprintf(label_file, "%s/%s%s", labels_dir, basename, ext);

        iftFLIMGraph *graph = iftReadFLIMGraph(fs_graphs->files[i]->path);
        iftImage *label_img = iftReadImageByExt(label_file);
        iftMImage *features = iftFLIMGraphToMImage(graph, label_img);

        sprintf(filename, "%s/%s.mimg", output_dir, basename);
        iftWriteMImage(features, filename);
        iftDestroyFLIMGraph(&graph);
        iftDestroyImage(&label_img);
    }
}

bool GFLIM::ExtractFeaturesFromLayer(int layer)
{

    // Checking if model exists
    QDir weight_dir(this->param_dir);
    QFileInfoList weight_list = weight_dir.entryInfoList(QStringList() << "Weights" << QString("conv%1-*").arg(layer), QDir::Files);
    if (weight_list.empty())
    { // previous layer has no weights
        this->TrainLayer(layer);
    }

    // Setting layer input as previous layer output
    QString layer_input;
    QString layer_image_list;
    if (layer == 1)
    {
        layer_input = this->graphs_dir;
        layer_image_list = this->param_dir + "input.csv";
    }
    else
    {
        layer_input = this->param_dir + "graph_activation" + QVariant(layer - 1).toString() + "/";
        layer_image_list = this->param_dir + "activation" + QVariant(layer - 1).toString() + ".csv";
    }

    QDir dir(layer_input);
    QFileInfoList list = dir.entryInfoList(QStringList() << "Activations" << "*", QDir::Files);
    QDir image_dir(this->graphs_dir);
    QFileInfoList list_orig = image_dir.entryInfoList(QStringList() << "Images" << "*", QDir::Files);

    if (list.count() < list_orig.count())
    {
        this->ExtractFeaturesFromLayer(layer - 1);
        dir = QDir(layer_input);
        list = dir.entryInfoList(QStringList() << "Activations" << "*", QDir::Files); // update list to get new activations
    }

    // Creating list of images
    QFile images_csv(layer_image_list);
    if (images_csv.open(QIODevice::WriteOnly))
    {
        foreach (QFileInfo filename, list)
        {
            images_csv.write((filename.fileName() + "\n").toUtf8().data());
        }
    }
    images_csv.close();

    // I have to create a char pointer to the object dir due to the fact that the iftFLIMExtractFeatures function checks if the object dir is null. If, an empty string ("") is
    // sent as argument, it generates a bug.
    char *object_dir_ptr = nullptr;
    if (this->object_dir != "")
    {
        object_dir_ptr = iftCopyString(this->object_dir.toUtf8().data());
    }

    // Creating output dir for layer
    QString activation_dir = this->param_dir + "graph_activation" + QVariant(layer).toString() + "/";
    if (iftDirExists(activation_dir.toUtf8().data()))
    {
        iftRemoveDir(activation_dir.toUtf8().data());
        iftMakeDir(activation_dir.toUtf8().data());
    }
    else
    {
        iftMakeDir(activation_dir.toUtf8().data());
    }

    // Creating architecture file for layer
    QJsonObject *json_layer = new QJsonObject();
    json_layer->insert("stdev_factor", this->arch_json.value(QString("stdev_factor")));
    json_layer->insert("nlayers", 1);
    json_layer->insert("layer1", this->arch_json.value(QString("layer") + QVariant(layer).toString()));
    QJsonDocument save(*json_layer);
    QFile save_arch(this->param_dir + QString("arch_layer%1.json").arg(layer));
    if (!save_arch.open(QIODevice::WriteOnly))
    {
        qWarning("Could not open arch file");
        return FALSE;
    }
    save_arch.write(save.toJson());
    save_arch.close();
    delete json_layer;

    // Reading architecture
    iftFLIMArch *tmp_arch = iftReadFLIMArch((this->param_dir + QString("arch_layer%1.json").arg(layer)).toUtf8().data());

    // Extracting Features
    iftFLIMGraphExtractFeaturesFromLayer(layer_input.toUtf8().data(),
                                         layer_image_list.toUtf8().data(),
                                         tmp_arch,
                                         this->param_dir.toUtf8().data(),
                                         layer,
                                         activation_dir.toUtf8().data(),
                                         64);

    iftDestroyFLIMArch(&tmp_arch);

    if (object_dir_ptr != nullptr)
        free(object_dir_ptr);

    // Verify if layer<n> exists, which is necessary to visualize activations
    QString layer_output, activation_img;
    layer_output = this->param_dir + "/layer" + QVariant(layer).toString() + "/";
    if (!iftDirExists(layer_output.toUtf8().data()))
    {
        iftMakeDir(layer_output.toUtf8().data());
    }

    activation_img = this->param_dir + "activation" + QVariant(layer).toString() + "/";
    if (!iftDirExists(activation_img.toUtf8().data()))
    {
        iftMakeDir(activation_img.toUtf8().data());
    }

    this->GraphToMIMG(activation_dir.toUtf8().data(),
                      this->labels_path.toUtf8().data(),
                      activation_img.toUtf8().data());

    return TRUE;
}

bool GFLIM::ExtractFeaturesFromLayerForTraining(int layer)
{
    // Checking if model exists
    QDir weight_dir(this->param_dir);
    QFileInfoList weight_list = weight_dir.entryInfoList(QStringList() << "Weights" << QString("conv%1-*").arg(layer), QDir::Files);
    if (weight_list.empty())
    { // previous layer has no weights
        this->TrainLayer(layer);
    }

    // Setting layer input as previous layer output
    QString layer_input;
    QString layer_image_list;
    if (layer == 1)
    {
        layer_input = this->graphs_dir;
        layer_image_list = this->param_dir + "input.csv";
    }
    else
    {
        layer_input = this->param_dir + "layer" + QVariant(layer - 1).toString() + "/";
        layer_image_list = this->param_dir + "layer" + QVariant(layer - 1).toString() + ".csv";
    }

    QDir dir(layer_input);
    QFileInfoList list = dir.entryInfoList(QStringList() << "Activations" << "*", QDir::Files);
    QDir image_dir(this->graphs_dir);
    QFileInfoList list_orig = image_dir.entryInfoList(QStringList() << "Images" << "*", QDir::Files);

    if (list.count() < list_orig.count())
    {
        this->ExtractFeaturesFromLayer(layer - 1);
        dir = QDir(layer_input);
        list = dir.entryInfoList(QStringList() << "Activations" << "*", QDir::Files); // update list to get new activations
    }

    // Creating list of images
    QFile images_csv(layer_image_list);
    if (images_csv.open(QIODevice::WriteOnly))
    {
        foreach (QFileInfo filename, list)
        {
            images_csv.write((filename.fileName() + "\n").toUtf8().data());
        }
    }
    images_csv.close();

    // I have to create a char pointer to the object dir due to the fact that the iftFLIMExtractFeatures function checks if the object dir is null. If, an empty string ("") is
    // sent as argument, it generates a bug.
    char *object_dir_ptr = nullptr;
    if (this->object_dir != "")
    {
        object_dir_ptr = iftCopyString(this->object_dir.toUtf8().data());
    }

    // Creating output dir for layer
    QString activation_dir = this->param_dir + "layer" + QVariant(layer).toString() + "/";
    if (iftDirExists(activation_dir.toUtf8().data()))
    {
        iftRemoveDir(activation_dir.toUtf8().data());
        iftMakeDir(activation_dir.toUtf8().data());
    }
    else
    {
        iftMakeDir(activation_dir.toUtf8().data());
    }

    // Creating architecture file for layer
    QJsonObject *json_layer = new QJsonObject();
    json_layer->insert("stdev_factor", this->arch_json.value(QString("stdev_factor")));
    json_layer->insert("nlayers", 1);
    json_layer->insert("layer1", this->arch_json.value(QString("layer") + QVariant(layer).toString()));
    QJsonDocument save(*json_layer);
    QFile save_arch(this->param_dir + QString("arch_layer%1.json").arg(layer));
    if (!save_arch.open(QIODevice::WriteOnly))
    {
        qWarning("Could not open arch file");
        return FALSE;
    }
    save_arch.write(save.toJson());
    save_arch.close();
    delete json_layer;

    // Reading architecture
    iftFLIMArch *tmp_arch = iftReadFLIMArch((this->param_dir + QString("arch_layer%1.json").arg(layer)).toUtf8().data());
    // Changing pooling stride to 1
    tmp_arch->layer[0].pool_stride = 1;

    // Extracting Features
    iftFLIMGraphExtractFeaturesFromLayer(layer_input.toUtf8().data(),
                                         layer_image_list.toUtf8().data(),
                                         tmp_arch,
                                         this->param_dir.toUtf8().data(),
                                         layer,
                                         activation_dir.toUtf8().data(),
                                         64);

    iftDestroyFLIMArch(&tmp_arch);

    if (object_dir_ptr != nullptr)
        free(object_dir_ptr);

    // Verify if layer<n> exists, which is necessary to visualize activations
    QString layer_output, activation_img;
    layer_output = this->param_dir + "/layer" + QVariant(layer).toString() + "/";
    if (!iftDirExists(layer_output.toUtf8().data()))
    {
        iftMakeDir(layer_output.toUtf8().data());
    }

    activation_img = this->param_dir + "activation" + QVariant(layer).toString() + "/";
    if (!iftDirExists(activation_img.toUtf8().data()))
    {
        iftMakeDir(activation_img.toUtf8().data());
    }

    this->GraphToMIMG(activation_dir.toUtf8().data(),
                      this->labels_path.toUtf8().data(),
                      activation_img.toUtf8().data());

    return TRUE;
}

void GFLIM::WriteSeedsFromImage(iftImage *markers_img, char *seeds_name, char *spx_name_file)
{

    iftLabeledSet *seeds = nullptr;
    iftImage *spx_img = iftReadImageByExt(spx_name_file); // labeled (superpixels) image
    int *foreground_count = (int *)calloc(this->num_nodes, sizeof(int));
    int *background_count = (int *)calloc(this->num_nodes, sizeof(int));

    for (int p = 0; p < markers_img->n; p++)
    {
        if (markers_img->val[p] > 0)
        {
            if (markers_img->val[p] == 1)
                background_count[spx_img->val[p] - 1]++;
            else
                foreground_count[spx_img->val[p] - 1]++;
        }
    }
    iftDestroyImage(&spx_img);

    for (int p = 0; p < this->num_nodes; p++)
    {
        if (foreground_count[p] > 0 || background_count[p] > 0)
        {
            if (foreground_count[p] > background_count[p])
                iftInsertLabeledSet(&seeds, p, 1);
            else
                iftInsertLabeledSet(&seeds, p, 0);
        }
    }
    free(foreground_count);
    free(background_count);
    iftWriteSeedsGraph(seeds, seeds_name);
    iftDestroyLabeledSet(&seeds);
}

QString GFLIM::ExtractFeaturesWithPython()
{
    this->arch_path = this->param_dir + "/arch.json";

    /* Saving arch.json so we can read it as an IFT structure */
    QJsonDocument save(this->arch_json);
    QFile save_arch(this->arch_path);
    if (!save_arch.open(QIODevice::WriteOnly))
    {
        qWarning("Could not open arch file");
        return "Could not open arch file";
    }
    save_arch.write(save.toJson());
    save_arch.close();

    /* Training */
    // QStringList arguments { "feature_extraction.py", "-i "+this->graphs_dir, "-a "+this->arch_path, "-o "+this->param_dir};
    // qDebug() << arguments;
    QString script_path = QCoreApplication::applicationDirPath() + "../FLIMbuilder/feature_extraction";
    QStringList arguments{script_path, " -i " + this->graphs_dir, " -a " + this->arch_path, " -o " + this->param_dir};
    QProcess p;
    p.start("python", arguments);
    qDebug() << p.errorString();
    qDebug() << p.error();
    p.waitForFinished();

    return p.readAllStandardOutput();
}

void GFLIM::ExtractFeaturesFromGraphLayer(int layer)
{

    // converte o .json de saída de iftGFLIMTrainLayer para .mimg
    QDir dir(this->param_dir + "/layer" + QVariant(layer).toString() + "/");
    QFileInfoList list = dir.entryInfoList(QStringList() << "Layer" << "*.json", QDir::Files);

    foreach (QFileInfo filename, list)
    {

        QString basename = filename.baseName();

        QString label_name = this->labels_path + "/" + filename.baseName() + ".pgm";

        iftFLIMGraph *graph = iftReadFLIMGraph(filename.absoluteFilePath().toUtf8().data());

        iftImage *label_img = iftReadImageByExt(label_name.toUtf8().data());

        iftMImage *features = iftFLIMGraphToMImage(graph, label_img);

        QString features_file = this->param_dir + "/layer" + QVariant(layer).toString() + "/" + filename.baseName() + ".mimg";
        iftWriteMImage(features, features_file.toUtf8().data());

        iftDestroyFLIMGraph(&graph);
        iftDestroyImage(&label_img);
    }
}

/**
 * @brief Convert an image to a FLIMGraph
 * @param image_dir image file path
 * @param labels_dir superpixel images file path
 * @param graph_dir graph file path
 * @param num_init_seeds Number of initial seeds (used in DISF)
 * @param num_superpixels Number of superpixels / nodes in the graph
 * @note The image must be in .pgm or .ppm format
 */
void GFLIM::ConvertImageToFLIMGraph(char *image_dir,
                                    char *labels_dir,
                                    char *graph_dir,
                                    int num_init_seeds,
                                    int num_superpixels)
{
    iftImage *img, *label_img = NULL;

    if (num_superpixels >= num_init_seeds)
        iftError("Number of superpixels >= initial seeds", "ImageToFLIMGraph");

    img = iftReadImageByExt(image_dir);

    iftFLIMGraph *graph = iftImageToFLIMGraph(img, num_superpixels, num_init_seeds, &label_img,
                                            IFT_FLIMGRAPH_FEATURE_MEAN, IFT_FLIMGRAPH_TYPE_DISF);

    if (labels_dir != NULL)
        iftWriteImageByExt(label_img, labels_dir);

    iftDestroyImage(&label_img);
    iftDestroyImage(&img);

    if (graph_dir != NULL)
        iftWriteFLIMGraph(graph, graph_dir);

    iftDestroyFLIMGraph(&graph);
}

bool GFLIM::CreateGraph(QString images_path, QString basename)
{

    QString graph_dir = this->graphs_dir + "/" + basename + ".json";
    QString labels_dir = this->labels_path + "/" + basename + ".pgm";

    // Checking if the training images exists
    if (!iftFileExists(images_path.toUtf8().data()))
        return false;

    // Checking if the graphs' dir exists
    if (!iftDirExists(this->graphs_dir.toUtf8().data()))
    {
        iftMakeDir(this->graphs_dir.toUtf8().data());
    }

    if (iftFileExists(graph_dir.toUtf8().data()) &&
        iftFileExists(labels_dir.toUtf8().data()) &&
        !this->create_graphs)
    {
        return true;
    }

    this->ConvertImageToFLIMGraph(images_path.toUtf8().data(),
                                  labels_dir.toUtf8().data(),
                                  graph_dir.toUtf8().data(),
                                  this->num_init_seeds,
                                  this->num_nodes);
    return true;
}

bool GFLIM::CreateGraphs(QString images_path)
{

    // Checking if the graphs' dir exists
    if (!iftDirExists(images_path.toUtf8().data()))
        return false;

    QDir dir_images(images_path);
    QFileInfoList list_images = dir_images.entryInfoList(QStringList() << "Images" << "*", QDir::Files);

    if (list_images.count() <= 0)
        return false;
    if (!this->create_graphs)
        return true;

    QString graph_dir = this->graphs_dir + "/";
    QString labels_dir = this->labels_path + "/";

    foreach (QFileInfo filename, list_images)
    {
        // create links to the original images
        QFile f(filename.absoluteFilePath().toUtf8().data());

        if (!iftIsValidFormat(filename.absoluteFilePath().toUtf8().data()))
        {
            list_images.removeOne(filename);
            continue;
        }
        CreateGraph(filename.absoluteFilePath(), filename.baseName());
    }
    this->create_graphs = false;
    return true;
}
