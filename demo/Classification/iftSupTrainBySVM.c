#include "ift.h"

/* Modified by A.X. Falcão (Jan 2018) */

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
      iftError("Usage: iftSupTrainBySVM <...>\n"
	       "[1] input_dataset.zip\n"
	       "[2] kernel_type (0=Linear,1=RBF)\n"
	       "[3] multiclass_type (0=OVO,1=OVA)\n"
	       "[4] constant C (e.g., 1e2, 1e3, 1e5)\n"
	       "[5] kernelization (0-NO, 1-YES)\n"
	       "[6] output_classifier.zip\n", "main");
    }

    iftDataSet *Z   = iftReadDataSet(argv[1]);
		
    int kernel_type = atoi(argv[2]);
    int multiclass  = atoi(argv[3]);
    float C         = atof(argv[4]);
    int kernelize   = atoi(argv[5]);

    /* print input parameters */
    /*printf("- kernel: %s, multiclass: %s, constant: %.2e, kernelization: %s\n", kernel_type == IFT_LINEAR ? "Linear" : "RBF",
                                                                                multiclass == IFT_OVO ? "OVO" : "OVA",
                                                                                C,
                                                                                kernelize == 1 ? "Yes" : "No");*/
    printf("- nsamples: %d, nfeats: %d, nclasses: %d\n", Z->nsamples, Z->nfeats, Z->nclasses);

    iftSVM *svm = NULL;

    float sigma = 1.0 / Z->nfeats;

    svm         = iftCreateSVM(kernel_type, multiclass, C, sigma); // create SVM
    svm->params->nr_weight = Z->nclasses;
    svm->params->weight_label = iftAllocIntArray(Z->nclasses);
    svm->params->weight = iftAllocDoubleArray(Z->nclasses);

    int min_class = IFT_INFINITY_INT;
    for(int i = 0; i < Z->nsamples; i++) {
      if(Z->sample[i].truelabel < min_class) {
        min_class = Z->sample[i].truelabel;
      }
    }

    for(int i = 0; i < Z->nsamples; i++) {
      svm->params->weight[Z->sample[i].truelabel - min_class] += 1.0;
    }
    for(int i = 0; i < Z->nclasses; i++) {
      svm->params->weight_label[i] = i+min_class;
      svm->params->weight[i] = (double)Z->ntrainsamples / (Z->nclasses * svm->params->weight[i]);
    }

    timer *t1 = iftTic();
    // kernelization is implemented only for the Linear SVM
    if (kernelize && (kernel_type == 0))
    {
      uchar traceNormalize = 0;
      svm->kernelization = kernelize;
      float ktrace;
      iftDataSet *Zaux = iftKernelizeDataSet(Z, Z, kernel_type, traceNormalize, &ktrace);
      if (multiclass == IFT_OVO)
        iftSVMTrainOVO(svm, Zaux, Z);
      else
        iftSVMTrainOVA(svm, Zaux, Z);

      iftDestroyDataSet(&Zaux);
    }
    else
    {
      iftSVMTrain(svm, Z);
    }

    /*
    printf("svm->params->svm_type = %d\n", svm->params->svm_type);
    printf("svm->params->kernel_type = %d\n", svm->params->kernel_type);
    printf("svm->params->degree = %d\n", svm->params->degree); // for poly 
    printf("svm->params->gamma = %.2e\n", svm->params->gamma); // for poly/rbf/sigmoid 
    printf("svm->params->coef0 = %.2e\n", svm->params->coef0); // for poly/sigmoid 
    printf("svm->params->cache_size = %.2e\n", svm->params->cache_size); // in MB 
    printf("svm->params->eps = %.2e\n", svm->params->eps); // stopping criteria 
    printf("svm->params->C = %.2e\n", svm->params->C); // for C_SVC, EPSILON_SVR and NU_SVR 
    printf("svm->params->nr_weight = %d\n", svm->params->nr_weight); // for C_SVC 
    printf("svm->params->nu = %.2e\n", svm->params->nu); // for NU_SVC, ONE_CLASS, and NU_SVR 
    printf("svm->params->p = %.2e\n", svm->params->p); // for EPSILON_SVR 
    printf("svm->params->shrinking = %d\n", svm->params->shrinking); // use the shrinking heuristics 
    printf("svm->params->probability = %d\n", svm->params->probability); // do probability estimates 
    printf("svm->nmodels = %d\n", svm->nmodels);
    printf("svm->multiClass = %d\n", svm->multiClass);
    printf("svm->kernelization = %d\n", svm->kernelization);
    printf("svm->nr_weight = %d\n", svm->params->nr_weight);
    printf("svm->weight_label: ");
    if (svm->params->weight_label != NULL) {
      for (int i = 0; i < svm->params->nr_weight; i++)
        printf("%d ", svm->params->weight_label[i]);
    }
    printf("\n");
    printf("svm->weight: ");
    if (svm->params->weight != NULL) {
      for (int i = 0; i < svm->params->nr_weight; i++)
        printf("%.2e ", svm->params->weight[i]);
    }
    printf("\n");
    */

    iftWriteSVM(svm, argv[6]);
    iftDestroySVM(svm);
    iftDestroyDataSet(&Z);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(t1, iftToc())));

    return (0);
}
