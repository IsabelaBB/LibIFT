from skimage.filters import threshold_otsu
import numpy as np
from skimage import io
import os
import argparse

def binarize_saliency(saliency):
    threshold = threshold_otsu(saliency)
    
    # compute std saliency
    std_saliency = np.std(saliency)
    threshold -= std_saliency
    if(threshold < 0):
        threshold = 0

    bin_sal = np.copy(saliency)
    bin_sal[saliency>threshold] = 1
    bin_sal[saliency<=threshold] = 0

    #bin_sal=morphology.area_opening(bin_sal, area_threshold=64, connectivity=2).astype(int)
    #bin_sal=morphology.area_closing(bin_sal, area_threshold=32, connectivity=2).astype(int)
    #bin_sal=morphology.isotropic_erosion(bin_sal, radius=3).astype(int)
    
    return bin_sal

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("input_dir", help="Input directory containing images.", type=str)
parser.add_argument("output_dir", help="Output directory for binarized images.", type=str)
args = parser.parse_args()

imgs_dir = args.input_dir
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)
for i,filename in enumerate(os.listdir(imgs_dir)):
        basename = filename.split(".")[0]

        if os.path.isfile(imgs_dir+'/'+filename) == False:
            print("No result found for ", imgs_dir+'/'+filename)
            continue
        saliency = io.imread(imgs_dir+'/'+filename)

        if(len(saliency.shape) == 3):
            saliency = saliency[:,:,0]

        bin_sal = binarize_saliency(saliency)
        io.imsave(out_dir+'/'+basename+'.png', (bin_sal*255).astype(np.uint8))