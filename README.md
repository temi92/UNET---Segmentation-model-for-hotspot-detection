# UNET Segmentation-model-for-hotspot-detection

# Description

Real time hot spot/fire detection from aerial images. 
The approach consist of using segementation models to  hotspots in images

## Installation 
Dependencies can be installed using a conda enviroment with the ```enviroment.yml``` as follows
```bash
conda env create -n hotspot -f environment.yml
conda activate hotspot
```

## Mask Creation

Semantic Segmentation techniques require pixel wise annotations/masks. Labelling is often very tedious in order to help with this
Otsu's Adaptive threshold technique is used to provide the the image mask for training. This can be used as follows:
```bash
cd auto_annotations/
python save_masks.py -h

positional arguments:
  input_dir   directory to input images
  output_dir  created masks

optional arguments:
  -h, --help  show this help message and exit


```
After running this script you should have your generate image masks.